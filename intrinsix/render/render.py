"""
Adapted from https://github.com/jingsenzhu/IndoorInverseRendering/blob/main/lightnet/models/render/__init__.py
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional

from .brdf import pdf_ggx, eval_ggx, pdf_diffuse, eval_diffuse, pdf_disney, eval_disney


class IIR_RenderLayer(nn.Module):
    def __init__(self,
                 imWidth = 320,
                 imHeight = 240,
                 fov=85,
                 cameraPos = [0, 0, 0],
                 brdf_type = "ggx",
                 spp = 1,
                 double_sided=True,
                 use_specular=False,):
        super().__init__()
        self.imHeight = imHeight
        self.imWidth = imWidth

        self.use_specular = use_specular
        self.double_sided = double_sided

        self.fov = fov/180.0 * np.pi
        self.cameraPos = np.array(cameraPos, dtype=np.float32).reshape([1, 3, 1, 1])
        self.xRange = 1 * np.tan(self.fov/2)
        self.yRange = float(imHeight) / float(imWidth) * self.xRange
        x, y = np.meshgrid(np.linspace(-self.xRange, self.xRange, imWidth),
                np.linspace(-self.yRange, self.yRange, imHeight ) )
        y = np.flip(y, axis=0)
        z = -np.ones( (imHeight, imWidth), dtype=np.float32)

        pCoord = np.stack([x, y, z]).astype(np.float32)
        pCoord = pCoord[np.newaxis, :, :, :]
        v = self.cameraPos - pCoord
        v = v / np.sqrt(np.maximum(np.sum(v*v, axis=1), 1e-12)[:, np.newaxis, :, :] )
        v = v.astype(dtype = np.float32)

        v = torch.from_numpy(v)
        pCoord = torch.from_numpy(pCoord)

        up = torch.Tensor([0,1,0])
        # assert(brdf_type in ["disney", "ggx"])
        self.brdf_type = brdf_type
        self.spp = spp

        self.register_buffer('v', v, persistent=False)
        self.register_buffer('pCoord', pCoord, persistent=False)
        self.register_buffer('up', up, persistent=False)

    def forward(
            self,
            lighting_model: nn.Module,
            albedo: torch.Tensor,
            rough: torch.Tensor,
            metal: torch.Tensor,
            normal: torch.Tensor,
    ):
        """
        Render according to material, normal and lighting conditions
        Params:
            model: NeRF model to predict lights
            im, albedo, normal, rough, metal, vpos: (bn, c, h, w)
        """
        bn, _, row, col = albedo.shape
        # assert (bn == 1)  # Only support bn = 1
        assert (row == self.imHeight and col == self.imWidth), f"row: {row}, col: {col}, imHeight: {self.imHeight}, imWidth: {self.imWidth}"

        # No clipping
        center_x = col // 2
        center_y = row // 2
        radius = max(center_x, center_y)

        #############################################
        ########## Incident Sampling Start ##########
        #############################################

        cx, cy, cz = create_frame(normal)
        fx, fy, fz = torch.split(torch.stack([cx, cy, cz], dim=1).permute(0,2,1,3,4),1, dim=1)
        fx, fy, fz = fx.squeeze(1), fy.squeeze(1), fz.squeeze(1)

        # ============ W_i - Direction to the camera =================
        wi_world = self.v
        wi = (wi_world[:, 0:1, ...] * fx +
              wi_world[:, 1:2, ...] * fy +
              wi_world[:, 2:3, ...] * fz)
        if self.double_sided:
            wi[:, 2, ...] = torch.abs(wi[:, 2, ...].clone())

        wi_mask = torch.where(wi[:, 2:3, ...] < 1e-6, torch.zeros_like(wi[:, 2:3, ...]),
                              torch.ones_like(wi[:, 2:3, ...]))

        wi[:, 2, ...] = torch.clamp(wi[:, 2, ...].clone(), min=1e-3)
        wi = functional.normalize(wi, dim=1, eps=1e-6)
        wi = wi.unsqueeze(1)  # (bn, 1, 3, h, w)

        # Clipping
        left = max(center_x - radius, 0)
        right = center_x + radius
        top = max(center_y - radius, 0)
        bottom = center_y + radius
        wi = wi[:, :, :, top:bottom, left:right]
        wi_mask = wi_mask[:, :, top:bottom, left:right]
        cx = cx[:, :, top:bottom, left:right]
        cy = cy[:, :, top:bottom, left:right]
        cz = cz[:, :, top:bottom, left:right]
        albedo_clip = albedo[:, :, top:bottom, left:right]
        metal_clip = metal[:, :, top:bottom, left:right]
        rough_clip = rough[:, :, top:bottom, left:right]

        irow = wi.size(3)
        icol = wi.size(4)

        # ============ W_o - Direction to the light =================
        wo_emitter = lighting_model.sample_direction(vpos=None, normal=normal.unsqueeze(1))
        wo = (wo_emitter[:, :, 0:1, ...] * fx.unsqueeze(1) +
              wo_emitter[:, :, 1:2, ...] * fy.unsqueeze(1) +
              wo_emitter[:, :, 2:3, ...] * fz.unsqueeze(1))
        if self.double_sided:
            wo[:, :, 2, ...] = torch.abs(wo[:, :, 2, ...].clone())

        # Convert to world space
        direction = wo_emitter
        direction = direction.permute(0, 1, 3, 4, 2)  # (bn, spp, h, w, 3)
        direction = direction.permute(1, 0, 2, 3, 4)  # (spp, bn, h, w, 3)
        direction = direction.reshape(lighting_model.spp, -1, 3)

        # W_o BRDF evaluation
        if self.brdf_type == "ggx":
            pdfs = pdf_ggx(albedo_clip, rough_clip, metal_clip, wi, wo).unsqueeze(2)
            eval_diff, eval_spec, mask = eval_ggx(albedo_clip, rough_clip, metal_clip, wi, wo)
        elif self.brdf_type == "diffuse":
            pdfs = pdf_diffuse(wi, wo)
            eval_diff, eval_spec, mask = eval_diffuse(albedo_clip, wi, wo)
        else:
            pdfs = pdf_disney(rough_clip, metal_clip, wi, wo).unsqueeze(2)
            eval_diff, eval_spec, mask = eval_disney(albedo_clip, rough_clip, metal_clip, wi, wo)

        # Since we are using emitter sampling, the sampling pdf is 1
        pdfs_brdf = torch.ones_like(pdfs)
        pdfs_brdf = torch.clamp(pdfs_brdf, min=0.001)

        brdfDiffuse = eval_diff.expand([bn, lighting_model.spp, 3, irow, icol]) / pdfs_brdf
        brdfSpec = eval_spec.expand([bn, lighting_model.spp, 3, irow, icol]) / pdfs_brdf
        # del ndl, pdfs, eval_diff, eval_spec
        #############################################
        ########### Incident Sampling End ###########
        #############################################

        light = lighting_model(direction=direction)

        pdf_emitter = lighting_model.pdf_direction(vpos=None, direction=wo_emitter)
        pdf_emitter = torch.clamp(pdf_emitter, min=0.001)
        ndl = torch.clamp(wo[:, :, 2:, ...], min=0)

        # light = get_light_chunk(model, im, model_kwargs, direction.size(0), self.chunk)
        if light.shape[1] == 1:
            light = light.view(1, lighting_model.spp, 1, 1, -1)
            light = light.expand(1, lighting_model.spp, irow, icol, -1)
        else:
            light = light.view(1, lighting_model.spp, irow, icol, -1)
        light = light.permute(0, 1, 4, 2, 3)  # (bn, spp, 3, h, w)

        light = light * ndl / pdf_emitter

        colorDiffuse = torch.sum(brdfDiffuse * light, dim=1)
        if self.use_specular:
            colorSpec = torch.sum(brdfSpec * light, dim=1)
        else:
            colorSpec = torch.zeros_like(colorDiffuse)

        shading = light
        shading = torch.sum(shading, dim=1)
        ##############################
        ####### Integrator End #######
        ##############################

        return colorDiffuse, colorSpec, wi_mask, shading


def create_frame(n: torch.Tensor, eps:float = 1e-6):
    """
    Generate orthonormal coordinate system based on surface normal
    [Duff et al. 17] Building An Orthonormal Basis, Revisited. JCGT. 2017.
    :param: n (bn, 3, ...)
    """
    z = functional.normalize(n, dim=1, eps=eps)
    sgn = torch.where(z[:,2,...] >= 0, 1.0, -1.0)
    a = -1.0 / (sgn + z[:,2,...])
    b = z[:,0,...] * z[:,1,...] * a
    x = torch.stack([1.0 + sgn * z[:,0,...] * z[:,0,...] * a, sgn * b, -sgn * z[:,0,...]], dim=1).to(n.dtype)
    y = torch.stack([b, sgn + z[:,1,...] * z[:,1,...] * a, -z[:,1,...]], dim=1).to(n.dtype)
    return x, y, z
