import einops
import numpy as np
import torch
from torch import nn
from einops import einsum, rearrange

# ====================== LIGHTING ======================

class ConstantIntensity(nn.Module):
    def __init__(self,
                 value,
                 exp_val=True):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.float32))
        self.exp_val = exp_val

    def forward(self, direction):
        val = self.value
        if self.exp_val:
            val = torch.exp(val)
        return val.unsqueeze(0).expand_as(direction)[..., :1]

    def reg_loss(self):
        val = self.value
        if self.exp_val:
            val = torch.exp(val)
        return torch.sum(val)

    def sample_direction(self, vpos, normal):
        # (bn, spp, 3, h, w)
        return normal

    @property
    def spp(self):
        return 1
    
class DirectionalLight_LatLong(nn.Module):
    def __init__(self,
                 ch=1,
                 solid_angle=1,
                 weight_init=0):
        super().__init__()

        self.Num = 1

        self.ch = ch
        self.solid_angle = np.cos(np.deg2rad(solid_angle))

        self.weight, self.theta, self.phi = self.init_pos(weight_init=weight_init)

        is_enabled = torch.tensor(True)
        self.register_buffer('is_enabled', is_enabled)

    def init_pos(self, weight_init=0):
        weight = nn.Parameter(-torch.ones((1, self.ch), dtype=torch.float32) + weight_init)
        theta = nn.Parameter(torch.ones((1, 1), dtype=torch.float32) * torch.pi / 2)
        phi = nn.Parameter(torch.zeros((1, 1), dtype=torch.float32))

        return weight, theta, phi

    def deparameterize(self):
        theta = self.theta
        phi = self.phi

        weight = self.deparameterize_weight()

        return weight, theta, phi

    def deparameterize_weight(self):
        weight = torch.exp(self.weight)
        return weight

    def get_axis(self, theta, phi):
        # Get axis
        axisX = torch.sin(theta) * torch.sin(phi)
        axisY = torch.cos(theta)
        axisZ = -torch.sin(theta) * torch.cos(phi)

        axis = torch.cat([axisX, axisY, axisZ], dim=1)

        return axis

    def forward(self, direction):
        if self.is_enabled:
            weight, theta, phi = self.deparameterize()
            axis = self.get_axis(theta, phi)

            if direction.ndim == 2:
                cos_angle = einsum(direction, axis, 'b c, sg c -> b sg')
                cos_angle = rearrange(cos_angle, 'b sg -> b sg 1')
                weight = rearrange(weight, 'sg c -> 1 sg c')
                weight = weight * (cos_angle > self.solid_angle).float()
                val = torch.sum(weight, dim=1)
            elif direction.ndim == 3:
                cos_angle = einsum(direction, axis, 'sg b c, sg c -> b sg')
                cos_angle = rearrange(cos_angle, 'b sg -> b sg 1')
                weight = rearrange(weight, 'sg c -> 1 sg c')
                weight = weight * (cos_angle > self.solid_angle).float()
                val = weight
                val = rearrange(val, 'b sg c -> sg b c')
            else:
                raise NotImplementedError()
        else:
            val = torch.zeros_like(direction)
        return val

    def reg_loss(self):
        if self.is_enabled:
            val = self.deparameterize_weight()
            val = torch.sum(val)

            return val
        else:
            return torch.tensor(0, device=self.weight.device, dtype=torch.float32)

    @torch.no_grad()
    def to_envmap(self, size=(256, 512)):
        envHeight, envWidth = size

        phi = np.linspace(-np.pi, np.pi, envWidth)
        theta = np.linspace(0, np.pi, envHeight)
        phi, theta = np.meshgrid(phi, theta)

        phi = torch.from_numpy(phi)[None, None]
        theta = torch.from_numpy(theta)[None, None]

        directions = self.get_axis(theta, phi)

        directions = einops.rearrange(directions, 'b c h w -> (b h w) c').to(torch.float32).to(self.theta.device)
        envmap = self.forward(directions)
        envmap = einops.rearrange(envmap, '(b h w) c -> b c h w', h=envHeight, w=envWidth)
        return envmap

    def sample_direction(self, vpos, normal):
        # (bn, spp, 3, h, w)
        weight, theta, phi = self.deparameterize()
        axis = self.get_axis(theta, phi)
        return axis[None, :, :, None, None]
        # return axis[None, :, :, None, None].expand_as(normal)

    @property
    def spp(self):
        return 1

class GlobalIncidentLighting(nn.Module):
    def __init__(self,
                 value=ConstantIntensity((-2, -2, -2), exp_val=True)):
        super().__init__()
        self.value = value

    @property
    def spp(self):
        return self.value.spp

    def sample_direction(self, vpos, normal):
        # (bn, spp, 3, h, w)
        return self.value.sample_direction(vpos, normal)

    def pdf_direction(self, vpos, direction):
        # (bn, spp, 3, h, w)
        return torch.ones_like(direction[:, :, :1, ...])

    def forward(self, direction):
        return self.value(direction)

    def val_reg_loss(self):
        return torch.zeros_like(self.value.reg_loss())

    def pos_reg_loss(self):
        return torch.zeros_like(self.value.reg_loss())
