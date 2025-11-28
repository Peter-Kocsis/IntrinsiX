import math
import os
import tempfile

import PIL
import numpy as np

from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
from torchvision.transforms import ToPILImage, ToTensor
import numpy as np
from PIL import Image
import requests

import gradio as gr

from huggingface_hub import hf_hub_download

from .render.lighting import DirectionalLight_LatLong, GlobalIncidentLighting
from .render.render import IIR_RenderLayer

from .model.intrinsix import IntrinsiXPipeline

@torch.no_grad()
def run_intrinsix(pipe, prompt, seed=0, **kwargs):
    """
    Generate Albedo, Rough, Metallic and Normal properties from a given text prompt.
    """
    image = pipe(prompt=prompt, width=512, height=512, num_components=3, generator=torch.Generator(device="cuda").manual_seed(seed)).images
    return {
        "albedo": image[0],
        "roughness": PIL.Image.fromarray(np.asarray(image[1])[..., 0]),
        "metallic": PIL.Image.fromarray(np.asarray(image[1])[..., 1]),
        "normal": image[2] 
    }

@torch.no_grad()
def run_rendering(albedo, roughness, metallic, normal, ligting_longitude, lighting_latitude, ambient=0.2):
    def tonemap(image):
        mu = 64
        return torch.log(1 + mu * image)/ torch.log(torch.tensor(1 + mu, device=image.device, dtype=image.dtype))
    
    # Prepare lighting
    device = albedo.device
    lighting_model = GlobalIncidentLighting(value=DirectionalLight_LatLong(weight_init=2)).to(device)
    lighting_model.value.theta[:] = lighting_latitude
    lighting_model.value.phi[:] = ligting_longitude

    # Prepare the renderer
    renderer = IIR_RenderLayer(imWidth=albedo.shape[1],
                               imHeight=albedo.shape[2],
                               brdf_type="disney",
                               double_sided=False,
                               use_specular=True).to(device)
    
    # Render the image
    colorDiffuse, colorSpec, wi_mask, shading = renderer(lighting_model=lighting_model,
                                                         albedo=albedo.unsqueeze(0),
                                                         rough=roughness.unsqueeze(0),
                                                         metal=metallic.unsqueeze(0),
                                                         normal=normal.unsqueeze(0))
    rendered_image = (1 - ambient) * (colorDiffuse + colorSpec) + ambient * albedo.unsqueeze(0).to(device)
    rendered_image = rendered_image[0]
    
    # Tonemapping
    rendered_image = tonemap(rendered_image.clamp(0, 1))

    return rendered_image


def intrinsix_demo():
    # Define the device
    device = "cuda"

    # Load the model
    print(f"Loading the model (can take some time)")
    pipe = IntrinsiXPipeline.from_pretrained(
        pretrained_model_name_or_path="PeterKocsis/IntrinsiX", 
        torch_dtype=torch.bfloat16,
        cache_dir=os.path.join("models"),
        device_map="balanced")

    # Generator function
    def generate_pbr(input_text, seed):
        generated_pbrs = run_intrinsix(pipe, input_text, seed=seed)
        state = dict()
        return generated_pbrs["albedo"], generated_pbrs["roughness"], generated_pbrs["metallic"], generated_pbrs["normal"], state
    
    # Rendering function
    def render_pbr(albedo, roughness, metallic, normal, ligting_longitude, lighting_latitude):
        to_tensor, to_pil = ToTensor(), ToPILImage()

        # Convert to torch
        albedo_torch = to_tensor(albedo)
        roughness_torch = to_tensor(roughness)
        metallic_torch = to_tensor(metallic)
        normal_torch = to_tensor(normal)

        # Normalize
        normal_torch = normal_torch * 2 - 1
        normal_torch = torch.nn.functional.normalize(normal_torch, dim=0)
        albedo_torch = albedo_torch ** 2.2
        ligting_longitude = ligting_longitude * np.pi / 180.
        lighting_latitude = lighting_latitude * np.pi / 180.

        # Call the renderer
        rendered_image_torch = run_rendering(albedo_torch, roughness_torch, metallic_torch, normal_torch, ligting_longitude, lighting_latitude)

        # Convert to PIL Image
        return to_pil(rendered_image_torch)

    # head_style = """
    # <style>
    # @media (min-width: 720px)
    # {
    #     .gradio-container {
    #         min-width: var(--size-full) !important;
    #     }
    # }
    # </style>
    # """

    with gr.Blocks(title="IntrinsiX") as demo:
        with gr.Row():
            gr.Markdown("""
                        # [IntrinsiX](https://peter-kocsis.github.io/IntrinsiX/): High-Quality PBR Generation with Image Priors  

                        Turn text prompts into physically based rendering (PBR) materials.  
                        Generate albedo, normal, roughness, and metallic maps — and interactively relight them.
            """)
        with gr.Row(variant="panel"):
            with gr.Column():
                    input_text = gr.TextArea(
                        value="An astronaut riding a unicorn on the moon",
                        label="Input Text",
                        elem_id="input_text"
                    )

                    with gr.Accordion(label="Generation Settings", open=False):
                        seed = gr.Slider(0, np.iinfo(np.int32).max, label="Seed", value=0, step=1)

                    with gr.Accordion(label="Rendering Settings", open=False):
                        lighting_latitude = gr.Slider(0, 90, label="Lighting Latitude", value=45, step=5)
                        lighting_longitude = gr.Slider(-90, 90, label="Lighting Longitude", value=45, step=5)

                    submit = gr.Button("Generate PBR", elem_id="button_generate", variant="primary")

            with gr.Column(scale=2):
                with gr.Row():
                    albedo_image = gr.Image(
                        label="Albedo",
                        type="pil",
                        interactive=False
                    )
                    normal_image = gr.Image(
                        label="Normal",
                        type="pil",
                        interactive=False
                    )
                
                with gr.Row():
                    roughness_image = gr.Image(
                        label="Rough",
                        type="pil",
                        interactive=False
                    )
                    metallic_image = gr.Image(
                        label="Metallic",
                        type="pil",
                        interactive=False
                    )

            with gr.Column(scale=2):
                with gr.Row():
                    rendered_image = gr.Image(
                        label="Rendering",
                        type="pil",
                        interactive=False
                    )

        # state = gr.State()

        # Define the demo pipeline
        generate_kwarg = {
             "fn": generate_pbr,
             "inputs": [input_text, seed],
             "outputs": [albedo_image, roughness_image, metallic_image, normal_image]
        }

        relight_kwargs = {
            "fn": render_pbr,
            "inputs": [albedo_image, roughness_image, metallic_image, normal_image, lighting_longitude, lighting_latitude],
            "outputs": [rendered_image]
        }

        submit.click(**generate_kwarg).success(**relight_kwargs)

        lighting_longitude.change(**relight_kwargs)
        lighting_latitude.change(**relight_kwargs)

        demo.queue(max_size=10)
        demo.launch(server_name="0.0.0.0", debug=True)


if __name__ == "__main__":
    intrinsix_demo()
