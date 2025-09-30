from typing import Tuple, Union, List, Optional, Dict, Callable, Any

from huggingface_hub import snapshot_download
import numpy as np
import torch
from diffusers import FluxTransformer2DModel, FluxPipeline
from diffusers.configuration_utils import register_to_config
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock
from diffusers.pipelines.flux.pipeline_flux import XLA_AVAILABLE, retrieve_timesteps, calculate_shift, \
    EXAMPLE_DOC_STRING
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput
from diffusers.utils import replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
import copy


from huggingface_hub.utils import validate_hf_hub_args

from .cross_intrinsic_attention import CrossIntrinsicAttnProcessor2_0
from .batch_lora import inject_trainable_batched_lora, extract_loras, save_lora_weights


class IntrinsiXPipeline(FluxPipeline):
    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, 
                        pretrained_model_name_or_path, 
                        base_model_path="black-forest-labs/FLUX.1-dev", 
                        **kwargs):
        # Load the Base model
        pipe = FluxPipeline.from_pretrained(pretrained_model_name_or_path=base_model_path, **kwargs)
        pipe = IntrinsiXPipeline.from_pipe(pipe)

        # Inject the LoRA modules
        features = [
            "albedo",
            "material",
            "normal"
        ]

        lora_configs = list()
        for feature in features:
            if feature == "albedo":
                lora_configs.append({
                                    "r": 64,
                                    "dropout_p": 0.0,
                                    "scale": 1.0
                                })
            elif feature == "material":
                lora_configs.append({
                                    "r": 64,
                                    "dropout_p": 0.0,
                                    "scale": 1.0
                                })
            elif feature == "normal":
                lora_configs.append({
                                    "r": 64,
                                    "dropout_p": 0.0,
                                    "scale": 1.0
                                })
            elif feature == "shading":
                lora_configs.append({
                                    "r": 64,
                                    "dropout_p": 0.0,
                                    "scale": 1.0
                                })
            elif feature == "im":
                lora_configs.append(None)
        inject_trainable_batched_lora(model=pipe.transformer,
                                      target_modules={"to_k", "to_q", "to_v", "to_out.0"},
                                      lora_configs=lora_configs,
                                      verbose=True)
        
        # Load the LoRA weights
        cache_dir = kwargs.pop("cache_dir", None)
        lora_path = snapshot_download(repo_id=pretrained_model_name_or_path, cache_dir=cache_dir)
        lora_state_dict = cls.lora_state_dict(lora_path)
        loras_state_dict = {
                        f'{k.replace("transformer.", "")}': v
                        for k, v in lora_state_dict.items() if k.startswith("transformer.")
                    }
        pipe.transformer.load_state_dict(loras_state_dict, strict=False)

        # Set Cross-Intrinsic-Attention
        crossattn_processor = CrossIntrinsicAttnProcessor2_0()
        pipe.transformer.set_attn_processor(crossattn_processor)

        return pipe

    @classmethod
    def lora_state_dict(
            cls,
            pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
            return_alphas: bool = False,
            **kwargs,
    ):
        r"""
        Return state dict for lora weights and the network alphas.

        <Tip warning={true}>

        We support loading A1111 formatted LoRA checkpoints in a limited capacity.

        This function is experimental and might change in the future.

        </Tip>

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.

        """
        # Load the main state dict first which has the LoRA layers for either of
        # transformer and text encoder or both.
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        state_dict = cls._fetch_state_dict(
            pretrained_model_name_or_path_or_dict=pretrained_model_name_or_path_or_dict,
            weight_name=weight_name,
            use_safetensors=use_safetensors,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            allow_pickle=allow_pickle,
        )

        if return_alphas:
            return state_dict, None
        else:
            return state_dict

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        all_latents = []
        for latent in torch.split(latents, split_size_or_sections=1, dim=0):
            all_latents.append(
                super(IntrinsiXPipeline, IntrinsiXPipeline)._pack_latents(latent, 1, 16, height, width))
        all_latents = torch.cat(all_latents, dim=0)
        return all_latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        all_latents = []
        for latent in torch.split(latents, split_size_or_sections=1, dim=0):
            all_latents.append(
                super(IntrinsiXPipeline, IntrinsiXPipeline)._unpack_latents(latent, height, width,
                                                                                    vae_scale_factor))
        all_latents = torch.cat(all_latents, dim=0)
        return all_latents

    def prepare_latents(
            self,
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents=None,
            shared_noise=False,
            num_components=2
    ):
        assert batch_size == 1, "IntrinsiXPipeline only supports batch_size=1"

        height = 2 * (int(height) // self.vae_scale_factor)
        width = 2 * (int(width) // self.vae_scale_factor)

        if shared_noise:
            shape = (batch_size, 16, height, width)
        else:
            shape = (num_components, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        if shared_noise:
            latents = torch.cat([latents] * num_components, dim=0)  # Albedo, Shading, Image ...
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        return latents, latent_image_ids

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 28,
            timesteps: List[int] = None,
            guidance_scale: float = 3.5,
            num_images_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            max_sequence_length: int = 512,
            shared_noise=False,
            conditioning=None,
            conditioning_type=None,
            num_components=2,
            negative_prompt: Union[str, List[str]] = "",
            negative_prompt_2: Optional[Union[str, List[str]]] = "",
            true_cfg_scale: float = 1.0,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.
            shared_noise (bool, *optional*):
                Flag to indicate to use the same noise map for all the modalities (default False).
            conditioning ('List[Tensor]'): 
                Conditioning information passed to the model. Can be used during cross-conditioning. 
            conditioning_type (None | "cross" | "repaint"): 
                Used method for taking the conditioning information into account. If none, no conditioning information is used. If "cross", then cross-conditioning is used.
                If "repaint", then the RePaint approach is used. 

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            assert len(prompt) == 3, f"Only single-batch generation is supported"
            batch_size = 1
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if true_cfg_scale > 1:
            (
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            shared_noise=shared_noise,
            num_components=num_components
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # Use cross-conditioning if needed
        scheduler_copy = copy.deepcopy(self.scheduler)
        def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
            timestep = timesteps[0]
            if scheduler_copy.step_index is None:
                scheduler_copy._init_step_index(timestep)

            sigma = scheduler_copy.sigmas[scheduler_copy.step_index]
            scheduler_copy._step_index += 1
            return sigma.unsqueeze(0)
            
        conditioning_latents = None
        conditioning_noise = None
        if conditioning_type is not None:
            if conditioning_type in ("cross", "repaint"):
                if conditioning is not None:
                    conditioning_latents = []
                    conditioning_noise = []
                    for idx, conditioning in enumerate(conditioning):
                        if conditioning is None:
                            conditioning_latents.append(None)
                            conditioning_noise.append(None)
                        else:
                            conditioning_latent = conditioning[None].to(device=self.vae.device, dtype=self.vae.dtype)
                            conditioning_latent = conditioning_latent * 2 - 1
                            conditioning_latent = self.vae.encode(conditioning_latent).latent_dist.sample()
                            conditioning_latent = (conditioning_latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                            conditioning_latent = conditioning_latent.to(dtype=self.vae.dtype)
                            
                            conditioning_latents.append(conditioning_latent)
                            conditioning_noise.append(latents[idx])

        # 6. Denoising loop
        intermediates = dict()
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # Use cross-conditioning if needed
                if conditioning_latents is not None:
                    for idx, conditioning_latent in enumerate(conditioning_latents):
                        if conditioning_latent is None:
                            continue
                        if conditioning_type == "cross":
                            latents[idx] = self._pack_latents(
                                conditioning_latent,
                                batch_size=1,
                                num_channels_latents=conditioning_latent.shape[1],
                                height=conditioning_latent.shape[2],
                                width=conditioning_latent.shape[3],
                            )[0]
                            timestep[idx] = 0
                        elif conditioning_type == "repaint":
                            # Add noise according to the timestep
                            noise = conditioning_noise[idx].unsqueeze(0)
                            sigmas = get_sigmas(timestep[None, idx], n_dim=conditioning_latent.ndim, dtype=conditioning_latent.dtype)
                            conditioning_latent = self._pack_latents(
                                conditioning_latent,
                                batch_size=1,
                                num_channels_latents=conditioning_latent.shape[1],
                                height=conditioning_latent.shape[2],
                                width=conditioning_latent.shape[3],
                            )[0]
                            noisy_model_input = (1.0 - sigmas) * conditioning_latent + sigmas * noise
                            latents[idx] = noisy_model_input

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if true_cfg_scale > 1:
                    noise_pred_uncond = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred_text = noise_pred.clone()

                    noise_pred = noise_pred_uncond + true_cfg_scale * (noise_pred_text - noise_pred_uncond)

                    # Normalize
                    original_norm = noise_pred_text.norm(dim=(-1, -2), keepdim=True)
                    new_norm = noise_pred.norm(dim=(-1, -2), keepdim=True)
                    noise_pred = noise_pred * (original_norm / new_norm)
                
                # compute the previous noisy sample x_t -> x_t-1
                # WORKAROUND for cross-conditioning
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if conditioning_latents is not None:
                    for idx, conditioning_latent in enumerate(conditioning_latents):
                        if conditioning_latent is None:
                            continue
                        latents[idx] = self._pack_latents(
                                conditioning_latent,
                                batch_size=1,
                                num_channels_latents=conditioning_latent.shape[1],
                                height=conditioning_latent.shape[2],
                                width=conditioning_latent.shape[3],
                            )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            image = latents
        else:
            images = []
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            for latent in latents:
                latent = latent.unsqueeze(0)
                latent = (latent / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                image = self.vae.decode(latent, return_dict=False)[0]
                image = self.image_processor.postprocess(image, output_type="np")
                images.append(image)

            image = np.stack(images, axis=1).squeeze(0)
            if output_type == "pil":
                image = self.numpy_to_pil(image)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)

    def _decode_latent(self, latent, height, width):
        latent = self._unpack_latents(latent, height, width, self.vae_scale_factor)
        latent = (latent / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latent, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type="np")
        image = np.concatenate(image, axis=1)[None]
        image = self.numpy_to_pil(image)[0]
        return image
