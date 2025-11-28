#!/usr/bin/env python
# Adapted from https://github.com/huggingface/diffusers/blob/main/examples/flux-control/train_control_lora_flux.py

import sys

import safetensors

sys.path.append("../")
sys.path.append(".")

import os

import argparse
import copy
import itertools
import logging
import math
import os
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from huggingface_hub.utils import insecure_hashlib
from hydra import initialize, compose
from omegaconf import OmegaConf
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
    FluxPipeline
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.torch_utils import is_compiled_module

from intrinsix.data.dataloading import BatchTransform, Clamp, Compose, MultiplyTransform, NormalRandomHorizontalFlip
from intrinsix.data.dataset import InteriorVerseDataset


from batch import Batch

import hydra
from intrinsix.io.config_io import range2list


if is_wandb_available():
    import wandb

import lovely_tensors as lt

lt.monkey_patch()

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

logger = get_logger(__name__)


def parse_args(input_args=None):
    print(f"Input args: {input_args}")
    with initialize(version_base=None, config_path="../configs/train"):
        args = compose(config_name=input_args[0], overrides=input_args[1:])
        OmegaConf.resolve(args)
        args = OmegaConf.to_container(args)
        args = argparse.Namespace(**args)
    print(f"Config: \n{args}")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    args.output_dir = os.path.join(args.output_dir, args.project_name, args.exp_name)

    return args


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant, cache_dir=args.model_cache_dir
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant, cache_dir=args.model_cache_dir
    )
    return text_encoder_one, text_encoder_two


def log_validation(
        pipeline,
        args,
        accelerator,
        pipeline_args,
        epoch,
        torch_dtype,
        is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device, dtype=torch_dtype)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    validation_prompts = pipeline_args.pop("prompt")
    if not isinstance(validation_prompts, list):
        validation_prompts = [validation_prompts]
    validation_prompts = [prompt for prompt in validation_prompts for _ in range(args.num_validation_images)]

    with accelerator.autocast():
        images = [pipeline(prompt=prompt, **pipeline_args, generator=generator, width=512, height=512).images[0]
                  for prompt in validation_prompts]

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {prompt}") for i, (prompt, image) in enumerate(zip(validation_prompts, images))
                    ]
                }
            )

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images


def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision, cache_dir=args.model_cache_dir
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def collate_fn(examples):
    batch = Batch.from_batch_list(*examples)
    batch = batch.map(torch.stack, on_failure="keep")
    return batch.to_dict()


def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
        text_encoder,
        tokenizer,
        max_sequence_length=512,
        prompt=None,
        num_images_per_prompt=1,
        device=None,
        text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
        text_encoder,
        tokenizer,
        prompt: str,
        device=None,
        text_input_ids=None,
        num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
        text_encoders,
        tokenizers,
        prompt: str,
        max_sequence_length,
        device=None,
        num_images_per_prompt: int = 1,
        text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    # text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    # text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)
    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def main(args):
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir) if args.output_dir is not None else None

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load the tokenizers
    logger.info(f"Loading tokenizers...")
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        cache_dir=args.model_cache_dir
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        cache_dir=args.model_cache_dir
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    logger.info(f"Loading noise scheduler...")
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="scheduler",
        cache_dir=args.model_cache_dir
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    logger.info(f"Loading text encoders...")
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)

    logger.info(f"Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        cache_dir=args.model_cache_dir
    )

    logger.info(f"Loading transformer...")
    transformer = FluxTransformer2DModel.from_pretrained(args.pretrained_model_name_or_path, 
                                                         subfolder="transformer",
                                                         cache_dir=args.model_cache_dir)

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()

    # now we will add new LoRA weights to the attention layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        use_dora=args.use_dora if hasattr(args, "use_dora") else False,
    )
    transformer.add_adapter(transformer_lora_config)
    if args.train_text_encoder:
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            use_dora=args.use_dora if hasattr(args, "use_dora") else False,
        )
        text_encoder_one.add_adapter(text_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                elif isinstance(model, type(unwrap_model(text_encoder_one))):
                    text_encoder_one_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            FluxPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None
        text_encoder_one_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = FluxPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )
        if args.train_text_encoder:
            # Do we need to call `scale_lora_layers()` here?
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            if args.train_text_encoder:
                models.extend([text_encoder_one_])
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

        # load in/out layers
        def load_in_out(transformer, input_dir):
            def load_params_to_layer(state_dict, layer, prefix):
                layers_weights = {k.replace(f"{prefix}.", ""): v for k, v in state_dict if k.startswith(f"{prefix}.")}
                layer.load_state_dict(layers_weights)

            state_dict = safetensors.torch.load_file(Path(input_dir, "channel_change.pt").as_posix()).items()
            load_params_to_layer(state_dict, transformer.x_embedder, "transformer.x_embedder")
            load_params_to_layer(state_dict, transformer.proj_out, "transformer.proj_out")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        if args.train_text_encoder:
            models.extend([text_encoder_one])
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    if args.train_text_encoder:
        text_lora_parameters_one = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    if args.train_text_encoder:
        # different learning rate for text encoder and unet
        text_parameters_one_with_lr = {
            "params": text_lora_parameters_one,
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        params_to_optimize = [
            transformer_parameters_with_lr,
            text_parameters_one_with_lr,
        ]
    else:
        params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warning(
                f"Learning rates were provided both for the transformer and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters_one and text_encoder_parameters_two to be
            # --learning_rate
            params_to_optimize[1]["lr"] = args.learning_rate
            params_to_optimize[2]["lr"] = args.learning_rate

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    data_features = copy.deepcopy(args.features_to_cat)

    if args.caption_prefix is not None:
        data_features.append("caption")
        assert args.instance_prompt is None, "Cannot use both --caption_prefix and --instance_prompt"
    if "extra_features" in args and args.extra_features is not None:
        data_features.extend(args.extra_features)

    train_dataset = InteriorVerseDataset(
        root=args.dataset_path,
        extra_source=args.dataset_extra_source if "dataset_extra_source" in args else None,
        allow_missing_scenes=True,
        fixed_caption=args.instance_prompt,
        caption_prefix=args.caption_prefix,
        features_to_include=data_features,
        fixed_features_to_include={"caption": args.instance_prompt} if args.instance_prompt is not None else None,
        transform=BatchTransform(
            reset_params=True,
            transform={
                "albedo": Compose({
                    "t0": transforms.ToTensor(),
                    "t1": transforms.Resize(size=args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                    "t2": transforms.RandomCrop(size=args.resolution),
                    "t3": transforms.Normalize(mean=[0.5], std=[0.5]),
                    "t4": transforms.RandomHorizontalFlip(p=0.5),
                }),
                "material": Compose({
                    "t0": transforms.ToTensor(),
                    "t1": Clamp(min=0.0, max=1.0),
                    "t2": MultiplyTransform(factor=[[[1]], [[1]], [[0]]]),  # Zero out the Blue channel
                    "t3": transforms.Resize(size=args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                    "t4": transforms.RandomCrop(size=args.resolution),
                    "t5": transforms.Normalize(mean=[0.5], std=[0.5]),
                    "t6": transforms.RandomHorizontalFlip(p=0.5),
                }),
                "normal": Compose({
                    "t0": transforms.ToTensor(),
                    "t1": transforms.Resize(size=args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                    "t2": transforms.RandomCrop(size=args.resolution),
                    "t3": NormalRandomHorizontalFlip(p=0.5),
                }),
                "idx": None,
                "timestep": None,
                "caption": None})
    )

    sampler_cfg = args.sampler if "sampler" in args else None
    sampler = None
    if sampler_cfg is not None:
        sampler = hydra.utils.instantiate(sampler_cfg,
                                          indices=range2list(sampler_cfg.pop("indices"),
                                          max_length=len(train_dataset)))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
    )

    # Modify the validation prompts accordingly
    if args.instance_prompt is not None:
        args.validation_prompt = [args.instance_prompt + " " + prompt for prompt in args.validation_prompt]
    if args.caption_prefix is not None:
        args.validation_prompt = [args.caption_prefix + " " + prompt for prompt in args.validation_prompt]

    if not args.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    text_encoders, tokenizers, prompt, args.max_sequence_length
                )
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
                text_ids = text_ids.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds, text_ids

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    if not args.train_text_encoder and not train_dataset.custom_instance_prompts:
        instance_prompt_hidden_states, instance_pooled_prompt_embeds, instance_text_ids = compute_text_embeddings(
            args.instance_prompt, text_encoders, tokenizers
        )

    # Clear the memory here
    if not args.train_text_encoder and not train_dataset.custom_instance_prompts:
        del text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
        free_memory()

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.

    if not train_dataset.custom_instance_prompts:
        logger.info(f"Using the same instance prompt for all images: {args.instance_prompt}")
        if not args.train_text_encoder:
            prompt_embeds = instance_prompt_hidden_states
            pooled_prompt_embeds = instance_pooled_prompt_embeds
            text_ids = instance_text_ids
        # if we're optimizing the text encoder (both if instance prompt is used for all images or custom prompts)
        # we need to tokenize and encode the batch prompts on all training steps
        else:
            tokens_one = tokenize_prompt(tokenizer_one, args.instance_prompt, max_sequence_length=77)
            tokens_two = tokenize_prompt(
                tokenizer_two, args.instance_prompt, max_sequence_length=args.max_sequence_length
            )
    else:
        logger.info("Using custom instance prompts for each image.")

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        (
            transformer,
            text_encoder_one,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            transformer,
            text_encoder_one,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = args.project_name
        wandb_dir = args.output_dir
        os.makedirs(wandb_dir, exist_ok=True)
        accelerator.init_trackers(tracker_name, config=vars(args), init_kwargs={"wandb": {"name": args.exp_name, 
                                                                                         "dir": wandb_dir }})

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Total params = {sum(p.numel() for p in transformer_lora_parameters)}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    if accelerator.is_main_process:
        optimizer_to(optimizer, "cpu")
        free_memory()
        # if True:
        # create pipeline
        if not args.train_text_encoder:
            text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one,
                                                                    text_encoder_cls_two)
            text_encoder_one.to(accelerator.device, dtype=weight_dtype)
            text_encoder_two.to(accelerator.device, dtype=weight_dtype)

        pipeline = FluxPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            text_encoder=accelerator.unwrap_model(text_encoder_one),
            text_encoder_2=accelerator.unwrap_model(text_encoder_two),
            transformer=accelerator.unwrap_model(transformer),
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
            cache_dir=args.model_cache_dir,
        )
        # prompt = "unicorn on a beach"
        # with accelerator.autocast():
        #     image = pipeline(prompt).images[0]
        # display(image)
        pipeline_args = {"prompt": args.validation_prompt}
        images = log_validation(
            pipeline=pipeline,
            args=args,
            accelerator=accelerator,
            pipeline_args=pipeline_args,
            epoch=0,
            torch_dtype=weight_dtype,
        )
        if not args.train_text_encoder:
            del text_encoder_one, text_encoder_two
            free_memory()
        del pipeline
        optimizer_to(optimizer, accelerator.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            # set top parameter requires_grad = True for gradient checkpointing works
            accelerator.unwrap_model(text_encoder_one).text_model.embeddings.requires_grad_(True)

        for step, batch in enumerate(train_dataloader):  # Step-based validation
            models_to_accumulate = [transformer]
            if args.train_text_encoder:
                models_to_accumulate.extend([text_encoder_one])
            with accelerator.accumulate(models_to_accumulate):
                prompts = batch["caption"]

                # Get max timtestep
                min_timestep = batch.get("timestep", None)

                # encode batch prompts when custom prompts are provided for each image -
                if train_dataset.custom_instance_prompts:
                    if not args.train_text_encoder:
                        prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                            prompts, text_encoders, tokenizers
                        )
                    else:
                        tokens_one = tokenize_prompt(tokenizer_one, prompts, max_sequence_length=77)
                        tokens_two = tokenize_prompt(
                            tokenizer_two, prompts, max_sequence_length=args.max_sequence_length
                        )
                        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                            text_encoders=[text_encoder_one, text_encoder_two],
                            tokenizers=[None, None],
                            text_input_ids_list=[tokens_one, tokens_two],
                            max_sequence_length=args.max_sequence_length,
                            device=accelerator.device,
                            prompt=prompts,
                        )
                else:
                    if args.train_text_encoder:
                        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                            text_encoders=[text_encoder_one, text_encoder_two],
                            tokenizers=[None, None],
                            text_input_ids_list=[tokens_one, tokens_two],
                            max_sequence_length=args.max_sequence_length,
                            device=accelerator.device,
                            prompt=args.instance_prompt,
                        )

                # Convert images to latent space
                model_inputs = Batch(batch)
                model_inputs = model_inputs[args.features_to_cat]
                model_inputs = model_inputs.map(lambda x: vae.encode(x.to(dtype=vae.dtype)).latent_dist.sample())
                model_input = torch.cat(model_inputs.to_list(per_element=True), dim=1)
                model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                vae_scale_factor = 2 ** (len(vae_config_block_out_channels))

                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2],
                    model_input.shape[3],
                    accelerator.device,
                    weight_dtype,
                )
                # Sample noise that we'll add to the latents
                # print(f"Model input: {model_input}")
                noise = torch.randn_like(model_input[:, :model_input.shape[1]], )
                noise = torch.cat([noise], dim=1)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                # Restrict the sampled timestep based on the input
                if min_timestep is not None:
                    t_cont = min_timestep[0] / (noise_scheduler_copy.config.num_train_timesteps - 1)
                    u = u * (1 - t_cont)
                    loss_weight = (1 - t_cont)
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
                # logger.info(f"Min t: {min_timestep}, sampled: {timesteps}, loss_weight: {loss_weight}")

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )

                # handle guidance
                if True:
                    # if transformer.config.guidance_embeds:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None

                if torch.isnan(timesteps).any():
                    logger.warning("NaN timesteps! Skipping the update step.")
                    continue
                if torch.isnan(pooled_prompt_embeds).any():
                    logger.warning("NaN pooled_prompt_embeds! Skipping the update step.")
                    continue
                if torch.isnan(prompt_embeds).any():
                    logger.warning("NaN prompt_embeds! Skipping the update step.")
                    continue
                if torch.isnan(text_ids).any():
                    logger.warning("NaN text_ids! Skipping the update step.")
                    continue
                if torch.isnan(latent_image_ids).any():
                    logger.warning("NaN latent_image_ids! Skipping the update step.")
                    continue
                if torch.isnan(packed_noisy_model_input).any():
                    logger.warning("NaN packed_noisy_model_input! Skipping the update step.")
                    continue
                # Predict the noise residual
                # print(f"========\nINPUT: {packed_noisy_model_input}\nL1: {packed_noisy_model_input[...,:64]}\nL2: {packed_noisy_model_input[...,64:]}")
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                if torch.isnan(model_pred).any():
                    logger.warning("NaN pred!")

                # print(f"========\nOUTPUT: {model_pred}\nL1: {model_pred[...,:64]}\nL2: {model_pred[...,64:]}")
                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=int(model_input.shape[2] * vae_scale_factor / 2),
                    width=int(model_input.shape[3] * vae_scale_factor / 2),
                    vae_scale_factor=vae_scale_factor,
                )
                # print(f"========\nOUTPUT_UNPACKED: {model_pred}\nL1: {model_pred[:,:16]}\nL2: {model_pred[:,16:]}")

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = noise - model_input

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                if min_timestep is not None:
                    loss = loss_weight * loss
                
                if torch.isnan(loss).any():
                    logger.warning("NaN loss detected! Skipping the update step.")
                    continue

                accelerator.backward(loss)
                params = list(itertools.chain(transformer.parameters(),
                                          text_encoder_one.parameters()) if args.train_text_encoder else transformer.parameters())
                if accelerator.sync_gradients:
                    if args.max_grad_norm is not None:
                        accelerator.clip_grad_norm_(params, args.max_grad_norm)

                def nan_grad(parameters):
                    for param in parameters:
                        if param.grad is not None and torch.isnan(param.grad).any():
                            return True
                    return False

                def nan_param(parameters):
                    for param in parameters:
                        if param is not None and torch.isnan(param).any():
                            return True
                    return False

                if not nan_grad(params):
                    optimizer.step()
                    lr_scheduler.step()
                else:
                    logger.warning("NaN gradients detected. Skipping the update step.")
                optimizer.zero_grad()

                if nan_param(params):
                    logger.warning("NaN params detected.")
                    exit()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        optimizer_to(optimizer, "cpu")
                        free_memory()
                        # if True:
                        # create pipeline
                        if not args.train_text_encoder:
                            text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one,
                                                                                    text_encoder_cls_two)
                            text_encoder_one.to(accelerator.device, dtype=weight_dtype)
                            text_encoder_two.to(accelerator.device, dtype=weight_dtype)
                        pipeline = FluxPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            vae=vae,
                            text_encoder=accelerator.unwrap_model(text_encoder_one),
                            text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                            transformer=accelerator.unwrap_model(transformer),
                            revision=args.revision,
                            variant=args.variant,
                            torch_dtype=weight_dtype,
                            cache_dir=args.model_cache_dir
                        )

                        pipeline_args = {"prompt": args.validation_prompt}
                        images = log_validation(
                            pipeline=pipeline,
                            args=args,
                            accelerator=accelerator,
                            pipeline_args=pipeline_args,
                            epoch=global_step,
                            torch_dtype=weight_dtype,
                        )
                        if not args.train_text_encoder:
                            del text_encoder_one, text_encoder_two
                            free_memory()
                        del pipeline
                        optimizer_to(optimizer, accelerator.device)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        if args.upcast_before_saving:
            transformer.to(torch.float32)
        else:
            transformer = transformer.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)

        if args.train_text_encoder:
            text_encoder_one = unwrap_model(text_encoder_one)
            text_encoder_lora_layers = get_peft_model_state_dict(text_encoder_one.to(torch.float32))
        else:
            text_encoder_lora_layers = None

        final_out_path = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(final_out_path, exist_ok=True)
        FluxPipeline.save_lora_weights(
            save_directory=final_out_path,
            transformer_lora_layers=transformer_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
        )

        # Save in and out layers
        def save_in_out(output_dir):
            def add_params_from_layer(layer, prefix):
                layers_weights = layer.state_dict()
                layers_state_dict = {f"{prefix}.{module_name}": param for module_name, param in
                                     layers_weights.items()}
                return layers_state_dict

            state_dict = {}
            state_dict.update(add_params_from_layer(transformer.x_embedder, "transformer.x_embedder"))
            state_dict.update(add_params_from_layer(transformer.proj_out, "transformer.proj_out"))

            safetensors.torch.save_file(state_dict, Path(output_dir, "channel_change.pt").as_posix(),
                                        metadata={"format": "pt"})

        # Final inference
        # Load previous pipeline
        pipeline = FluxPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
            cache_dir=args.model_cache_dir
        )
        # load attention processors
        pipeline.load_lora_weights(final_out_path)

        # load in/out layers
        def load_in_out(pipe, input_dir):
            def load_params_to_layer(state_dict, layer, prefix):
                layers_weights = {k.replace(f"{prefix}.", ""): v for k, v in state_dict if k.startswith(f"{prefix}.")}
                layer.load_state_dict(layers_weights)

            state_dict = safetensors.torch.load_file(Path(input_dir, "channel_change.pt").as_posix()).items()
            load_params_to_layer(state_dict, pipe.transformer.x_embedder, "transformer.x_embedder")
            load_params_to_layer(state_dict, pipe.transformer.proj_out, "transformer.proj_out")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
