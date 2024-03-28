#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, Image
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import randn_tensor
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import itertools
from attribution import MappingNetwork
from customization import customize_vae_decoder
import inspect
from torchvision.utils import save_image
import lpips
import wandb
from attack_methods.attack_initializer import attack_initializer #For augmentation
import hydra
from hydra import compose, initialize

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp_1",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--lr_mult",
        type=float,
        default=1,
        help="Learning rate multiplier for the affine layers",
    )
    parser.add_argument(
        "--pre_latents",
        type=str,
        default=None,
        help="Path to pre-extracted latents for validation",
    )
    parser.add_argument(
        "--phi_dimension",
        type=int,
        default=32,
        help="phi_dimension",
    )
    parser.add_argument(
        "--int_dimension",
        type=int,
        default=128,
        help="intermediate dimension",
    )
    parser.add_argument(
        "--mapping_layer",
        type=int,
        default=2,
        help="FC layers of mapping network",
    )
    parser.add_argument(
        "--attack",
        type=str,
        default='',
        help=(
            "which attack methods to apply ('c' | 'r' | 'g' | 'b' | 'n' | 'e' | 'j' | ... | 'crgbnej' or 'all' || 'AE_b_1' | 'AE_c_6' | ...)"
            "e.g. 'cr' denotes random cropping ('c') and rotation ('r')"
            "Use 'crgbnej' or 'all' for the combined attack in the paper"
        ),
    )
    parser.add_argument(
        "--num_gradient_from_last",
        type=int,
        default=1,
        help="number of getting gradient from last in the denoising loop",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-base",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceM4/COCO",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--train_steps_per_epoch",
        type=int,
        default=1000,
        help="Number of training steps per epoch. If provided, limits the number of iterations for each epoch",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=50000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine_with_restarts",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--cosine_cycle",
        type=int,
        default=1000,
        help=(
            "cosine_with_restarts option for cycle"
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


dataset_name_mapping = {
    'lambdalabs/pokemon-blip-captions': ('image', 'text'),
    'HuggingFaceM4/COCO': ('image', 'sentences_raw'),
    'imagenet-1k': ('image', 'label')
}


def get_phis(phi_dimension, batch_size ,eps = 1e-8):
    phi_length = phi_dimension
    b = batch_size
    phi = torch.empty(b,phi_length).uniform_(0,1)
    return torch.bernoulli(phi) + eps


def check_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#Reference https://github.com/huggingface/diffusers/blob/8178c840f265d4bee91fe9cf9fdd6dfef091a720/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L296
def get_ecoded_prompt_for_image_generation(
    prompt_embeds,
    batch_size,
    tokenizer,
    text_encoder,
    device,
    num_images_per_prompt = 1,
    do_classifier_free_guidance = True,
    negative_prompt_embeds = None,
    negative_prompt=None,):

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance and negative_prompt_embeds is None:
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [""] * batch_size
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt

        max_length = prompt_embeds.shape[1]
        uncond_input = tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            attention_mask = uncond_input.attention_mask.to(device)
        else:
            attention_mask = None

        negative_prompt_embeds = text_encoder(
            uncond_input.input_ids.to(device),
            attention_mask=attention_mask,
        )
        negative_prompt_embeds = negative_prompt_embeds[0]

    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoder.dtype, device=device)

        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    return prompt_embeds


def prepare_extra_step_kwargs(generator, eta, scheduler):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs


def prepare_latents(batch_size, num_channels_latents, height, width, dtype, device, generator, vae_scale_factor, scheduler, latents=None):
    shape = (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
        latents = latents.to(device)

    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * scheduler.init_noise_sigma
    return latents


def decode_latents(vae, latents, enconded_fingerprint):
    latents = 1 / 0.18215 * latents
    image = accelerator.unwrap_model(vae).decode(latents, enconded_fingerprint).sample
    image = image.clamp(-1,1)
    return image


def image_generation(
        prompt_for_image_generation,
        height,
        width,
        device,
        scheduler,
        unet,
        vae,
        tokenizer,
        text_encoder,
        batch_size,
        num_channels_latents, #unet.in_channels
        enconded_fingerprint,
        num_gradient_from_last, #Whether random choice time step or not. if not last time step is used.
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback=None,
        callback_steps: Optional[int] = 1,
        validation = False,
    ):

    do_classifier_free_guidance = validation
    enconded_fingerprint_fg = torch.cat([enconded_fingerprint] * 2) if do_classifier_free_guidance else enconded_fingerprint

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    # 3. Encode input prompt
    prompt_embeds = get_ecoded_prompt_for_image_generation(prompt_for_image_generation, batch_size, tokenizer, text_encoder, device, do_classifier_free_guidance=do_classifier_free_guidance)

    # 4. Prepare timesteps
    if num_inference_steps < num_gradient_from_last:
        raise ValueError("num inferece steps should be larger than num_gradient_from_last ")

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # 5. Prepare latent variables
    if latents != None: #The case for training
        latents = scheduler.add_noise(latents, torch.randn_like(latents), timesteps[-(num_gradient_from_last+1)].unsqueeze(0))
        timesteps = timesteps[-(num_gradient_from_last+1):]
    else: #The case for validation
        latents = prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                vae_scale_factor,
                scheduler,
                latents,
            )

    #6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = prepare_extra_step_kwargs(generator, eta, scheduler)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order

    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = unet(
                latent_model_input,
                t,
                prompt_embeds,
            ).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # call the callback, if provided
        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

     # 8. Post-processing
    image = decode_latents(vae, latents, enconded_fingerprint)
    return image, latents


def get_params_optimize(vaed, mapping_network, decoding_network):
    params_to_optimize = itertools.chain(vaed.parameters(), mapping_network.parameters(), decoding_network.parameters())
    return params_to_optimize


def acc_calculation(args, phis, decoding_network, generated_image, bsz = None, vae = None):
    reconstructed_keys = decoding_network(generated_image)
    gt_phi = (phis > 0.5).int()
    reconstructed_keys = (torch.sigmoid(reconstructed_keys) > 0.5).int()
    bit_acc = ((gt_phi == reconstructed_keys).sum(dim=1)) / args.phi_dimension

    return bit_acc


def load_val_latents(args, batch_size, val_step):
    val_latents = None
    step = val_step*args.train_batch_size
    for i in range(batch_size):
        vl = torch.load(os.path.join(args.pre_latents, f'{step+i}.pth'))
        if val_latents is None:
            val_latents = vl.unsqueeze(0)
        else:
            val_latents = torch.cat((val_latents, vl.unsqueeze(0)), 0)

    return val_latents

def val(args, epoch, step, accelerator, weight_dtype, generation_scheduler, tokenizer, vae, mapping_network, decoding_network, test_dataloader, valid_aug, resize, metrics):
    os.makedirs(os.path.join(args.output_dir,'sample_images'), exist_ok=True)
    list_validation = []
    list_txt_validation=[]

    #Change network eval mode
    vae.eval()
    mapping_network.eval()
    decoding_network.eval()

    with torch.no_grad():
        for val_step, batch in enumerate(test_dataloader):
            if (val_step+1)*args.train_batch_size >= 5000:
                break
            # Convert images to latent space
            latents = accelerator.unwrap_model(vae).encode(batch["pixel_values"].to(weight_dtype).to(accelerator.device)).latent_dist.sample()
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            bsz = latents.shape[0]

            # Sampling fingerprints and get the embeddings
            phis = get_phis(args.phi_dimension, bsz).to(latents.device)
            encoded_phis = mapping_network(phis)

            #Image construction
            val_latents = load_val_latents(args, bsz, val_step)
            generated_image = resize(decode_latents(vae, val_latents, encoded_phis))
            if 'AE_' in args.attack:
                augmented_image = valid_aug.forward((generated_image / 2 + 0.5).clamp(0, 1))['x_hat'].clamp(0, 1)
            else:
                augmented_image = valid_aug((generated_image / 2 + 0.5).clamp(0, 1))
            list_txt_validation.extend(acc_calculation(args, phis, decoding_network, augmented_image, bsz, vae).tolist())

            #Training's validation
            generated_image_latent_0 = resize(decode_latents(vae, latents, encoded_phis))
            if 'AE_' in args.attack:
                augmented_image_latent_0 = valid_aug.forward((generated_image_latent_0 / 2 + 0.5).clamp(0, 1))['x_hat'].clamp(0, 1)
            else:
                augmented_image_latent_0 = valid_aug((generated_image_latent_0 / 2 + 0.5).clamp(0, 1))
            list_validation.extend(acc_calculation(args, phis, decoding_network, augmented_image_latent_0, bsz, vae).tolist())

            outputs = {'images': (generated_image / 2 + 0.5).clamp(0, 1), 'images_trainval': (generated_image_latent_0 / 2 + 0.5).clamp(0, 1)}
            for metric in metrics:
                metric.infer_batch(accelerator, decoding_network, vae, batch, outputs, phis, bsz)

    for metric in metrics:
        metric.print_results(accelerator, epoch, step)

    #Saving Image
    generated_image = (generated_image / 2 + 0.5).clamp(0, 1)
    save_image(generated_image, '{0}/sample_images/sample_e_{1}_s_{2}.png'.format(args.output_dir,epoch,step),normalize=True, value_range=(0,1), scale_each=True)
    wandb.log({"Examples": wandb.Image(generated_image, caption="sample_e_{0}_s_{1}".format(epoch, step))})
    wandb.log({"Val Acc": float(np.mean(list_validation))})
    wandb.log({"Text-Val Acc": float(np.mean(list_txt_validation))})


def main():
    args = parse_args()
    wandb.init(name=args.exp_name, project="WOUAF")
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    with initialize(version_base=None, config_path="configs", job_name="test_app"):
        cfg_hydra = compose(config_name="metrics", overrides=[])

    metrics = []
    for _, cb_conf in cfg_hydra.items():
        metrics.append(hydra.utils.instantiate(cb_conf))

    os.environ['WANDB_DISABLE_SERVICE'] = 'true'

    global accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    generation_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    vae.decoder.requires_grad_(True)
    unet.requires_grad_(False) #Unet does not have to be trained. It is only used for validation.
    text_encoder.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    mapping_network = MappingNetwork(args.phi_dimension, args.int_dimension, num_layers=args.mapping_layer)
    decoding_network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    decoding_network.fc = torch.nn.Linear(2048, args.phi_dimension)

    loss_fn_vgg = lpips.LPIPS(net='vgg').to(accelerator.device)

    # Weight modulation to vae's decoder
    vae = customize_vae_decoder(vae, args.int_dimension, args.lr_mult)

    # For mixed precision training we cast the vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    mapping_network.to(accelerator.device, dtype=weight_dtype)
    decoding_network.to(accelerator.device, dtype=weight_dtype)

    optimizer = optimizer_cls(
        get_params_optimize(vae.decoder, mapping_network, decoding_network),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.

    if args.dataset_name is None:
        data_files = {"train": "metadata_train.csv", "val": "metadata_val.csv"}
        dataset = load_dataset(
            "data/laion_aesthetics_6plus",
            data_files=data_files,
            delimiter='\t',
            on_bad_lines='skip',
        ).cast_column("image", Image())
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder
    elif args.dataset_name in ['HuggingFaceM4/COCO', 'imagenet-1k']:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
    else:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split="train"
        )
        dataset = dataset.train_test_split(test_size = 0.2, shuffle=False)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = dataset_name_mapping.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            # For imagenet-1k (temporary solution)
            if isinstance(caption, int):
                caption = str(caption)

            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        examples["captions"] = examples[caption_column]
        return examples

    def preprocess_test(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [test_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples, False)
        examples["captions"] = examples[caption_column]
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)
        test_split = [sp for sp in dataset if sp not in ['train', 'test']]
        test_dataset  = dataset["test" if not test_split else test_split[0]].with_transform(preprocess_test)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        captions = [example["captions"] for example in examples]
        return {"pixel_values": pixel_values, "input_ids": input_ids, "captions": captions}


    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.train_steps_per_epoch is None:
        train_steps_per_epoch = num_update_steps_per_epoch
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * train_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles = args.cosine_cycle * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    vae, mapping_network, decoding_network, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        vae, mapping_network, decoding_network, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.train_steps_per_epoch is None:
        args.train_steps_per_epoch = num_update_steps_per_epoch
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * args.train_steps_per_epoch


    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / args.train_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    #Augmentation
    train_aug = attack_initializer(args, is_train = True, device = accelerator.device)
    valid_aug = attack_initializer(args, is_train = False, device = accelerator.device)

    if args.resolution == 256:
        resize = transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
    else:
        resize = torch.nn.Identity()

    # Setup all metrics
    for metric in metrics:
        metric.setup(accelerator, args)

    for epoch in range(first_epoch, args.num_train_epochs):
        vae.train()
        mapping_network.train()
        decoding_network.train()

        local_step = 0
        train_loss = 0.0
        list_train_bit_acc = []

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(accelerator.unwrap_model(vae).decoder), accelerator.accumulate(mapping_network), accelerator.accumulate(decoding_network):
                # Convert images to latent space
                latents = accelerator.unwrap_model(vae).encode(batch["pixel_values"].to(weight_dtype).to(accelerator.device)).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                bsz = latents.shape[0]

                # Sampling fingerprints and get the embeddings
                phis = get_phis(args.phi_dimension, bsz).to(latents.device)
                encoded_phis = mapping_network(phis)
                generated_image = decode_latents(vae, latents, encoded_phis)

                generated_image = resize(generated_image)
                if 'AE_' in args.attack:
                    augmented_image = train_aug.forward((generated_image / 2 + 0.5).clamp(0, 1))['x_hat'].clamp(0, 1)
                else:
                    augmented_image = train_aug((generated_image / 2 + 0.5).clamp(0, 1))

                #Image to key (vae.encoder -> flatten -> FC)
                reconstructed_keys = decoding_network(augmented_image)

                #Key reconstruction loss = Element-wise BCE
                loss_key = F.binary_cross_entropy_with_logits(reconstructed_keys, phis)
                loss_lpips_reg = loss_fn_vgg.forward(generated_image, resize(batch['pixel_values'])).mean()
                loss = loss_key + loss_lpips_reg

                #Calculate batch accuracy
                gt_phi = (phis > 0.5).int()
                reconstructed_keys = (torch.sigmoid(reconstructed_keys) > 0.5).int()
                bit_acc = ((gt_phi == reconstructed_keys).sum(dim=1)) / args.phi_dimension
                list_train_bit_acc.append(bit_acc.mean().item())

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(get_params_optimize(accelerator.unwrap_model(vae).decoder, mapping_network, decoding_network), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                local_step += 1
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                accelerator.log({"key_loss": loss_key.item()}, step=global_step)
                accelerator.log({"lpips_reg": loss_lpips_reg.item()}, step=global_step)
                train_loss = 0.0

                if (global_step % args.checkpointing_steps) == 0:
                    torch.save(accelerator.unwrap_model(vae).decoder.state_dict(), args.output_dir + "/vae_decoder.pth")
                    torch.save(mapping_network.state_dict(), args.output_dir + "/mapping_network.pth")
                    torch.save(decoding_network.state_dict(), args.output_dir + "/decoding_network.pth")

            logs = {"loss_key": loss_key.detach().item(), "loss_lpips":loss_lpips_reg.item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if local_step >= args.train_steps_per_epoch or global_step >= args.max_train_steps:
                break


        train_acc = torch.mean(torch.tensor(list_train_bit_acc))
        print("Training Acc: Bit-wise Acc in Epoch {0}: {1}".format(epoch, train_acc))
        wandb.log({"Train Acc": train_acc.item()})

        val(args, epoch, step, accelerator, weight_dtype, generation_scheduler, tokenizer, vae, mapping_network, decoding_network, test_dataloader, valid_aug, resize, metrics)
        if global_step >= args.max_train_steps:
            break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        vae = accelerator.unwrap_model(vae)
        mapping_network = accelerator.unwrap_model(mapping_network)
        decoding_network = accelerator.unwrap_model(decoding_network)

        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

        torch.save(mapping_network.state_dict(), args.output_dir + "/mapping_network.pth")
        torch.save(decoding_network.state_dict(), args.output_dir + "/decoding_network.pth")

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)


    accelerator.end_training()
    wandb.finish()


if __name__ == "__main__":
    main()
