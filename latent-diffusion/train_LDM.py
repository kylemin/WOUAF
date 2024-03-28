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
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import pytorch_lightning as pl

import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, Image
from diffusers import DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler, \
    EulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, randn_tensor
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from taming.data.faceshq import FFHQTrain, FFHQValidation
from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config

# for attribution
import itertools
from attribution import MappingNetwork, FullyConnectedLayer, FullyConnectedLayer_normal, MappingNetwork_normal
from customization import customize_vae_decoder
import inspect
from torchvision.utils import save_image
import lpips
import wandb
import copy
import shutil
from attack_methods.attack_initializer import attack_initializer  # For augmentation
from torch.utils.data import random_split, DataLoader, Dataset, Subset

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, open_dict, OmegaConf

logger = get_logger(__name__, log_level="INFO")

def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use.")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="offline",
        help=(
            "online || offline"
        ),
    )
    parser.add_argument(
        "--mapping_network_type",
        type=str,
        default="sg2",
        help=(
            " sg2 || normal"
        ),
    )
    parser.add_argument(
        "--decoding_network_type",
        type=str,
        default="resnet",
        help=(
            "fc || resnet || swin"
        ),
    )
    parser.add_argument(
        "--lpips_reg",
        default=False,
        action="store_true",
        help=(
            "lpips reg og and generated image"
        ),
    )
    parser.add_argument(
        "--run_nni",
        default=False,
        action="store_true",
        help=(
            "nni hyper-parameter search"
        ),
    )
    parser.add_argument(
        "--mapping_normalization",
        default=False,
        action="store_true",
        help=(
            "2nd moment Normalization in mapping function"
        ),
    )
    parser.add_argument(
        "--decoupling_type",
        type=str,
        default="l2",
        help=(
            "l2 || lpips"
        ),
    )
    parser.add_argument(
        "--phi_dimension",
        type=int,
        default=32,
        help=(
            "phi_dimension"
        ),
    )
    parser.add_argument(
        "--int_dimension",
        type=int,
        default=128,
        help=(
            "intermediate dimension"
        ),
    )
    parser.add_argument(
        "--mapping_layer",
        type=int,
        default=4,
        help=(
            "FC layers of mapping network"
        ),
    )
    parser.add_argument(
        "--attack",
        type=str,
        required=True,
        help="which components to modulate ('c' | 'r' | 'g' | 'b' | 'n' | 'e' | 'j' | 'm' | ... | 'crgbnejm' or 'all')",
    )
    parser.add_argument(
        "--modulation",
        type=str,
        required=True,
        help="which components to modulate ('d' | 'e' | 'q' | 'k' | 'v' | ... | 'deqkv')",
    )
    parser.add_argument(
        "--finetune",
        type=str,
        required=True,
        help="which components to finetune ('match', 'all'): 'match' indicates the same components specified by 'modulation'. every option includes affine",
    )
    parser.add_argument(
        "--weight_offset",
        default=False,
        action="store_true",
        help=(
            "Whether to apply an offset formulation"
        ),
    )
    parser.add_argument(
        "--num_gradient_from_last",
        type=int,
        default=1,
        help=(
            "number of getting gradient from last in the denoising loop"
        ),
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=False,
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
        default=None,
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
        default="sd-model-finetuned",
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
        default=False,
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
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
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
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--cosine_cycle",
        type=int,
        default=1,
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
        default=None,
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
        default=500,
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
    parser.add_argument(
        "--vae_config",
        type=str,
        default="configs/autoencoder/autoencoder_kl_64x64x3",
    )
    parser.add_argument(
        "--unet_config",
        type=str,
        default="configs/latent-diffusion/ffhq-ldm-vq-4",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)

def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


dataset_name_mapping = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
    "JotDe/mscoco_20k_unique_imgs_6k_test": ("image", "text"),
    "JotDe/mscoco_15k": ("image", "text"),
}


def get_phis(phi_dimension, batch_size, eps=1e-8):
    phi_length = phi_dimension
    b = batch_size
    phi = torch.empty(b, phi_length).uniform_(0, 1)
    return torch.bernoulli(phi) + eps


def check_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Reference https://github.com/huggingface/diffusers/blob/8178c840f265d4bee91fe9cf9fdd6dfef091a720/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L296
def get_ecoded_prompt_for_image_generation(
        prompt_embeds,
        batch_size,
        tokenizer,
        text_encoder,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt_embeds=None,
        negative_prompt=None, ):
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


def prepare_latents(batch_size, num_channels_latents, height, width, dtype, device, generator, vae_scale_factor,
                    scheduler, latents=None):
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
    image = vae.decode(latents, enconded_fingerprint).sample
    image = image.clamp(-1, 1)
    # image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    # image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    # image = image.permute(0, 2, 3, 1) # cause we need gradients from it
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
        num_channels_latents,  # unet.in_channels
        enconded_fingerprint,
        num_gradient_from_last,  # Whether random choice time step or not. if not last time step is used.
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
        validation=False,
):
    do_classifier_free_guidance = validation
    # do_classifier_free_guidance = False #In finetuning, it does not require this.
    enconded_fingerprint_fg = torch.cat(
        [enconded_fingerprint] * 2) if do_classifier_free_guidance else enconded_fingerprint

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    # 3. Encode input prompt
    prompt_embeds = get_ecoded_prompt_for_image_generation(prompt_for_image_generation, batch_size, tokenizer,
                                                           text_encoder, device,
                                                           do_classifier_free_guidance=do_classifier_free_guidance)

    # 4. Prepare timesteps
    if num_inference_steps < num_gradient_from_last:
        raise ValueError("num inferece steps should be larger than num_gradient_from_last ")

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # 5. Prepare latent variables
    if latents != None:  # The case for training
        latents = scheduler.add_noise(latents, torch.randn_like(latents),
                                      timesteps[-(num_gradient_from_last + 1)].unsqueeze(0))
        timesteps = timesteps[-(num_gradient_from_last + 1):]
    else:  # The case for validation
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

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = prepare_extra_step_kwargs(generator, eta, scheduler)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order

    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        # with torch.set_grad_enabled(i >= len(timesteps)-num_gradient_from_last):
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


def get_params_optimize(vaed, mapping_network, decoding_network, modulation, finetune):
    if finetune == 'all':
        params_to_optimize = itertools.chain(vaed.parameters(), mapping_network.parameters(),
                                             decoding_network.parameters())
    elif finetune == 'match':
        q = 'q' in modulation
        k = 'k' in modulation
        v = 'v' in modulation
        d = 'd' in modulation
        e = 'e' in modulation

        vaed_finetune_params = []
        for name, params in vaed.named_parameters():
            q_cond = q and (('attentions' in name and 'query' in name) or 'affine_q' in name)
            k_cond = k and (('attentions' in name and 'key' in name) or 'affine_k' in name)
            v_cond = v and (('attentions' in name and 'value' in name) or 'affine_v' in name)
            d_cond = d and (('resnets' in name and 'conv1' in name) or 'affine_d' in name)
            e_cond = e and (('resnets' in name and 'conv2' in name) or 'affine_e' in name)
            if q_cond or k_cond or v_cond or d_cond or e_cond:
                vaed_finetune_params.append(params)

        params_to_optimize = vaed_finetune_params + list(mapping_network.parameters()) + list(
            decoding_network.parameters())
    else:
        raise ValueError('unknown "finetune" argument')

    return params_to_optimize


def main():

    args = parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if args.run_nni:
        import nni
        tp = nni.get_next_parameter()

    wandb_report_name = [args.dataset_name,
                         "res", args.resolution,
                         "N-type", args.decoding_network_type,
                         "lr", args.learning_rate,
                         "cosine-cycle", args.cosine_cycle,
                         "warm-up", args.lr_warmup_steps,
                         "modulation", args.modulation,
                         "mapping-type", args.mapping_network_type,
                         "n-mapping", args.mapping_layer,
                         "dim-mapping", args.int_dimension,
                         "key-dim", args.phi_dimension,
                         "atack", args.attack,
                         ]
    wandb_report_name = [str(word) for word in wandb_report_name]
    wandb_report_name = "_".join(wandb_report_name)
    args.output_dir = args.output_dir + "/" + wandb_report_name

    wandb.init(mode=args.wandb_mode, name=wandb_report_name,
               entity = "apg-cogint", project = "gotta_catch_em_all")
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    print("out_dir: {}".format(args.output_dir))

    with initialize(version_base=None, config_path="configs", job_name="test_app"):
        cfg_hydra = compose(config_name="metrics", overrides=[])

    metrics = []
    for _, cb_conf in cfg_hydra.items():
        metrics.append(hydra.utils.instantiate(cb_conf))

    if args.mapping_network_type == "normal":
        args.mixed_precision = 'fp16'

    os.environ['WANDB_DISABLE_SERVICE'] = 'true'

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
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
    # noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # generation_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # generation_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path,
    #                                                               subfolder="scheduler")

    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    # )
    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    # )
    # vae = AutoencoderKL.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    # )

    configs = [OmegaConf.load(cfg) for cfg in [args.vae_config]]
    unknown = []
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)


    vae = instantiate_from_config(config.model)
    vae.load_state_dict(torch.load(config.model.load_path)['state_dict'])
    # unet = instantiate_from_config(args.unet_config)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    vae.decoder.requires_grad_(True)
    # unet.requires_grad_(False)  # Unet does not have to be trained. It is only used for validation.
    # text_encoder.requires_grad_(False)

    # if args.enable_xformers_memory_efficient_attention:
    #     if is_xformers_available():
    #         unet.enable_xformers_memory_efficient_attention()
    #     else:
    #         raise ValueError("xformers is not available. Make sure it is installed correctly")

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

    # mapping_network = MappingNetwork(args.phi_dimension, 0, args.int_dimension, None, num_layers=args.mapping_layer, w_avg_beta=None, normalization = args.mapping_normalization).to(accelerator.device)
    # mapping_network = MappingNetwork(args.phi_dimension, 0, args.int_dimension, None, num_layers=args.mapping_layer, lr_multiplier=np.sqrt(args.int_dimension), w_avg_beta=None, normalization = args.mapping_normalization).to(accelerator.device)
    if args.mapping_network_type == "sg2":
        mapping_network = MappingNetwork(args.phi_dimension, 0, args.int_dimension, None, num_layers=args.mapping_layer,
                                         w_avg_beta=None, normalization=args.mapping_normalization)
    elif args.mapping_network_type == "normal":
        mapping_network = MappingNetwork_normal(args.phi_dimension, args.int_dimension, num_layers=args.mapping_layer,
                                                mapping_normalization=args.mapping_normalization)
    else:
        print(args.mapping_network_type)
        raise ValueError("not support mapping network")

    if args.decoding_network_type == "fc":
        raise ValueError("FC is not supported anymore")
        # decoding_network = FullyConnectedLayer(4096*(args.resolution//256)**2, args.phi_dimension).to(accelerator.device)
        # decoding_network = FullyConnectedLayer(4096*(args.resolution//256)**2, args.phi_dimension, lr_multiplier=np.sqrt(args.int_dimension)).to(accelerator.device)
    elif args.decoding_network_type == "resnet":
        from torchvision.models import resnet50, ResNet50_Weights
        decoding_network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        decoding_network.fc = torch.nn.Linear(2048, args.phi_dimension)
    elif args.decoding_network_type == "swin":
        from torchvision.models import swin_v2_t, Swin_V2_T_Weights
        decoding_network = swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)
        decoding_network.head = torch.nn.Linear(768, args.phi_dimension)
    else:
        raise ValueError("Decoding network should be specified.")

    if args.decoupling_type == "lpips" or args.lpips_reg:
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(accelerator.device)

    vae = customize_vae_decoder(vae, args.int_dimension, args.modulation, args.finetune, args.weight_offset)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    vae.to(accelerator.device, dtype=weight_dtype)
    vae = accelerator.prepare(vae)


    mapping_network.to(accelerator.device, dtype=weight_dtype)
    mapping_network = accelerator.prepare(mapping_network)

    decoding_network.to(accelerator.device, dtype=weight_dtype)
    decoding_network = accelerator.prepare(decoding_network)

    optimizer = optimizer_cls(
        get_params_optimize(vae.decoder, mapping_network, decoding_network, args.modulation, args.finetune),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # if args.dataset_name is not None:
    #     # Downloading and loading a dataset from the hub.
    #     dataset = load_dataset(
    #         args.dataset_name,
    #         args.dataset_config_name,
    #         cache_dir=args.cache_dir,
    #         split="train"
    #     )
    #     dataset = dataset.train_test_split(test_size=0.2, shuffle=False)
    # else:
    #     data_files = {"train": "metadata_train.csv", "val": "metadata_val.csv"}
    #     dataset = load_dataset(
    #         "data/laion_aesthetics_6plus",
    #         data_files=data_files,
    #         delimiter='\t',
    #         on_bad_lines='skip',
    #     ).cast_column("image", Image())
    #     # See more about loading custom images at
    #     # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder
    #
    # # Preprocessing the datasets.
    # # We need to tokenize inputs and targets.
    # column_names = dataset["train"].column_names
    #
    # # 6. Get the column names for input/target.
    # dataset_columns = dataset_name_mapping.get(args.dataset_name, None)
    # if args.image_column is None:
    #     image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    # else:
    #     image_column = args.image_column
    #     if image_column not in column_names:
    #         raise ValueError(
    #             f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
    #         )
    # if args.caption_column is None:
    #     caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    # else:
    #     caption_column = args.caption_column
    #     if caption_column not in column_names:
    #         raise ValueError(
    #             f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
    #         )
    #
    # # Preprocessing the datasets.
    #
    # # Preprocessing the datasets.
    # train_transforms = transforms.Compose(
    #     [
    #         transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
    #         transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
    #         transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5], [0.5]),
    #     ]
    # )
    #
    # test_transforms = transforms.Compose(
    #     [
    #         transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
    #         transforms.CenterCrop(args.resolution),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5], [0.5]),
    #     ]
    # )
    #
    # def preprocess_train(examples):
    #     images = [image.convert("RGB") for image in examples[image_column]]
    #     examples["pixel_values"] = [train_transforms(image) for image in images]
    #     # examples["input_ids"] = tokenize_captions(examples)
    #     # examples["captions"] = examples["text"]
    #     return examples
    #
    # def preprocess_test(examples):
    #     images = [image.convert("RGB") for image in examples[image_column]]
    #     examples["pixel_values"] = [test_transforms(image) for image in images]
    #     # examples["input_ids"] = tokenize_captions(examples)
    #     # examples["captions"] = examples["text"]
    #     return examples
    #
    # with accelerator.main_process_first():
    #     if args.max_train_samples is not None:
    #         dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
    #     # Set the training transforms
    #     train_dataset = dataset["train"].with_transform(preprocess_train)
    #     test_dataset = dataset["test"].with_transform(preprocess_test)
    #
    # def collate_fn(examples):
    #     pixel_values = torch.stack([example["pixel_values"] for example in examples])
    #     pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    #     # input_ids = torch.stack([example["input_ids"] for example in examples])
    #     # captions = [example["captions"] for example in examples]
    #     return {"pixel_values": pixel_values}

    def acc_calculation(args, phis, decoding_network, generated_image, bsz=None, vae=None):
        if args.decoding_network_type == "fc":
            reconstructed_keys = decoding_network(vae.encode(generated_image).latent_dist.sample().reshape(bsz, -1))
        elif args.decoding_network_type == "resnet":
            reconstructed_keys = decoding_network(generated_image)
        else:
            raise ValueError("Not suported network")

        gt_phi = (phis > 0.5).int()
        reconstructed_keys = (torch.sigmoid(reconstructed_keys) > 0.5).int()
        bit_acc = ((gt_phi == reconstructed_keys).sum(dim=1)) / args.phi_dimension

        return bit_acc

    # DataLoaders creation:
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     shuffle=True,
    #     collate_fn=collate_fn,
    #     batch_size=args.train_batch_size,
    #     num_workers=args.dataloader_num_workers,
    # )
    #
    # test_dataloader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     shuffle=False,
    #     collate_fn=collate_fn,
    #     batch_size=args.train_batch_size,
    #     num_workers=args.dataloader_num_workers,
    # )
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
    train_dataloader = data.train_dataloader()
    train_dataset = train_dataloader.dataset
    args.train_batch_size = train_dataloader.batch_size
    test_dataloader = data.val_dataloader()

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.cosine_cycle * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler
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
        accelerator.init_trackers("LDM-fine-tune", config=vars(args))

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

    if accelerator.is_main_process:
        wandb.watch(mapping_network, log='all', log_freq=1000)
        # wandb.watch(decoding_network, log='all', log_freq=1000)

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

    # Augmentation
    train_aug = attack_initializer(args, is_train=True, device=accelerator.device)
    valid_aug = attack_initializer(args, is_train=False, device=accelerator.device)

    # Setup all metrics
    for metric in metrics:
        metric.setup(accelerator, args)

    for epoch in range(first_epoch, args.num_train_epochs):

        train_loss = 0.0
        list_train_bit_acc = []

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue


            # Convert images to latent space
            # vae.requires_grad_(False)
            latents = vae.encode(batch['image'].permute(0, 3, 1, 2).to(weight_dtype).to(accelerator.device))[0]
            # latents = torch.tile(latents, (2,1,1,1))
            # latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            # noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            # timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
            # timesteps = timesteps.long()

            # Sampling fingerprints and get the embeddings
            phis = get_phis(args.phi_dimension, bsz).to(latents.device)
            encoded_phis = mapping_network(phis)
            """
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps) #For DDPM

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, encoded_phis).sample
            loss_diffusion = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            """

            # Image construction
            # unet_in_channels = 4 #This is manually coded. Distributed lib does not support directly accessing unet.

            # Num inference param should be changed for setting which timestep you want to go and back.
            # generated_image, latent_hat = image_generation(encoder_hidden_states, args.resolution, args.resolution, accelerator.device, generation_scheduler, unet, vae, tokenizer, text_encoder, bsz, unet_in_channels, encoded_phis, num_gradient_from_last = args.num_gradient_from_last, num_inference_steps = 40, latents=latents, validation = False)
            generated_image = decode_latents(vae, latents, encoded_phis)

            # Image to key (vae.encoder -> flatten -> FC)
            # vae.encoder.requires_grad_(True) # need to double check
            if args.decoding_network_type == "fc":
                reconstructed_keys = decoding_network(
                    vae.encode(train_aug((generated_image / 2 + 0.5).clamp(0, 1)))[0].reshape(bsz,-1))
            elif args.decoding_network_type == "resnet" or args.decoding_network_type == "swin":
                reconstructed_keys = decoding_network(train_aug((generated_image / 2 + 0.5).clamp(0, 1)))
                # reconstructed_keys = decoding_network(generated_image)
            else:
                raise ValueError("Not suported network")

            # Key reconstruction loss = Element-wise BCE
            loss_key = F.binary_cross_entropy_with_logits(reconstructed_keys, phis)

            loss_lpips_reg = loss_fn_vgg.forward(generated_image,
                                                 batch['image'].permute(0, 3, 1, 2)).mean() if args.lpips_reg else 0.

            # key Disentangle Loss - loss_decouple in Responsible Paper
            # if args.decoupling_type == "lpips":
            #    loss_decouple = loss_fn_vgg(generated_image[:bsz//2], generated_image[bsz//2:]).squeeze()
            # elif args.decoupling_type == "l2":
            #    loss_decouple = F.mse_loss(generated_image[:bsz//2], generated_image[bsz//2:])

            # loss = loss_diffusion + loss_key + loss_decouple #TODO add hyper-param
            # loss = loss_diffusion + loss_key + loss_lpips_reg
            loss = loss_key + loss_lpips_reg

            # Calculate batch accuracy
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
                accelerator.clip_grad_norm_(
                    get_params_optimize(vae.decoder, mapping_network, decoding_network, args.modulation,
                                        args.finetune), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Validation
            if (epoch + 1) % 5 == 0 and step == 0:
                os.makedirs(os.path.join(args.output_dir, 'sample_images'), exist_ok=True)
                list_validation = []

                # Change network eval mode
                vae.eval()

                mapping_network.eval()
                decoding_network.eval()

                with torch.no_grad():
                    for val_step, batch in enumerate(test_dataloader):

                        # Convert images to latent space
                        # vae.requires_grad_(False)
                        latents = vae.encode(
                            batch['image'].permute(0, 3, 1, 2).to(weight_dtype).to(accelerator.device))[0]
                        # latents = torch.tile(latents, (2,1,1,1))
                        # latents = latents * 0.18215

                        # Sample noise that we'll add to the latents
                        bsz = latents.shape[0]

                        # Sampling fingerprints and get the embeddings
                        phis = get_phis(args.phi_dimension, bsz).to(latents.device)
                        encoded_phis = mapping_network(phis)

                        # # Get the text embedding for conditioning
                        # encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]
                        # # encoder_hidden_states = torch.tile(encoder_hidden_states, (2,1,1))
                        #
                        # # Image construction
                        # unet_in_channels = 4  # This is manually coded. Distributed lib does not support directly accessing unet.
                        #
                        # # num_inference_steps = 20 for EulerScheudler
                        # # vae.decoder.requires_grad_(True) # need to double checks
                        # generated_image, _ = image_generation(encoder_hidden_states, args.resolution,
                        #                                       args.resolution, accelerator.device,
                        #                                       generation_scheduler,
                        #                                       unet, vae, tokenizer, text_encoder, bsz,
                        #                                       unet_in_channels, encoded_phis,
                        #                                       num_gradient_from_last=args.num_gradient_from_last,
                        #                                       num_inference_steps=20, validation=True)
                        # Image to key (vae.encoder -> flatten -> FC)
                        # vae.encoder.requires_grad_(True) # need to double check

                        # Training's validation
                        generated_image_latent_0 = decode_latents(vae, latents, encoded_phis)
                        list_validation.extend(
                            acc_calculation(args, phis, decoding_network, generated_image_latent_0, bsz,
                                            vae).tolist())

                        outputs = {
                            'images': generated_image,
                            'images_trainval': generated_image_latent_0,
                        }
                        for metric in metrics:
                            ans = metric.infer_batch(accelerator, decoding_network, vae, batch, outputs, phis,
                                                     bsz)

                for metric in metrics:
                    metric.print_results(accelerator, epoch, step)

                # Saving Image
                generated_image = (generated_image / 2 + 0.5).clamp(0, 1)
                save_image(generated_image,
                           '{0}/sample_images/sample_e_{1}_s_{2}.png'.format(args.output_dir, epoch, step),
                           normalize=True, value_range=(0, 1), scale_each=True)
                a = wandb.Image(generated_image, caption="sample_e_{0}_s_{1}".format(epoch, step))
                wandb.log({"examples": a})

                if args.run_nni:
                    print("Reporting intermediate results to NNI")
                    nni.report_intermediate_result(float(torch.mean(list_validation).cpu().numpy()))

                # Change network train mode
                vae.train()

                mapping_network.train()
                decoding_network.train()
                # END Validation
            # END Validation

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                # accelerator.log({"diffusion_loss": loss_diffusion.item()}, step=global_step)
                # accelerator.log({"disentangle_loss": loss_decouple.item()}, step=global_step)
                accelerator.log({"key_loss": loss_key.item()}, step=global_step)
                if args.lpips_reg:
                    accelerator.log({"lpips_reg": loss_lpips_reg.item()}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        shutil.rmtree(
                            os.path.join(args.output_dir, f"checkpoint-{global_step - args.checkpointing_steps}"),
                            ignore_errors=True)
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        torch.save(vae.decoder.state_dict(), args.output_dir + "/vae_decoder.pth")
                        torch.save(mapping_network.state_dict(), args.output_dir + "/mapping_network.pth")
                        torch.save(decoding_network.state_dict(), args.output_dir + "/decoding_network.pth")

            # logs = {"diffusion_loss": loss_diffusion.detach().item(), "loss_key": loss_key.detach().item(), "loss_decouple": loss_decouple.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if args.lpips_reg:
                logs = {"loss_key": loss_key.detach().item(), "loss_lpips": loss_lpips_reg.item(),
                        "lr": lr_scheduler.get_last_lr()[0]}
            else:
                logs = {"loss_key": loss_key.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        train_acc = torch.mean(torch.tensor(list_train_bit_acc))
        print("Training Acc: Bit-wise Acc in Epoch {0}: {1}".format(epoch, train_acc))
        wandb.log({"Train Acc": train_acc.item()})

    if args.run_nni:
        print("Reporting final results to NNI")
        # nni.report_final_result(float(torch.mean(list_ncG_bit_acc).cpu().numpy()))
        nni.report_final_result(float(torch.mean(torch.tensor(list_train_bit_acc)).cpu().numpy()))

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # unet = accelerator.unwrap_model(unet)
        #
        # pipeline = StableDiffusionPipeline.from_pretrained(
        #     args.pretrained_model_name_or_path,
        #     text_encoder=text_encoder,
        #     vae=vae,
        #     unet=unet,
        #     revision=args.revision,
        # )
        # pipeline.save_pretrained(args.output_dir)

        torch.save(mapping_network.state_dict(), args.output_dir + "/mapping_network.pth")
        torch.save(decoding_network.state_dict(), args.output_dir + "/decoding_network.pth")

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()
    wandb.finish()


if __name__ == "__main__":
    main()
