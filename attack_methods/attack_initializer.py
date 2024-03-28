from .Gaussian_blur import Gaussian_blur
from .Gaussian_noise import Gaussian_noise
from .Jpeg_compression import Jpeg
from .DiffJPEG_master.DiffJPEG import DiffJPEG
from .Combination import Combination_attack
from .Crop import Crop
from kornia import augmentation as K
from kornia.augmentation import RandomGaussianBlur, RandomGaussianNoise,RandomBrightness, RandomCrop, RandomRotation
import random
import torch
from torchvision import transforms
from compressai.zoo import (bmshj2018_factorized, cheng2020_anchor)

import sys
sys.path.append("..")


_ATTACK = ['c', 'r', 'g', 'b', 'n', 'e', 'j']


def clamp_transform(x, min_val=0, max_val=1):
    return torch.clamp(x, min_val, max_val)

def apply_with_prob(p, transform):
    def apply_transform(x):
        if random.random() > p:
            return x
        return transform(x)
    return apply_transform


def attack_initializer(args, is_train, device):

    if 'AE_' in args.attack:
        attack, quality = args.attack.split('_')[1:]
        if 'b' in attack:
            net = bmshj2018_factorized
        elif 'c' in attack:
            net = cheng2020_anchor

        return net(quality=int(quality), pretrained=True).eval().to(device)

    attack_prob = 0.3 if is_train else 1.
    if args.attack == 'all':
        args.attack = ''.join(_ATTACK)

    assert all(a in _ATTACK for a in args.attack)
    resolution = args.resolution

    # define custom lambda function
    def apply_diffjpeg(x):
        quality = random.choice([50, 60, 70, 80, 90]) if is_train else 50  # randomly select quality parameter
        return DiffJPEG(height=resolution, width=resolution, differentiable=True, quality=quality).to(device)(x)

    aug_list = K.AugmentationSequential(
        K.RandomCrop((resolution,resolution), p=attack_prob if 'c' in args.attack else 0, keepdim=True, padding=int(resolution * random.choice([0.02, 0.05, 0.08, 0.1])), pad_if_needed=True, cropping_mode="resample"), #around maximally 20% cropping
        K.RandomRotation(degrees=(-30, 30), p = attack_prob if 'r' in args.attack else 0, keepdim=True),
        K.RandomGaussianBlur(kernel_size=random.choice([(3,3), (5,5), (7,7)]), sigma=(1,3), p = attack_prob if 'g' in args.attack else 0, keepdim=True),
        K.RandomBrightness((0.7, 1.3), p = attack_prob if 'b' in args.attack else 0, keepdim=True), #0.7
        K.RandomGaussianNoise(mean=0., std = 0.2, p = attack_prob if 'n' in args.attack else 0, keepdim=True),
        K.RandomErasing(p = attack_prob if 'e' in args.attack else 0, keepdim=True),
    )

    attack = transforms.Compose([
        aug_list,
        transforms.Lambda(lambda x: torch.clamp_(x, 0., 1.)),  # use torch.clamp_ to perform operation in-place
        transforms.Lambda(apply_with_prob(attack_prob if 'j' in args.attack else 0, apply_diffjpeg)),  # add conditional transformation
        K.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    return attack
