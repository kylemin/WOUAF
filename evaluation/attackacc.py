from attack_methods.Gaussian_blur import Gaussian_blur
from attack_methods.Gaussian_noise import Gaussian_noise
from attack_methods.Jpeg_compression import Jpeg
from attack_methods.DiffJPEG_master.DiffJPEG import DiffJPEG
from attack_methods.Combination import Combination_attack
from attack_methods.Crop import Crop
from kornia.augmentation import RandomGaussianBlur, RandomGaussianNoise,RandomBrightness, RandomCrop, RandomRotation
import random
from kornia import augmentation as K
import torch
from torchvision import transforms

from evaluation.utils import MetricModel
import sys
sys.path.append("..")


class AttackBitAcc(MetricModel):
    def __init__(self, _name, data=None):
        super().__init__(_name)
        self.data = data
        self.initialize(_name)

    def initialize(self, _name):
        if not self.data:
            self.results = {n_: [] for n_ in _name}
        else:
            self.results = {}
            for n in _name:
                self.results[n]={k: [] for en,k in enumerate(self.data[n])}

    def setup(self, accelerator, _args):
        self._args = _args
        if not self.data:
            self.attacks = {n: attack_initializer(_args, n, False, accelerator.device) for n in self._name}
        else:
            self.attacks = {}
            for n in self._name:
                self.attacks[n]={k: attack_initializer(_args, n, False, accelerator.device, k) for en,k in enumerate(self.data[n])}


    def infer_batch(self, accelerator=None, decoding_network=None, vae=None, batch=None, outputs=None, phis=None, bsz=None):
        for n_ in list(self._name):
            if not self.data:
                self.results[n_].extend(
                    acc_calculation(
                        self._args,
                        phis,
                        decoding_network,
                        outputs['images'],
                        self.attacks[n_],
                        bsz,
                        vae
                    )
                )
            else:
                for k in list(self.attacks[n_].keys()):
                    self.results[n_][k].extend(
                        acc_calculation(
                            self._args,
                            phis,
                            decoding_network,
                            outputs['images'],
                            self.attacks[n_][k],
                            bsz,
                            vae
                        )
                    )


    def get_results(self, accelerator):
        if not self.data:
            results = (torch.mean(torch.tensor(self.results[n_])) for n_ in self._name)
            self.initialize(self._name)
            return results

        results = []
        for n_ in self._name:
            tmp = []
            for k, v in self.results[n_].items():
                tmp.append(torch.mean(torch.tensor(v)))
            results.append(tuple(tmp))
        self.initialize(self._name)
        return results

def acc_calculation(args, phis, decoding_network, generated_image, aug, bsz = None,vae = None):
    reconstructed_keys = decoding_network(aug(generated_image))
    gt_phi = (phis > 0.5).int()
    reconstructed_keys = (torch.sigmoid(reconstructed_keys) > 0.5).int()
    bit_acc = ((gt_phi == reconstructed_keys).sum(dim=1)) / args.phi_dimension

    return bit_acc

def clamp_transform(x, min_val=0, max_val=1):
    return torch.clamp(x, min_val, max_val)

def apply_with_prob(p, transform):
    def apply_transform(x):
        if random.random() > p:
            return x
        return transform(x)
    return apply_transform


def attack_initializer(args, attack_name, is_train, device, attack_value=None):

    attack_prob = 0.2 if is_train else 1.
    resolution = args.resolution

    # define custom lambda function
    def apply_diffjpeg(x):
        if attack_value:
            quality = attack_value
        else:
            quality = random.choice([50, 60, 70, 80, 90]) if is_train else 50  # randomly select quality parameter
        return DiffJPEG(height=resolution, width=resolution, differentiable=True, quality=quality).to(device)(x)

    if not attack_value:
        aug_list = K.AugmentationSequential(
            K.RandomCrop((resolution,resolution), p=attack_prob if attack_name == "crop" else 0, keepdim=True, padding=int(resolution * random.choice([0.02, 0.05, 0.08, 0.1])), pad_if_needed=True, cropping_mode="resample"), #around maximally 20% cropping
            K.RandomRotation(degrees=(-30, 30), p = attack_prob if attack_name == "rotate" else 0, keepdim=True),
            K.RandomGaussianBlur(kernel_size=random.choice([(3,3), (5,5), (7,7)]), sigma=(1,3), p = attack_prob if attack_name == "blur" else 0, keepdim=True),
            K.RandomBrightness((0.7, 1.3), p = attack_prob if attack_name == "bright" else 0, keepdim=True), #0.7
            K.RandomGaussianNoise(mean=0., std = 0.2, p = attack_prob if attack_name == "noise" else 0, keepdim=True),
            K.RandomErasing(p = attack_prob if attack_name == "erase" else 0, keepdim=True),
        )
        attack = transforms.Compose([
            aug_list,
            transforms.Lambda(lambda x: torch.clamp_(x, 0., 1.)),  # use torch.clamp_ to perform operation in-place
            transforms.Lambda(apply_with_prob(attack_prob if attack_name == "jpeg" else 0, apply_diffjpeg)),  # add conditional transformation
            K.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    else:
        aug_list = None
        if attack_name=="crop":
            aug_list = (
                K.RandomCrop((resolution,resolution), p=1., keepdim=True, padding=int(resolution * attack_value), pad_if_needed=True, cropping_mode="resample")
            )
        elif attack_name=="rotate":
            aug_list = (
                K.RandomRotation(degrees=(attack_value, attack_value), p=1., keepdim=True)
            )
        elif attack_name=="blur":
            aug_list = (
                K.RandomGaussianBlur(kernel_size=(attack_value, attack_value), p=1., sigma=(1,3), keepdim=True)
            )
        elif attack_name=="bright":
            bright_value = 1.0 + attack_value
            aug_list = (
                K.RandomBrightness((bright_value, bright_value),p=1., keepdim=True)
            )
        elif attack_name=="noise":
            aug_list = (
                K.RandomGaussianNoise(mean=0., std = attack_value, p=1., keepdim=True)
            )
        elif attack_name=="erase":
            aug_list = (
                K.RandomErasing(scale=(attack_value, attack_value), p=1., keepdim=True)
            )

        if attack_name=="jpeg":
            attack = transforms.Compose([
                transforms.Lambda(lambda x: torch.clamp_(x, 0., 1.)),  # use torch.clamp_ to perform operation in-place
                transforms.Lambda(apply_with_prob(1. if attack_name == "jpeg" else 0, apply_diffjpeg)),  # add conditional transformation
                K.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            attack = transforms.Compose([
                aug_list,
                transforms.Lambda(lambda x: torch.clamp_(x, 0., 1.)),  # use torch.clamp_ to perform operation in-place
                transforms.Lambda(apply_with_prob(1. if attack_name == "jpeg" else 0, apply_diffjpeg)),  # add conditional transformation
                K.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

        if attack_name == "all":
            aug_list = K.AugmentationSequential(
                K.RandomCrop((resolution,resolution), p=1.,keepdim=True, padding=int(resolution * attack_value[0]), pad_if_needed=True, cropping_mode="resample"),
                K.RandomRotation(degrees=(attack_value[1], attack_value[1]), keepdim=True, p=1.),
                K.RandomGaussianBlur(kernel_size=(attack_value[2], attack_value[2]), sigma=(1,3), keepdim=True, p=1.),
                K.RandomBrightness((1. + attack_value[3], 1. + attack_value[3]), keepdim=True, p=1.),
                K.RandomGaussianNoise(mean=0., std = attack_value[4], keepdim=True, p=1.),
                K.RandomErasing(scale=(attack_value[5], attack_value[5]), keepdim=True, p=1.),
            )

            attack = transforms.Compose([
                aug_list,
                transforms.Lambda(lambda x: torch.clamp_(x, 0., 1.)),  # use torch.clamp_ to perform operation in-place
                transforms.Lambda(apply_with_prob(1., DiffJPEG(height=resolution, width=resolution, differentiable=True, quality=attack_value[6]).to(device))),
                K.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

    return attack
