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
    def __init__(self, _name):
        super().__init__(_name)
        self.results = {n_: [] for n_ in _name}
    
    def setup(self, accelerator, _args):
        self._args = _args
        self.attacks = {n: attack_initializer(_args, n, False, accelerator.device) for n in self._name}

    def infer_batch(self, accelerator=None, decoding_network=None, vae=None, batch=None, outputs=None, phis=None, bsz=None):        
        for n_ in list(self._name):
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


    def get_results(self, accelerator):
        return (torch.mean(torch.tensor(self.results[n_])) for n_ in self._name)

def acc_calculation(args, phis, decoding_network, generated_image, aug, bsz = None,vae = None):
    if args.decoding_network_type == "fc":
        reconstructed_keys = decoding_network(vae.encode(aug((generated_image / 2 + 0.5).clamp(0, 1))).latent_dist.sample().reshape(bsz, -1))
    elif args.decoding_network_type == "resnet":
        reconstructed_keys = decoding_network(aug((generated_image / 2 + 0.5).clamp(0, 1)))
    else:
        raise ValueError("Not suported network")

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


def attack_initializer(args, attack_name, is_train, device):

    attack_prob = 0.5 if is_train else 1.
    resolution = args.resolution
    
    # define custom lambda function
    def apply_diffjpeg(x):
        quality = random.choice([50, 60, 70, 80, 90]) if is_train else 50  # randomly select quality parameter
        return DiffJPEG(height=resolution, width=resolution, differentiable=True, quality=quality).to(device)(x)
    
    aug_list = {
        "rotate": K.RandomRotation(degrees=(-30, 30), p = attack_prob, keepdim=True),
        "blur": K.RandomGaussianBlur(kernel_size=random.choice([(3,3), (5,5), (7,7)]), sigma=(1,3), p = attack_prob, keepdim=True),
        "bright": K.RandomBrightness((0.7, 1.3), p = attack_prob, keepdim=True),
        "noise": K.RandomGaussianNoise(mean=0., std = 0.2, p = attack_prob, keepdim=True),
    }
    augmentation = K.AugmentationSequential(
         aug_list[attack_name]
    )

    attack = transforms.Compose([
        augmentation,
        transforms.Lambda(lambda x: torch.clamp_(x, 0., 1.)),  # use torch.clamp_ to perform operation in-place
        transforms.Lambda(apply_with_prob(0.2 if is_train else 1.1, apply_diffjpeg)),  # add conditional transformation
        K.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    

    return attack