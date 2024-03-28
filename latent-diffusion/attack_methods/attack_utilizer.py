import torch
import torch.nn.functional as F

#Currently, this code is not referenced
#You can ignore this code.

def to_PNG(image):
    min = -1
    max = 1

    # Normalize to [0,1]
    img = image
    img = torch.clamp(img, -1, 1)
    img = img.add(-min).div(max - min + 1e-5)

    # To PNG
    img = torch.clamp(img.mul(255).add(0.5), 0, 255)

    return img

def to_gan_tensor(image):
    img = image
    # To [0,1] again
    img = img.div(255)
    # To [-1,1] Like dataloader
    img = (img - 0.5) / 0.5


def actual_scenario_normalization(image):

    img = to_PNG(image)
    img = to_gan_tensor(img)

    return img