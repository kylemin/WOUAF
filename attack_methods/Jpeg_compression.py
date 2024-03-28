import torch.nn as nn
from .DiffJPEG_master.DiffJPEG import DiffJPEG


class Jpeg(nn.Module):
    def __init__(self, is_train, param, image_size):
        super(Jpeg, self).__init__()

        if image_size is None:
            raise ValueError("Image size should be passed")
        else:
            self.attack = DiffJPEG(height=image_size, width=image_size, differentiable=is_train, quality=param)

    def forward(self, image):
        image = (image + 1.) / 2. #To put into Jpeg attack
        image = self.attack(image) #output is in [0,1]
        image = (image * 2.) - 1. #Resacling

        return image