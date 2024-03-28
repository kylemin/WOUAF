import torch
import torch.nn as nn
import numpy as np
import random



device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

class Gaussian_noise(nn.Module):
    def __init__(self, variance_range, is_train):
        """
        :param height_ratio_range:
        :param width_ratio_range:
        """
        super(Gaussian_noise, self).__init__()

        self.variance_range = variance_range
        self.mean = 0
        self.is_train = is_train

    def forward(self, image):

        if (self.is_train):
            self.std = random.uniform(self.variance_range[0], self.variance_range[1])
        else:
            # In testcase, only using the most intensive attacks.
            self.std = random.uniform(self.variance_range[-1], self.variance_range[-1])

        noised_image = image
        image_size = image.shape

        noise = torch.zeros(image.size()).to(device)
        noise.normal_(self.mean, self.std)

        noised_image = noised_image + noise
        noised_image = torch.clamp(noised_image, -1 , 1)
        noised_image = noised_image.float()
        return noised_image

