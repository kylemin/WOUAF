import torch.nn as nn
import random

class Combination_attack(nn.Module):
    def __init__(self, attacks, is_train):
        super(Combination_attack, self).__init__()

        self.attacks = attacks
        self.is_train = is_train

        #If it is training time, attack will be applied for 50% prob.
        #If it is testing time, all attack will be applied.
        if(self.is_train):
            self.attack_threshold = 0.5
        else:
            self.attack_threshold = 0


    def forward(self, image):

        for i in range(len(self.attacks)):
            attack_prob = random.random()  # uniform [0, 1)
            if(attack_prob > self.attack_threshold):
                image = self.attacks[i](image)

        return image
