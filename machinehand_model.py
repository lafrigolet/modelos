import torch
import os
from PIL import Image
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
import custom_dataset
import model
from helpers import normalize_images as NI

class MachineHandModel(model.Model):

    def cook_images(self, path, label, image_width, image_height):
        files = os.listdir(path)
        normalized_images = []
            
        for file in files:
            img = Image.open(path + '/' + file)
            normalized_images.append(NI.normalize_image(img, image_width, image_height))

        return normalized_images

