import torch
import os
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys


# Construcción de la base de datos.
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):

        # 1. Inicialización de rutas y nombres de ficheros
        self.lista_imagenes = []
        self.list_img       = []
        self.list_label     = []

        # print(self.lista_imagenes)
        # print(len(self.lista_imagenes))
        # print(len(self.list_img))
        # print(len(self.list_label))
        
        pass

    def append_images(self, images, label):
        self.list_img   += images;
        self.list_label += [label for _ in range(len(images))]
    
    def __getitem__(self, index):
        return (self.list_img[index], self.list_label[index])
        
    def __len__(self):
        return len(self.list_img)




