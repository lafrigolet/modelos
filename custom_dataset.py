import torch
import os
from PIL import Image
import cv2
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
import numpy
from helpers import normalize_images as NI

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
        assert all(isinstance(img, torch.Tensor) for img in images)

        self.list_img   += images
        self.list_label += [label for _ in range(len(images))]

    def append_dir(self, cooked_path, label, image_width, image_height):
        to_pytorch_tensor      = transforms.ToTensor()
        files                  = os.listdir(cooked_path)
        cv2_handwritten_images = [cv2.imread(cooked_path + '/' + file, 0) for file in files]
        pil_handwritten_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in cv2_handwritten_images]
        normalized_images      = [NI.normalize_image(img, image_width, image_height) for img in pil_handwritten_images]
        normalized_tensors     = [to_pytorch_tensor(img) for img in normalized_images]

        self.list_img   += normalized_tensors
        self.list_label += [label for _ in range(len(normalized_tensors))]

    
    def __getitem__(self, index):
        return (self.list_img[index], self.list_label[index])
        
    def __len__(self):
        return len(self.list_img)




