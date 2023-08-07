import os
from PIL import Image
from helpers import normalize_images as NI
import cv2
import torch
import torchvision.transforms as transforms
import custom_dataset
from models.cnn import CNN

class MachineHandModel(CNN):
    def build_loader(self, path, batchsize, shuffle, image_width, image_height):
        # Usar la base de datos construida
        dataset = custom_dataset.CustomDataset()
        dataset.append_images(self.cook_images(path + '/0', image_width, image_height), 0)
        dataset.append_images(self.cook_images(path + '/1', image_width, image_height), 1)

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batchsize, 
                                             shuffle=shuffle)

        return loader


    def cook_images(self, path, image_width, image_height):
        files = os.listdir(path)

        to_pytorch_tensor  = transforms.ToTensor()
        cv2_images         = [cv2.imread(path + '/' + file, 0) for file in files]
        pil_images         = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in cv2_images]
        normalized_images  = [NI.normalize_image(img, image_width, image_height) for img in pil_images]
        normalized_tensors = [to_pytorch_tensor(img) for img in normalized_images]
        
        return normalized_tensors

