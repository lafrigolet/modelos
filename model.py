import torch
import os
from PIL import Image
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
import custom_dataset
from helpers import normalize_images as NI
from models.cnn import CNN
import torchvision.transforms.functional as TF

class Model():
    def __init__(self):
        self.network       = CNN()

        # Move the model to the GPU if available
        if torch.cuda.is_available():
            self.network.to(torch.device("cuda"))

        # Check the device of the model's parameters
        param_device = next(self.network.parameters()).device

        if param_device.type == "cuda":
            print("Model is using GPU.")
        else:
            print("Model is using CPU.")

    def load(self, pth):
        self.network.load(pth)

    def cook_images(self, path, label, image_width, image_height): # template method
        raise NotImplementedError()

    def build_loader(self, path, batchsize, shuffle, image_width, image_height):
        # Usar la base de datos construida
        dataset = custom_dataset.CustomDataset()
        dataset.append_images(self.cook_images(path + '/0', image_width, image_height), 0)
        dataset.append_images(self.cook_images(path + '/1', image_width, image_height), 1)

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batchsize, 
                                             shuffle=shuffle)

        return loader

    def save(self, output_pth_file):
        self.network.save(output_pth_file)
        
    def train(self, train_loader, test_loader, n_epochs, learning_rate):
        self.network.train_model(train_loader, test_loader, n_epochs, learning_rate)

    def test(self, loader):
        return self.network.test(loader)

    def eval_image(self, img, image_width, image_height):
        self.network.eval()

        assert isinstance(img, Image.Image)

        img = TF.to_tensor(img)

        img = img.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                       
        return torch.exp(self.network(img))
                         
    def eval(self, path, image_width, image_height):
        dataset = custom_dataset.CustomDataset()
        dataset.append_images(self.cook_images(path, image_width, image_height), 0) # label doesn't matter
        self.network.eval()

        with torch.no_grad():
            data = data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            result = [torch.exp(self.network(data)) for data, target in dataset]
            
        return result
