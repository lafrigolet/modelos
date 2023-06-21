import os
from PIL import Image
import model
from helpers import normalize_images as NI
import cv2
import torchvision.transforms as transforms

class MachineHandModel(model.Model):

    def cook_images(self, path, image_width, image_height):
        files = os.listdir(path)

        to_pytorch_tensor  = transforms.ToTensor()
        cv2_images         = [cv2.imread(path + '/' + file, 0) for file in files]
        pil_images         = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in cv2_images]
        normalized_images  = [NI.normalize_image(img, image_width, image_height) for img in pil_images]
        normalized_tensors = [to_pytorch_tensor(img) for img in normalized_images]
        
        return normalized_tensors

