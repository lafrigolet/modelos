import os
from PIL import Image
import model
import machinehand_model as MHM
from helpers import line_selector as LS
import cv2
import torch
import torchvision.transforms as transforms
from helpers import normalize_images as NI
#import numpy as np

class SuicideModel(model.Model):
    def __init__(self, machinehand_model):  # For training the model
        super().__init__()
        # Load the model
        self.machinehand_model = MHM.MachineHandModel()
        self.machinehand_model.load(machinehand_model)

        self.suicide_model = None
    """
    def __Init__(self, machinehand_model, suicide_model): # To eval width the Model
        # Load the model
        self.machinehand_model = loaders.Model(model.Net(), None)
        self.machinehand_model.load(machinehand_model)
        
        self.suicide_model = loaders.Model(model.Net(), None)
        self.suicide_model.load(suicide_model)
    """

    def cook_images(self, path, label, image_width, image_height):
        def hand_written(img):
            output = self.machinehand_model.eval_image(img, image_width, image_height)
#            print(output)
            return output[0][0] < output[0][1]  # output[0] is probability of machine written, output[1] hand written

        def crop_image(img, image_width, image_height):
            page                  = LS.handwritten_text_line_detection(img,dpi=300,break_connected_lines=False,
                                                                       dilate_ink=True,min_word_dimmension=10)
            numpy_cropped_images  = LS.crop_page(page, image_height, image_width)

            return numpy_cropped_images
        

        files = os.listdir(path)
        handwritten_images = []
        
        for file in files:
            print(path + '/' + file)
            img = cv2.imread(path + '/' + file, 0)
            cv2_cropped_images = crop_image(img, image_width, image_height)
            pil_cropped_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in cv2_cropped_images]
            normalized_images  = [NI.normalize_image(img, image_width, image_height) for img in pil_cropped_images]
            handwritten_images += [img for img in normalized_images if hand_written(img)]

        print('len(handwritten_images): ', len(handwritten_images))
        
        return handwritten_images

        
        
    def suicide_score(self, file, height, width):
        if self.suicide_model == None:
            raise Exception('No suicide model specified to eval suicide_score')
                
        def hand_written(file):
            output = self.machinehand_model.eval_image(file, image_width, image_height)
            return output[0] < output[1]  # output[0] is probability of machine written, output[1] hand written

        def suicide(file):
            output = suicide_model.eval(file, image_width, image_height)
            return output[1]  # output[1] is probability of suicide

        img                 = cv2.imread(file, 0)
        page                = handwritten_text_line_detection(img,dpi=300,break_connected_lines=False,
                                                              dilate_ink=True,min_word_dimmension=10)
        cropped_images      = crop_page(page, height, width)
        hand_written_images = list(filter(hand_written, cropped_images))
        suicide_images      = list(map(suicide, hand_written_images))
        suicide_score       = sum(suicide_images) / len(suicide_images)
        print('Suicide Score for ' + file + ': ' + suicide_score)
        return suicide_score

