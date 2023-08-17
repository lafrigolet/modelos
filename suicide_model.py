import os
from PIL import Image
import machinehand_model as MHM
from helpers import line_selector as LS
import cv2
import torch
import torchvision.transforms as transforms
from helpers import normalize_images as NI
from matplotlib import pyplot as plt
import custom_dataset
import math
import statistics as stats
from models.cnn import CNN

class SuicideModel(CNN):
    def __init__(self, machinehand_model):  # For training the model
        super().__init__()
        # Load the model
        self.machinehand_model = MHM.MachineHandModel()
        self.machinehand_model.load(machinehand_model)

        self.suicide_model = None


    def hand_written(self, img, image_width, image_height):
        output = self.machinehand_model.eval_image(img, image_width, image_height)
        #            print(output)

        return output[0][0] < output[0][1]  # output[0] is probability of machine written, output[1] hand written
    

    def preprocess(self, img, image_width, image_height):
        cv2_cropped_images = crop_image(img, image_width, image_height)
        pil_cropped_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in cv2_cropped_images]
        normalized_images  = [NI.normalize_image(img, image_width, image_height) for img in pil_cropped_images]
        handwritten_images = [img for img in normalized_images if self.hand_written(img, image_width, image_height)]

        return handwritten_images
    

    
    def suicide_score(self, file, image_height, image_width):
                
        def suicide(img):
            output = self.eval_image(img, image_width, image_height)
            return output.data[0][1]  # output[0][1] is probability of suicide


        img                = cv2.imread(file, 0)
        handwritten_images = self.preprocess(img, image_width, image_height)
        suicide_scores     = [suicide(img) for img in handwritten_images]
        print(suicide_scores)
        suicide_score      = sum(suicide_scores) / len(suicide_scores)
        print('Suicide Score for ', file, ': ', suicide_score)
        return suicide_score


    def histogram(self, file, image_height, image_width):

        def suicide(img):
            output = self.eval_image(img, image_width, image_height)
            return output.data[0][1].item()  # output[0][1] is probability of suicide


        img                = cv2.imread(file, 0)
        handwritten_images = self.preprocess(img, image_width, image_height)
        suicide_scores     = [suicide(img) for img in handwritten_images]

        histogram = [0 for _ in range(11)]
        for score in suicide_scores:
            index = int(round(score, 1) * 10)
            histogram[index] += 1

        print(histogram)

        return histogram

    def cook_images(self, path, image_width, image_height):
        print(path)
        path_list = path.split('/')
        path_list[1] += '.cooked'
        cooked_path = '/'.join(path_list)
        print(cooked_path)

        handwritten_images = []
        if not os.path.exists(cooked_path) or not os.listdir(cooked_path):
            os.makedirs(cooked_path, exist_ok = True)
            files = os.listdir(path)
            i = 0
            for file in files:
                print(path + '/' + file)
                img = cv2.imread(path + '/' + file, 0)
                handwritten_images = self.preprocess(img, image_width, image_height)
                for handwritten_image in handwritten_images:
                    handwritten_image.save(cooked_path + '/handwritten_image_' + str(i) + '.jpg')
                    i += 1

        to_pytorch_tensor      = transforms.ToTensor()
        files                  = os.listdir(cooked_path)
        cv2_handwritten_images = [cv2.imread(cooked_path + '/' + file, 0) for file in files]
        pil_handwritten_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in cv2_handwritten_images]
        normalized_images      = [NI.normalize_image(img, image_width, image_height) for img in pil_handwritten_images]
        normalized_tensors     = [to_pytorch_tensor(img) for img in normalized_images]
            
        print('len(normalized_tensors): ', len(normalized_tensors))

        return normalized_tensors

    def build_loader(self, path, batchsize, shuffle, image_width, image_height):
        # Usar la base de datos construida
        dataset = custom_dataset.CustomDataset()
        dataset.append_images(self.cook_images(path + '/0', image_width, image_height), 0)
        dataset.append_images(self.cook_images(path + '/1', image_width, image_height), 1)
        
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batchsize, 
                                             shuffle=shuffle)
        
        return loader


def crop_image(img, image_width, image_height):
        page                  = LS.handwritten_text_line_detection(img,dpi=300,break_connected_lines=False,
                                                                   dilate_ink=True,min_word_dimmension=10)
        numpy_cropped_images  = LS.crop_page(page, image_height, image_width)
        
        return numpy_cropped_images

    
def mode(path, file, label, image_width, image_height, batchsize):
        def cook_images(path, file, image_width, image_height):
            img = cv2.imread(path + '/' + file, 0)
            cv2_cropped_images = crop_image(img, image_width, image_height)
            pil_cropped_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in cv2_cropped_images]
            normalized_images  = [NI.normalize_image(img, image_width, image_height) for img in pil_cropped_images]
            handwritten_images = [img for img in normalized_images if self.hand_written(img, image_width, image_height)]
            to_pytorch_tensor  = transforms.ToTensor()
            normalized_tensors = [to_pytorch_tensor(img) for img in normalized_images]
            
            return normalized_tensors

        def image_to_loader(images, label, batchsize):
            dataset = custom_dataset.CustomDataset()
            dataset.append_images(images, label)
    
            loader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=batchsize, 
                                                 shuffle=False)
    
            return loader


        images = cook_images(path, file, image_width, image_height)
        loader = image_to_loader(images, label, batchsize)
        results, labels, test_loss, correct = self.test(loader)
        
        results = [math.exp(result[1]) for result in results]
        results = [round(result, 1)    for result in results]
        
        mode    =  stats.mode(results) # statistical mode
        
        return mode, label



    
