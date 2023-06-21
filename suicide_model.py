import os
from PIL import Image
import model
import machinehand_model as MHM
from helpers import line_selector as LS
import cv2
import torch
import torchvision.transforms as transforms
from helpers import normalize_images as NI
from matplotlib import pyplot as plt

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

    def hand_written(self, img, image_width, image_height):
        output = self.machinehand_model.eval_image(img, image_width, image_height)
        #            print(output)

        return output[0][0] < output[0][1]  # output[0] is probability of machine written, output[1] hand written
    
    def crop_image(self, img, image_width, image_height):
        page                  = LS.handwritten_text_line_detection(img,dpi=300,break_connected_lines=False,
                                                                   dilate_ink=True,min_word_dimmension=10)
        numpy_cropped_images  = LS.crop_page(page, image_height, image_width)
        
        return numpy_cropped_images

    def preprocess(self, img, image_width, image_height):
        cv2_cropped_images = self.crop_image(img, image_width, image_height)
        pil_cropped_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in cv2_cropped_images]
        normalized_images  = [NI.normalize_image(img, image_width, image_height) for img in pil_cropped_images]
        handwritten_images = [img for img in normalized_images if self.hand_written(img, image_width, image_height)]

        """
        # Convert the tensor to a PIL image

        transform = transforms.ToPILImage()
        for i,tensor in enumerate(handwritten_images):
            pil_image = transform(tensor)
            pil_image.save('temp/img_{}.jpg'.format(i))
        """
        return handwritten_images
    
    def cook_images_bak(self, path, image_width, image_height):
        files = os.listdir(path)
        handwritten_images = []
        
        for file in files:
            print(path + '/' + file)
            img = cv2.imread(path + '/' + file, 0)
            handwritten_images += self.preprocess(img, image_width, image_height)
            
        print('len(handwritten_images): ', len(handwritten_images))
        
        return handwritten_images


    def cook_images_bak2(self, path, image_width, image_height):
        print(path)
        path_list = path.split('/')
        path_list[1] += '.cooked'
        cooked_path = '/'.join(path_list)
        print(cooked_path)


        if not os.path.exists(cooked_path) or not os.listdir(cooked_path):
            os.makedirs(cooked_path, exist_ok = True)
            files = os.listdir(path)
            i = 0
            for file in files:
                print(path + '/' + file)
                img = cv2.imread(path + '/' + file, 0)
                cv2_cropped_images = self.crop_image(img, image_width, image_height)
                for cropped_image in cv2_cropped_images:
                    cv2.imwrite(cooked_path + '/cropped_image_' + str(i) + '.jpg', cropped_image)
                    i += 1

        cooked_images      = os.listdir(cooked_path)
        cv2_cropped_images = [cv2.imread(cooked_path + '/' + file, 0) for file in cooked_images]
        pil_cropped_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in cv2_cropped_images]
        normalized_images  = [NI.normalize_image(img, image_width, image_height) for img in pil_cropped_images]
        handwritten_images = [img for img in normalized_images if self.hand_written(img, image_width, image_height)]
        
        print('len(handwritten_images): ', len(handwritten_images))

        return handwritten_images


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
                cv2_cropped_images = self.crop_image(img, image_width, image_height)
                pil_cropped_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in cv2_cropped_images]
                normalized_images  = [NI.normalize_image(img, image_width, image_height) for img in pil_cropped_images]
                handwritten_images = [img for img in normalized_images if self.hand_written(img, image_width, image_height)]
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

    
    def suicide_score(self, file, image_height, image_width):
        self.suicide_model = self.network
#        if self.suicide_model == None:
#            raise Exception('No suicide model specified to eval suicide_score')
                
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
        self.suicide_model = self.network
#        if self.suicide_model == None:
#            raise Exception('No suicide model specified to eval suicide_score')
                
        def suicide(img):
            output = self.eval_image(img, image_width, image_height)
            return output.data[0][1].item()  # output[0][1] is probability of suicide


        img                = cv2.imread(file, 0)
        handwritten_images = self.preprocess(img, image_width, image_height)
        suicide_scores     = [suicide(img) for img in handwritten_images]

        histogram = [0 for _ in range(10)]
        for score in suicide_scores:
            index = int(round(score, 1) * 10)
            histogram[index] += 1

        print(histogram)

        return histogram
