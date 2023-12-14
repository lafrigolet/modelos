#!/usr/bin/env python3
# coding: utf-8

# From a path containing .jpg files of width, height size it builds
# .cooked.pt path with converted jpgs to tensor files .pt


import argparse
import os
from itertools import groupby
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch

from helpers import line_selector as LS
from helpers import normalize_images as NI
import machinehand_model as MHM

# Create the parser
parser = argparse.ArgumentParser(description='Command-line argument parser')

# Add arguments
parser.add_argument('-p', '--input_path', type=str, help='Path to the directory with the files')
parser.add_argument('-m', '--mhm', type=str, help='Path to the machine hand model .pth file')
parser.add_argument('-w', '--width', type=int, help='Cropped image width')
parser.add_argument('-t', '--height', type=int, help='Cropped image height')

# Parse the command-line arguments
args = parser.parse_args()

def crop_image(img, image_width, image_height):
    page = LS.handwritten_text_line_detection(img,dpi=300,break_connected_lines=False,
                                              dilate_ink=True,min_word_dimmension=10)
    numpy_cropped_images  = LS.crop_page(page, image_height, image_width)
        
    return numpy_cropped_images

def hand_written(mhm, img, image_width, image_height):
    output = mhm.eval_image(img, image_width, image_height)
    return output[0][0] < output[0][1]  # output[0] is probability of machine written, output[1] hand written

def cook_images(path, mhm_file, image_width, image_height):
    mhm = MHM.MachineHandModel()
    mhm.load(mhm_file)

    # Check if a GPU is available
    if torch.cuda.is_available():
        # Move the model to the GPU
        mhm.to('cuda')
        print("Model is using GPU.")
    else:
        mhm.to('cpu')
        print("Model is using CPU.")
    
    
    print(path)
    path_list = path.split('/')
    path_list[1] += '.cooked.pt'
    cooked_path = '/'.join(path_list)
    print(cooked_path)

    os.makedirs(cooked_path, exist_ok = True)

    files = [path  + '/' + file for file in os.listdir(path)]

    def index(f):
        i, _ = f
        return i // 10
    
    # batch files in size 10 batches
    batches = [list(g) for k, g in groupby(enumerate(files), key=index)]
    i = 0
    normalized_tensors = []
    for batch in batches:
        print("Iteration ", i)
        print("====================");
        files = [file for _, file in batch]
        cv2_images = [cv2.imread(file, 0) for file in files]
        print('cv2_images ', len(cv2_images))
        cv2_cropped_images = []
        for img in cv2_images:
            cv2_cropped_images += crop_image(img, image_width, image_height)
        print('cv2_cropped_images ', len(cv2_cropped_images))
        pil_cropped_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in cv2_cropped_images]
        print('pil_cropped_images ', len(pil_cropped_images))
        normalized_images  = [NI.normalize_image(img, image_width, image_height) for img in pil_cropped_images]
        print('normalized_images ', len(normalized_images))
        handwritten_images = [img for img in normalized_images if hand_written(mhm, img, image_width, image_height)]
        print('handwritten_images ', len(handwritten_images))
        to_pytorch_tensor  = transforms.ToTensor()
        normalized_tensors = [to_pytorch_tensor(img) for img in handwritten_images]
        print('normalized_tensors', len(normalized_tensors))

        for tensor in normalized_tensors:
            torch.save(tensor, cooked_path + '/handwritten_' + str(i) + '.pt')
            i += 1

    
cook_images(args.path, args.mhm, args.width, args.height)

#  ./cook_images.py -p ./suicide_dataset.small/train/0 -m ./machinehand_model.pth -w 150 -t 30
