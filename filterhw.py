#!/usr/bin/env python3
# coding: utf-8

# From a path containing .jpg files of width, height size it builds
# .handwritten path with converted handwritten filtered jpgs 


import argparse
import os
from itertools import groupby
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch

from helpers import line_selector as LS
from helpers import normalize_images as NI
import models.cnn as CNN

# Create the parser
parser = argparse.ArgumentParser(description='Command-line argument parser')

# Add arguments
parser.add_argument('-i', '--input_path', type=str, help='Path to the directory with the images to filter')
parser.add_argument('-o', '--output_path', type=str, help='Path to the directory for the filtered images')
parser.add_argument('-m', '--mhm', type=str, help='Path to the machine hand model .pth file')

# Parse the command-line arguments
args = parser.parse_args()
print('args ', args)

def hand_written(mhm, img):
    output = CNN.eval(mhm, img)
    return output[0][0] < output[0][1]  # output[0] is probability of machine written, output[1] hand written

def handwritten_filter(input_path, output_path, pth):
    print('input path ', input_path)
    print('output path ', output_path)
    print('mhm_file ', pth)
    mhm = CNN.CNN(40, 150)
    mhm.load_state_dict(torch.load(pth))
    mhm.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    os.makedirs(output_path, exist_ok = True)

    files = [input_path  + '/' + file for file in os.listdir(input_path)]

    def index(f):
        i, _ = f
        return i // 10
    
    # batch files in size 10 batches
    # files need to be batched for avoiding memory fullfilling
    batches = [list(g) for k, g in groupby(enumerate(files), key=index)]
    i = 0
    normalized_tensors = []
    for batch in batches:
        # print("Iteration ", i)
        # print("====================");
        files = [file for _, file in batch]
        # cv2_images = [cv2.imread(file, 0) for file in files]
        # print('cv2_images ', len(cv2_images))
        # cv2_cropped_images = []
        # for img in cv2_images:
        #     cv2_cropped_images += crop_image(img, image_width, image_height)
        # print('cv2_cropped_images ', len(cv2_cropped_images))
        # pil_cropped_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in cv2_cropped_images]
        # print('pil_cropped_images ', len(pil_cropped_images))
        # normalized_images  = [NI.normalize_image(img, image_width, image_height) for img in pil_cropped_images]
        normalized_images = [Image.open(file) for file in files]
        # print('normalized_images ', len(normalized_images))
        handwritten_images = [img for img in normalized_images if hand_written(mhm, img)]
        # print('handwritten_images ', len(handwritten_images))

        for img in handwritten_images:
            filename = output_path + '/handwritten_' + str(i) + '.jpg'
            # print('Saving ', filename)
            img.save(filename)
            i += 1

    
handwritten_filter(args.input_path, args.output_path, args.mhm)

# ./filterhw.py -i ./suicide_dataset.small.cropped/train/0 -o ./suicide_dataset.small.handwritten/train/0 -m ./machinehand_model.pth
# ./filterhw.py -i ./suicide_dataset.small.cropped/train/1 -o ./suicide_dataset.small.handwritten/train/1 -m ./machinehand_model.pth
# ./filterhw.py -i ./suicide_dataset.small.cropped/test/0 -o ./suicide_dataset.small.handwritten/test/0 -m ./machinehand_model.pth
# ./filterhw.py -i ./suicide_dataset.small.cropped/test/1 -o ./suicide_dataset.small.handwritten/test/1 -m ./machinehand_model.pth

# ./filterhw.py -i ./suicide_dataset.cropped/train/0 -o ./suicide_dataset.handwritten/train/0 -m ./machinehand_model.pth
# ./filterhw.py -i ./suicide_dataset.cropped/train/1 -o ./suicide_dataset.handwritten/train/1 -m ./machinehand_model.pth
# ./filterhw.py -i ./suicide_dataset.cropped/test/0 -o ./suicide_dataset.handwritten/test/0 -m ./machinehand_model.pth
# ./filterhw.py -i ./suicide_dataset.cropped/test/1 -o ./suicide_dataset.handwritten/test/1 -m ./machinehand_model.pth
