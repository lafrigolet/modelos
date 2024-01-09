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

# Create the parser
parser = argparse.ArgumentParser(description='Command-line argument parser')

# Add arguments
parser.add_argument('-i', '--input', type=str, help='Path to the directory with the input files')
parser.add_argument('-o', '--output', type=str, help='Path to the directory for the cropped files')
parser.add_argument('-w', '--width', type=int, help='Cropped image width')
parser.add_argument('-t', '--height', type=int, help='Cropped image height')

# Parse the command-line arguments
args = parser.parse_args()

def crop_image(img, image_width, image_height):
    page = LS.handwritten_text_line_detection(img,dpi=300,break_connected_lines=False,
                                              dilate_ink=True,min_word_dimmension=10)
    numpy_cropped_images  = LS.crop_page(page, image_height, image_width)
        
    return numpy_cropped_images


def crop_text(input_path, cropped_path, image_width, image_height):
    print('input path ', input_path)
    print('cropped path ', cropped_path)

    os.makedirs(cropped_path, exist_ok = True)

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
        print("Iteration ", i)
        print("====================");
        for _, file in batch:
            cv2_image = cv2.imread(file, 0)
            cv2_cropped_images = crop_image(cv2_image, image_width, image_height)
            print('cv2_cropped_images ', len(cv2_cropped_images))
            pil_cropped_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in cv2_cropped_images]
            print('pil_cropped_images ', len(pil_cropped_images))
            normalized_images  = [NI.normalize_image(img, image_width, image_height) for img in pil_cropped_images]
            print('normalized_images ', len(normalized_images))

            base_name = os.path.basename(file)
            filename_without_extension, _ = os.path.splitext(base_name)
            for i, crop in enumerate(normalized_images):
                filename = cropped_path + '/' + filename_without_extension + '.' + str(i) + '.jpg' 
                # print('Saving ', filename)
                crop.save(filename)
                i += 1
            
    
crop_text(args.input, args.output, args.width, args.height)

# This script make crops text into boxes of (width, height) according to Jose algorithm
# ./crop.py -i ./suicide_dataset.small/train/0 -o ./suicide_dataset.small.cropped/train/0 -w 150 -t 40
# ./crop.py -i ./suicide_dataset.small/train/1 -o ./suicide_dataset.small.cropped/train/1 -w 150 -t 40
# ./crop.py -i ./suicide_dataset.small/test/0 -o ./suicide_dataset.small.cropped/test/0 -w 150 -t 40
# ./crop.py -i ./suicide_dataset.small/test/1 -o ./suicide_dataset.small.cropped/test/1 -w 150 -t 40

# ./crop.py -i ./suicide_dataset/train/0 -o ./suicide_dataset.cropped/train/0 -w 150 -t 40
# ./crop.py -i ./suicide_dataset/train/1 -o ./suicide_dataset.cropped/train/1 -w 150 -t 40
# ./crop.py -i ./suicide_dataset/test/0 -o ./suicide_dataset.cropped/test/0 -w 150 -t 40
# ./crop.py -i ./suicide_dataset/test/1 -o ./suicide_dataset.cropped/test/1 -w 150 -t 40

#
#  ./crop.py -i ./machinehand_dataset/train/0 -o ./machinehand_dataset.cropped/train/0 -w 150 -t 40
