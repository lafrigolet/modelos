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
parser.add_argument('-p', '--input_path', type=str, help='Path to the directory with the image files to tensorize')
parser.add_argument('-o', '--output_path', type=str, help='Path to the directory for the tensor files')

# Parse the command-line arguments
args = parser.parse_args()

def tensorize(input_path, output_path):
    print('input_path ', input_path)
    print('output_path ', output_path)

    os.makedirs(output_path, exist_ok = True)

    files = [input_path  + '/' + file for file in os.listdir(input_path)]

    def index(f):
        i, _ = f
        return i // 10
    
    # batch files in size 10 batches
    batches = [list(g) for k, g in groupby(enumerate(files), key=index)]
    i = 0
    normalized_tensors = []
    for batch in batches:
        # print("Iteration ", i)
        # print("====================");
        files = [file for _, file in batch]
        imgs = [Image.open(file) for file in files]
        to_pytorch_tensor  = transforms.ToTensor()
        normalized_tensors = [to_pytorch_tensor(img) for img in imgs]
        # print('normalized_tensors', len(normalized_tensors))

        for tensor in normalized_tensors:
            torch.save(tensor, output_path + '/tensor_' + str(i) + '.pt')
            i += 1

    
tensorize(args.input_path, args.output_path)

"""
./tensorize.py -p ./machinehand_dataset/train/0 -o ./machinehand_dataset.tensor/train/0
./tensorize.py -p ./machinehand_dataset/train/1 -o ./machinehand_dataset.tensor/train/1
./tensorize.py -p ./machinehand_dataset/test/0  -o ./machinehand_dataset.tensor/test/0
./tensorize.py -p ./machinehand_dataset/test/1  -o ./machinehand_dataset.tensor/test/1

./tensorize.py -p ./suicide_dataset.small.handwritten/train/0 -o ./suicide_dataset.small.tensor/train/0
./tensorize.py -p ./suicide_dataset.small.handwritten/train/1 -o ./suicide_dataset.small.tensor/train/1
./tensorize.py -p ./suicide_dataset.small.handwritten/test/0 -o ./suicide_dataset.small.tensor/test/0
./tensorize.py -p ./suicide_dataset.small.handwritten/test/1 -o ./suicide_dataset.small.tensor/test/1

./tensorize.py -p ./suicide_dataset.handwritten/train/0 -o ./suicide_dataset.tensor/train/0
./tensorize.py -p ./suicide_dataset.handwritten/train/1 -o ./suicide_dataset.tensor/train/1
./tensorize.py -p ./suicide_dataset.handwritten/test/0 -o ./suicide_dataset.tensor/test/0
./tensorize.py -p ./suicide_dataset.handwritten/test/1 -o ./suicide_dataset.tensor/test/1

"""
