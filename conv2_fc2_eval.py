#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
import conv2_fc2_model as model
import argparse
import loaders

# create the argument parser
parser = argparse.ArgumentParser(description="Select a random sample of files from a directory")

parser.add_argument('-f', '--file', type=str, help="path to file to eval")

# parse the arguments
args = parser.parse_args()

# Load the model
network = model.Net()
network.load_state_dict(torch.load('conv2_fc2_train.py_model.pth'))
network.eval()

image_width      = 150
image_height     = 30
normalized_img = loaders.normalize_png_file(args.file, (image_width, image_height))
data = normalized_img
output = network(data)
#test_loss += F.nll_loss(output, target, size_average=False).item()
pred = output.data.max(1, keepdim=True)[1]
print('Result ', torch.exp(output))

