#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
import conv2_fc2_model as model
import argparse
import loaders

# create the argument parser
parser = argparse.ArgumentParser(description="Select a random sample of files from a directory")

parser.add_argument('-f', '--file',  type=str, help="path to file to eval")
parser.add_argument('-m', '--model', type=str, help="path to model file")

# parse the arguments
args = parser.parse_args()

# Load the model
network = model.Net()
model = loaders.Model(network)
model.load(args.model)

image_width    = 150
image_height   = 30
model.eval(args.file, image_width, image_height)


