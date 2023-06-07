#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
import machinehand_model
import argparse

# create the argument parser
parser = argparse.ArgumentParser(description="Select a random sample of files from a directory")

parser.add_argument('-f', '--file',  type=str, help="path to file to eval")
parser.add_argument('-p', '--pth', type=str, help="path to model pth file")
parser.add_argument('-w', '--cropped_width', type=int, help='cropped image width')
parser.add_argument('-t', '--cropped_height', type=int, help='cropped image height')

# parse the arguments
args = parser.parse_args()

# Load the model
mhm = machinehand_model.MachineHandModel()
#print(machinehand_model)
output = mhm.eval(args.pth, args.file, args.cropped_width, args.cropped_height)

print('Result ', output)



