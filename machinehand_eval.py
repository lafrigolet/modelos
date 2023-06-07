#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
import machinehand_model
import argparse

# create the argument parser
parser = argparse.ArgumentParser(description="Select a random sample of files from a directory")

parser.add_argument('-p', '--path',  type=str, help="path to directory with files to eval")
parser.add_argument('-m', '--model', type=str, help="path to model pth file")
parser.add_argument('-w', '--cropped_width', type=int, help='cropped image width')
parser.add_argument('-t', '--cropped_height', type=int, help='cropped image height')

# parse the arguments
args = parser.parse_args()

# Load the model
mhm = machinehand_model.MachineHandModel()
#print(machinehand_model)
#mhm.load(args.model)
output = mhm.eval(args.model, args.path, args.cropped_width, args.cropped_height)


for t in output:
    print('[{:.4f}, {:.4f}]'.format(t[0][0], t[0][1]))


# ./machinehand_eval.py -p ./machinehand_dataset/train/0  -m machinehand_model.pth -w 150 -t 30

