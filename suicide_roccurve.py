#!/usr/bin/env python3
# coding: utf-8

import torch
import suicide_model as SM
import models.utils as MU
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Command-line argument parser')

# Add arguments
parser.add_argument('-p', '--path', type=str, help='Path to the directory with files')
parser.add_argument('-m', '--machinehand_model', type=str, help='pth file for machine hand written model')
parser.add_argument('-s', '--suicide_model', type=str, help='pth file for suicide model')
parser.add_argument('-w', '--cropped_width', type=int, help='cropped image width')
parser.add_argument('-t', '--cropped_height', type=int, help='cropped image height')

# Parse the command-line arguments
args = parser.parse_args()

# PREPARING THE DATASET
random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)

# Build and train the model
suicide_model = SM.SuicideModel(args.machinehand_model)
suicide_model.load(args.suicide_model)

test_path    = args.path + '/test'
# TODO set the right batchsiz for testing
test_loader  = suicide_model.build_loader(test_path, 1, False, args.cropped_width, args.cropped_height)


results, labels, test_loss, correct = suicide_model.test(test_loader)

MU.roc_curve(results, labels)


# ./suicide_roccurve.py -p ./suicide_dataset/test/1 -m ./machinehand_model.pth -s ./suicide_model.pth -w 150 -t 30 

    
