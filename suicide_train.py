#!/usr/bin/env python3
# coding: utf-8

import torch
import suicide_model as sm
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Command-line argument parser')

# Add arguments
parser.add_argument('-p', '--path', type=str, help='Path to the directory with files')
parser.add_argument('-m', '--machinehand_model', type=str, help='pth file for machine hand written model')
parser.add_argument('-b', '--batch_size_train', type=int, help='size of the batch for training')
parser.add_argument('-s', '--batch_size_test', type=int, help='size of the batch for testing')
parser.add_argument('-e', '--epochs', type=int, help='epochs')
parser.add_argument('-l', '--learning_rate', type=float, help='learning rate')
parser.add_argument('-w', '--cropped_width', type=int, help='cropped image width')
parser.add_argument('-t', '--cropped_height', type=int, help='cropped image height')

# Parse the command-line arguments
args = parser.parse_args()

# PREPARING THE DATASET
random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)

# Build and train the model
suicide_model = sm.SuicideModel(args.machinehand_model)

train_path   = args.path + '/train'
test_path    = args.path + '/test'
train_loader = suicide_model.build_loader(train_path, args.batch_size_train, True, args.cropped_width, args.cropped_height)
test_loader  = suicide_model.build_loader(test_path, args.batch_size_test, False, args.cropped_width, args.cropped_height)

suicide_model.train(train_loader, test_loader, args.epochs, args.learning_rate)

suicide_model.save('suicide_model')

suicide_model.roc_curve(test_loader)

# ./suicide_train.py -p ./suicide_dataset -m ./machinehand_model.pth -b 64 -s 1000 -e 70 -w 150 -t 30 -l 0.0001

    
