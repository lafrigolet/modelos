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



random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)

suicide_model = sm.SuicideModel(args.machinehand_model)

#    def train(self, machinehand_model, learning_rate, train_path, test_path, batch_size_train, batch_size_test, n_epochs, image_width, image_height):
suicide_model.train('suicide_model', args.learning_rate, args.path + '/train',
                    args.path + '/test', args.batch_size_train, args.batch_size_test,
                    args.epochs, args.cropped_width, args.cropped_height)


# ./suicide_train.py -p ./suicide_dataset -m ./machinehand_model.pth -b 64 -s 1000 -e 70 -w 150 -t 30 -l 0.0001

    
