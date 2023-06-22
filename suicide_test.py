#!/usr/bin/env python3
# coding: utf-8

import torch
import suicide_model as sm
import argparse
import custom_dataset

# Create the parser
parser = argparse.ArgumentParser(description='Command-line argument parser')

# Add arguments
parser.add_argument('-p', '--test_path', type=str, help='Path to the directory with files')
parser.add_argument('-m', '--machinehand_model', type=str, help='pth file for machine hand written model')
parser.add_argument('-s', '--suicide_model', type=str, help='pth file for suicide model')
parser.add_argument('-b', '--batch_size_test', type=int, help='size of the batch for testing')
parser.add_argument('-w', '--cropped_width', type=int, help='cropped image width')
parser.add_argument('-t', '--cropped_height', type=int, help='cropped image height')


# Parse the command-line arguments
args = parser.parse_args()

random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)


suicide_model = sm.SuicideModel(args.machinehand_model)
suicide_model.load(args.suicide_model)

test_dataset = custom_dataset.CustomDataset()
test_dataset.append_images(suicide_model.cook_images(args.test_path + '/0', args.cropped_width, args.cropped_height), 0)
test_dataset.append_images(suicide_model.cook_images(args.test_path + '/1', args.cropped_width, args.cropped_height), 1)

        
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=args.batch_size_test, 
                                          shuffle=False)



suicide_model.roc_curve(test_loader)

# ./suicide_test.py -p ./suicide_dataset -m ./machinehand_model.pth -s suicide_model.pth -b 64

    
