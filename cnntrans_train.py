#!/usr/bin/env python3
# coding: utf-8

import torch
import argparse
import os

import models.cnntrans as CT
import models.utils as MU
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# Create the parser
parser = argparse.ArgumentParser(description='Command-line argument parser')

# Add arguments
parser.add_argument('-p', '--path', type=str, help='Path to the directory with tensor files')
parser.add_argument('-b', '--batch_size_train', type=int, help='size of the batch for training')
parser.add_argument('-s', '--batch_size_test', type=int, help='size of the batch for testing')
parser.add_argument('-e', '--epochs', type=int, help='epochs')
parser.add_argument('-l', '--learning_rate', type=float, help='learning rate')
parser.add_argument('-w', '--width', type=int, help='cropped image width')
parser.add_argument('-t', '--height', type=int, help='cropped image height')

# Parse the command-line arguments
args = parser.parse_args()

# PREPARING THE DATASET
random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)

# Build the Model
model = CT.CNNTrans()

# Move the model to the GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Check the device of the model's parameters
param_device = next(model.parameters()).device

if param_device.type == "cuda":
    print("Model is using GPU.")
else:
    print("Model is using CPU.")


# Prepare training data loader    
tensors0    = [args.path  + '/train/0/' + tensor for tensor in os.listdir(args.path + '/train/0/')]
tensors1    = [args.path  + '/train/1/' + tensor for tensor in os.listdir(args.path + '/train/1/')]
train_data  = [ (torch.load(tensor), 0) for tensor in tensors0] + [ (torch.load(tensor), 1) for tensor in tensors1]
train_dataset     = CustomDataset(train_data)
print("train_data len ", len(train_data))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size_train, 
                                           shuffle=True)

# Prepare testing data loader    
tensors0    = [args.path  + '/test/0/' + tensor for tensor in os.listdir(args.path + '/test/0/')]
tensors1    = [args.path  + '/test/1/' + tensor for tensor in os.listdir(args.path + '/test/1/')]
test_data  = [ (torch.load(tensor), 0) for tensor in tensors0] + [ (torch.load(tensor), 1) for tensor in tensors1]
test_dataset     = CustomDataset(test_data)
print("test_data len ", len(test_data))
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size_test, 
                                          shuffle=False)

CT.train(model, train_loader, test_loader, args.epochs, args.learning_rate)

output_pth_file = 'cnntrans_model'
torch.save(model.state_dict(), output_pth_file + '.pth')
torch.save(model.optimizer.state_dict(), output_pth_file + '_optimizer.pth')

results, labels, test_loss, correct = CT.test(model, test_loader)

MU.roc_curve(results.cpu(), labels.cpu())

# ./cnntrans_train.py -p ./suicide_dataset -b 64 -s 1000 -e 70 -w 150 -t 30 -l 0.0001

    
