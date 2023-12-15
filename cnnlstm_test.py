#!/usr/bin/env python3
# coding: utf-8

import torch
import argparse
import os

import models.tools as T
import models.cnnlstm as CL

# Create the parser
parser = argparse.ArgumentParser(description='Command-line argument parser')

# Add arguments
parser.add_argument('-p', '--path'           , type=str, help='Path to the directory with testing tensor files')
parser.add_argument('-o', '--pth'            , type=str, help='File name for the pth file for the model')
parser.add_argument('-s', '--batch_size_test', type=int, help='size of the batch for testing')
parser.add_argument('-w', '--width'          , type=int, help='cropped image width')
parser.add_argument('-t', '--height'         , type=int, help='cropped image height')

# Parse the command-line arguments
args = parser.parse_args()
print('args ', args)

# PREPARING THE DATASET
random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)

# Build the Model
model = CL.CNNLSTM(args.height, args.width)
model.load_state_dict(torch.load(args.pth))
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Move the model to the GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Check the device of the model's parameters
param_device = next(model.parameters()).device

if param_device.type == "cuda":
    print("Model is using GPU.")
else:
    print("Model is using CPU.")


# Prepare testing data loader    
tensors0     = [args.path  + '/test/0/' + tensor for tensor in os.listdir(args.path + '/test/0/')]
tensors1     = [args.path  + '/test/1/' + tensor for tensor in os.listdir(args.path + '/test/1/')]
test_data    = [(torch.load(tensor), 0) for tensor in tensors0] + [(torch.load(tensor), 1) for tensor in tensors1]
test_dataset = T.CustomDataset(test_data)
print("test_data len ", len(test_data))
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=args.batch_size_test, 
                                           shuffle=False)

    
results, labels, test_loss, correct = T.test(model, test_loader)

T.roc_curve("ROC", results.cpu(), labels.cpu())

# ./cnnlstm_train.py -p ./machinehand_dataset.tensor -o ./machinehand_model.cnnlstm.pth -s 1000 -w 150 -t 40

# ./cnnlstm_test.py -p ./suicide_dataset.tensor -o ./suicide_model.cnnlstm.pth -s 1000 -w 150 -t 40

# ./cnnlstm_test.py -p ./suicide_dataset.small.tensor -o ./suicide_model.small.cnnlstm.pth -s 1000 -w 150 -t 40

# ./cnnlstm_test.py -p ./osborne_dataset.tensor -o ./osborne_model.cnnlstm.pth -s 1000 -w 150 -t 40
    
