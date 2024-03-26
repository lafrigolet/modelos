#!/usr/bin/env python3
# coding: utf-8

import argparse
import os
import torch
import torch.nn.functional as F

import models.tools as T
import models.cnn as CNN

# Create the parser
parser = argparse.ArgumentParser(description='Command-line argument parser')

# Add arguments
parser.add_argument('-m', '--model', type=str, help='Model to use from models dir')
parser.add_argument('-p', '--path', type=str, help='Path to the directory with testing tensor files')
parser.add_argument('-o', '--pth', type=str, help='File name for the pth file for the model')
parser.add_argument('-c', '--csv', type=str, help='File name for the csv file for the model')
parser.add_argument('-g', '--png', type=str, help='File name for the roc_curve file')
parser.add_argument('-s', '--batch_size_test', type=int, help='size of the batch for testing')
parser.add_argument('-w', '--width', type=int, help='cropped image width')
parser.add_argument('-t', '--height', type=int, help='cropped image height')

# Parse the command-line arguments
args = parser.parse_args()
print('args ', args)

# PREPARING THE DATASET
random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)

if (args.model.lower() == "cnn"):
    import models.cnn as MODEL
    model = MODEL.CNN(args.height, args.width)
elif (args.model.lower() == "cnnlstm"):
    import models.cnnlstm as MODEL
    model = MODEL.CNNLSTM(args.height, args.width)
else:
    import sys
    print("Error: unrecognized model in parameters, choose cnn or cnnlstm")
    exit_code = 1
    sys.exit(exit_code)


# Build the Model
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
tensors0     = [(args.path  + '/0/' + tensor,0) for tensor in os.listdir(args.path + '/0/')]
tensors1     = [(args.path  + '/1/' + tensor,1) for tensor in os.listdir(args.path + '/1/')]
tensors      = tensors0 + tensors1

test_dataset = T.OnDiskDataset(tensors)

#test_data    = [(torch.load(tensor), 0) for tensor in tensors0] + [(torch.load(tensor), 1) for tensor in tensors1]

#test_dataset = T.CustomDataset(test_data)

test_loader  = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=args.batch_size_test, 
                                           shuffle=False)

    
results, labels, test_loss, correct = T.test(model, test_loader)

with open(args.csv, 'w') as csv_file:
    tensors = tensors0 + tensors1
    probabilities = [F.softmax(result)[1] for result in results.cpu()]
    print(len(tensors))
    print(len(probabilities))

    file_results= zip(tensors, probabilities, labels)
    print(f"file, probability, label", file=csv_file)
    for tensor, prob, label in file_results:
        print(f"{tensor}, {prob:.3f}, {int(label)}", file=csv_file)

    fpr, tpr, thresholds = T.roc_curve(f"ROC {args.pth}", probabilities, labels.cpu(), args.png)

    print(len(tensors))
    print(len(probabilities))
    print(len(fpr))
    print(len(tpr))
    print(len(thresholds))



# ./test.py --model cnn -p ./machinehand_dataset.tensor/test -o ./machinehand_model.cnn.pth -s 1000 -w 150 -t 40

# ./test.py --model cnn  -p ./suicide_dataset.tensor/test -o ./suicide_model.cnn.pth -s 1000 -w 150 -t 40

# ./test.py --model cnn  -p ./suicide_dataset.small.tensor/test -o ./suicide_model.small.cnn.pth -s 1000 -w 150 -t 40

# ./test.py --model cnn  -p ./osborne_dataset.tensor/test -o ./osborne_model.cnn.pth -s 1000 -w 150 -t 40
    
