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
parser.add_argument('-p', '--path', type=str, help='Path to the directory with tensor files')
parser.add_argument('-o', '--pth_file', type=str, help='File name for the pth file for the model')
parser.add_argument('-b', '--batch_size_train', type=int, help='size of the batch for training')
parser.add_argument('-s', '--batch_size_test', type=int, help='size of the batch for testing')
parser.add_argument('-e', '--epochs', type=int, help='epochs')
parser.add_argument('-l', '--learning_rate', type=float, help='learning rate')
parser.add_argument('-w', '--width', type=int, help='cropped image width')
parser.add_argument('-t', '--height', type=int, help='cropped image height')
parser.add_argument('-n', '--nbatches', type=int, help='Number of batches for input files to group')

# Parse the command-line arguments
args = parser.parse_args()
print('args ', args)

# PREPARING THE DATASET
random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)

# Build the Model
model = CL.CNNLSTM(args.height, args.width)

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

    
# Prepare training data loader    
files0      = [args.path  + '/train/0/' + file for file in os.listdir(args.path + '/train/0/')]
files1      = [args.path  + '/train/1/' + file for file in os.listdir(args.path + '/train/1/')]

batch0 = T.batchify(files0, args.nbatches)
batch1 = T.batchify(files1, args.nbatches)

print('len(batch0)', len(batch0))
print('len(batch1)', len(batch1))


for i in range(0, args.nbatches):
    print('Batch ', i, '======================================================================')
    train_data = [(torch.load(file), 0) for file in batch0[i]] + [(torch.load(file), 1) for file in batch1[i]]
    train_dataset = T.CustomDataset(train_data)
    print("train_data len ", len(train_data))
    train_loader  = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=args.batch_size_train, 
                                                shuffle=True)

    T.train(model, train_loader, test_loader, args.epochs, args.learning_rate)


torch.save(model.state_dict(), args.pth_file)
#torch.save(model.optimizer.state_dict(), args.pth_file + '_optimizer.pth')

results, labels, test_loss, correct = T.test(model, test_loader)

T.roc_curve("ROC", results.cpu(), labels.cpu())

# ./cnnlstm_train.py -p ./machinehand_dataset.tensor -o ./machinehand_model.pth -b 64 -s 1000 -e 70 -w 150 -t 40 -l 0.0001 -n 1

# ./cnnlstm_train.py -p ./suicide_dataset.tensor -o ./suicide_model.cnnlstm.5.pth -b 64 -s 1000 -e 70 -w 150 -t 40 -l 0.0001 -n 5
# ./cnnlstm_train.py -p ./suicide_dataset.tensor -o ./suicide_model.cnnlstm.10.pth -b 64 -s 1000 -e 70 -w 150 -t 40 -l 0.0001 -n 10
# ./cnnlstm_train.py -p ./suicide_dataset.tensor -o ./suicide_model.cnnlstm.50.pth -b 64 -s 1000 -e 70 -w 150 -t 40 -l 0.0001 -n 50
# ./cnnlstm_train.py -p ./suicide_dataset.tensor -o ./suicide_model.cnnlstm.100.pth -b 64 -s 1000 -e 70 -w 150 -t 40 -l 0.0001 -n 100

# ./cnnlstm_train.py -p ./suicide_dataset.small.tensor -o ./suicide_model.small.cnnlstm.pth -b 64 -s 1000 -e 70 -w 150 -t 40 -l 0.0001 -n 1

# ./cnnlstm_train.py -p ./osborne_dataset.tensor -o ./osborne_model.cnnlstm.pth -b 64 -s 1000 -e 150 -w 150 -t 40 -l 0.0001 -n 1
