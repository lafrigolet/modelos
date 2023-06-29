#!/usr/bin/env python3
# coding: utf-8

import torch
import suicide_model as sm
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Create the parser
parser = argparse.ArgumentParser(description='Command-line argument parser')

# Add arguments
parser.add_argument('-p', '--test_path', type=str, help='Path to the directory with files')
parser.add_argument('-m', '--machinehand_model', type=str, help='pth file for machine hand written model')
parser.add_argument('-s', '--suicide_model', type=str, help='pth file for suicide model')
parser.add_argument('-b', '--batch_size', type=int, help='size of the batch for testing')
parser.add_argument('-w', '--cropped_width', type=int, help='cropped image width')
parser.add_argument('-t', '--cropped_height', type=int, help='cropped image height')


# Parse the command-line arguments
args = parser.parse_args()

random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)

suicide_model = sm.SuicideModel(args.machinehand_model)
suicide_model.load(args.suicide_model)

files = os.listdir(args.test_path + '/0')

x = []
y = []

for file in files:
    print(args.test_path, '/0', file)
    mode, label = suicide_model.mode(args.test_path + '/0', file, 0, args.cropped_width, args.cropped_height, args.batch_size);
    x.append(mode)
    y.append(label)

files = os.listdir(args.test_path + '/1')


for file in files:
    print(args.test_path, '/1', file)
    mode, label = suicide_model.mode(args.test_path + '/1', file, 1, args.cropped_width, args.cropped_height, args.batch_size);
    x.append(mode)
    y.append(label)

print(x,y)
    
fpr, tpr, thresholds = roc_curve(y,x)
plt.plot(fpr,tpr,color="blue")
plt.grid()
plt.xlabel("FPR (especifidad)", fontsize=12, labelpad=10)
plt.ylabel("TPR (sensibilidad, Recall)", fontsize=12, labelpad=10)
plt.title("ROC de suicidios", fontsize=14)

nlabels = int(len(thresholds) / 5)
nlabels = 1 if nlabels == 0 else nlabels

for cont in range(0,len(thresholds)):
    if not cont % nlabels:
        plt.text(fpr[cont], tpr[cont], "  {:.2f}".format(thresholds[cont]),color="blue")
        plt.plot(fpr[cont], tpr[cont],"o",color="blue")
        
plt.show()


# ./suicide_test.py -p ./suicide_dataset -m ./machinehand_model.pth -s suicide_model.pth -b 64 -w 150 -t 30

    
