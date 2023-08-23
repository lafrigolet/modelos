#!/usr/bin/env python3
# coding: utf-8

import suicide_model as SM
import os
import argparse
import matplotlib.pyplot as plt

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

suicide_model = SM.SuicideModel(args.machinehand_model)
suicide_model.load(args.suicide_model)

files = os.listdir(args.path + '/0')

files = [args.path + '/0/' + f for f in files]

for file in files:
    print(file)
    score = suicide_model.suicide_score(file, args.cropped_height, args.cropped_width)
    plt.plot(score, color = 'blue')

files = os.listdir(args.path + '/1')

files = [args.path + '/1/' + f for f in files]

for file in files:
    print(file)
    score = suicide_model.suicide_score(file, args.cropped_height, args.cropped_width)
    plt.plot(score, color = 'red')

plt.xlabel('score index')
plt.ylabel('frequency')
plt.title('Suicide Score')
plt.legend()
plt.show()

# ./suicide_score.py -p ./suicide_dataset/test/1 -m ./machinehand_model.pth -s ./suicide_model.pth -w 150 -t 30 
    
