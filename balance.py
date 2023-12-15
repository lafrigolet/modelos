#!/usr/bin/env python3
# coding: utf-8

# Balance the number of files in two given directories by randomly removing
# files from the directory with more files

import os
import random
import shutil
import argparse

def balance_directories(dir1, dir2):
    # Get the list of files in each directory
    files_dir1 = [f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))]
    files_dir2 = [f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))]

    # Calculate the difference in the number of files
    diff = len(files_dir1) - len(files_dir2)

    if diff > 0:
        # Remove random files from dir1
        files_to_remove = random.sample(files_dir1, diff)
        for file_name in files_to_remove:
            file_path = os.path.join(dir1, file_name)
            os.remove(file_path)
            print(f"Removed: {file_path}")
    elif diff < 0:
        # Remove random files from dir2
        files_to_remove = random.sample(files_dir2, abs(diff))
        for file_name in files_to_remove:
            file_path = os.path.join(dir2, file_name)
            os.remove(file_path)
            print(f"Removed: {file_path}")
    else:
        print("Directories already have the same number of files.")

# Create the parser
parser = argparse.ArgumentParser(description='Command-line argument parser')

# Add arguments
parser.add_argument('-1', '--dir1', type=str  , help='Directory 1')
parser.add_argument('-2', '--dir2', type=str  , help='Directory 2')

# Parse the command-line arguments
args = parser.parse_args()
        
if __name__ == "__main__":
    # Specify the directories to balance
    directory1 = "/path/to/first/directory"
    directory2 = "/path/to/second/directory"

    # Call the function to balance the directories
    balance_directories(args.dir1, args.dir2)


# ./balance.py -1 osborne_dataset.tensor/train/0 -2 osborne_dataset.tensor/train/1
# ./balance.py -1 osborne_dataset.tensor/test/0 -2 osborne_dataset.tensor/test/1
