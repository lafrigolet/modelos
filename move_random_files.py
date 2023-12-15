#!/usr/bin/env python3
# coding: utf-8

import os
import random
import shutil
import argparse

def move_random_files(source_dir, dest_dir, percentage_to_move):
    # Get a list of all files in the source directory
    all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    # Calculate the number of files to move based on the percentage
    num_files = int(len(all_files) * percentage_to_move)

    # Randomly select files to move
    files_to_move = random.sample(all_files, num_files)

    # Make destination dir if it does not exist
    os.makedirs(dest_dir, exist_ok = True)
    
    # Move selected files to the destination directory
    for file_name in files_to_move:
        source_path = os.path.join(source_dir, file_name)
        dest_path = os.path.join(dest_dir, file_name)
        shutil.move(source_path, dest_path)
        print(f"Moved: {file_name}")

# Create the parser
parser = argparse.ArgumentParser(description='Command-line argument parser')

# Add arguments
parser.add_argument('-s', '--source'     , type=str  , help='Source path with the files')
parser.add_argument('-d', '--destination', type=str  , help='Destination path')
parser.add_argument('-p', '--percentage' , type=float, help='Percentaje of files to move between 0 and 1')

# Parse the command-line arguments
args = parser.parse_args()
        
if __name__ == "__main__":
    # Call the function to move random files based on the percentage
    move_random_files(args.source, args.destination, args.percentage)


# ./move_random_files.py -s osborne_dataset.tensor/juan_nicolas -d osborne_dataset.tensor/juan_nicolas/test -p 0.2
# ./move_random_files.py -s osborne_dataset.tensor/tomas_osborne -d osborne_dataset.tensor/tomas_osborne/test -p 0.2
