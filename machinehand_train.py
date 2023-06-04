#!/usr/bin/env python3
# coding: utf-8

import torch
import torchvision
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms
import torch.nn.functional as F
import machinehand_model
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

# PREPARING THE DATASET
n_epochs         = 70
batch_size_train = 64
batch_size_test  = 1000
learning_rate    = 0.0001
momentum         = 0.5
log_interval     = 20
dropout          = 20 # percentaje
image_width      = 150
image_height     = 30

random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)


machinehand_model = machinehand_model.MachineHandModel()
machinehand_model.train(args.machinehand_model, args.learning_rate, args.path + '/train',
                        args.path + '/test', args.batch_size_train, args.batch_size_test,
                        args.epochs, args.cropped_width, args.cropped_height)

# ./machinehand_train.py -m machinehand_model -p ./machinehand_dataset -b 64 -s 1000 -e 70 -w 150 -t 30 -l 0.0001

"""
plt.plot(trainer.train_losses())
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')


def print_failed(loader):
    network.eval()
    test_loss = 0
    correct = 0
    test_train_losses = []
    with torch.no_grad():
        for data, target in loader:
            #print('data', data.shape)
            output = network(data)
            #print('output', output)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            print('Result ', pred.eq(target.data.view_as(pred))[0][0], ' : ', torch.exp(output))
            if (not pred.eq(target.data.view_as(pred))[0][0]):
                plt.tight_layout()
                plt.imshow(data[0][0], cmap='gray', interpolation='none', vmin=0, vmax=1)
                plt.colorbar()
                plt.title("target {}, pred {}".format(target.data.view_as(pred)[0][0], pred[0][0]))
                plt.show()
    test_loss /= len(loader.dataset)
    test_train_losses.append(test_loss)
    print('Train set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))

#print_failed(torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False))
#print_failed(torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=False))


# In[ ]:


# MODEL PERFORMANCE 
fig = plt.figure()
plt.plot(trainer.train_counter(), trainer.train_losses(), color='blue')
plt.scatter(tester.test_counter(), tester.test_losses(), color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
fig


# In[ ]:


# DATASET INSPECTION

def dataset_instpection():
    examples = enumerate(train_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    print(example_data.shape)
    print(example_targets.shape)

# So one test data batch is a  tensor of shape: . This means we have 1000 examples of 28x28 pixels
# in grayscale (i.e. no rgb channels, hence the one). We can plot some of them using matplotlib.


for i in range(example_targets.shape[0]):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])

"""
