#!/usr/bin/env python3
# coding: utf-8

# In[1]:


import torch
import torchvision
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms



# In[50]:


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


import model

# NETWORK AND OPTIMIZER INITIALIZATION

import torch.nn.functional as F
import torch.optim as optim

network = model.Net()
# optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
optimizer = optim.Adam(network.parameters(), lr=learning_rate)
# optimizer = optim.Rprop(network.parameters(), lr=learning_rate)

import loaders
model = loaders.Model(network, optimizer)
model.train('./dataset/train', batch_size_train, './dataset/test', batch_size_test, n_epochs, image_width, image_height)

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

