# CNN DEFINITION

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
        
    def __init__(self):
        super(Net, self).__init__()
        layers_in_last_conv = 20
        rows = 30
        cols = 150
        kernel_size = 5
        
        # First convolutional layer with 1 input channel, 10 output channels, and kernel size of 5
        self.conv1 = nn.Conv2d(1, 10, kernel_size=kernel_size)
        rows = rows - (kernel_size - 1)
        cols = cols - (kernel_size - 1)
        rows = rows // 2
        cols = cols // 2
        # Second convolutional layer with 10 input channels, 20 output channels, and kernel size of 5
        self.conv2 = nn.Conv2d(10, layers_in_last_conv, kernel_size=kernel_size)
        rows = rows -(kernel_size - 1)
        cols = cols - (kernel_size - 1)
        rows = rows // 2
        cols = cols // 2
        # Dropout layer to prevent overfitting
        self.conv2_drop = nn.Dropout2d() 
        
        self.__neurons_in_first_fully_connected = layers_in_last_conv * rows * cols
        # First fully connected layer with 6120 input features and 50 output features
        self.fc1 = nn.Linear(self.__neurons_in_first_fully_connected, 50)
        # Second fully connected layer with 50 input features and 1 output features
        self.fc2 = nn.Linear(50, 2) 

    def forward(self, input_data):
        # Apply first convolutional layer, ReLU activation function, and max pooling with a 2x2 kernel size
        x = self.conv1(input_data)
        # print('conv2d 1', x.shape)
        x = F.relu(F.max_pool2d(x, 2)) 
        # print('relu 1', x.shape)
        # print('conv2d 1', x.shape) # conv2d 1 torch.Size([1, 10, 23, 73])
        # Apply second convolutional layer, dropout, ReLU activation function, and max pooling with a 2x2 kernel size
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # print('conv2d 2', x.shape) # conv2d 2 torch.Size([1, 20, 9, 34])
        # Reshape output from convolutional layers to fit fully connected layers
        # x = x.view(-1, 70720)
        x = x.view(-1, self.__neurons_in_first_fully_connected)
        # Apply ReLU activation function to first fully connected layer
        x = F.relu(self.fc1(x))
        # Apply dropout layer to prevent overfitting
        x = F.dropout(x, training=self.training)
        # Apply second fully connected layer
        x = self.fc2(x)
        # Apply log softmax activation function to output of the network
        return F.log_softmax(x)

    
