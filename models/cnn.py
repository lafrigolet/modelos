# CNN DEFINITION

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
#from torch.utils.tensorboard import SummaryWriter

from PIL import Image

class CNN(nn.Module):
        
    def __init__(self, rows, cols):
        super(CNN, self).__init__()

        ################################
        # CNN                          #
        ################################

        cnn_conv1_output_channels = 10
        cnn_conv1_kernel_size     = 5
        cnn_conv2_output_channels = 20
        cnn_conv2_kernel_size     = 5
        #rows = 30
        #cols = 150
        
        # First convolutional layer with 1 input channel, 10 output channels, and kernel size of 5
        self.conv1 = nn.Conv2d(1, cnn_conv1_output_channels, kernel_size=cnn_conv1_kernel_size)
        rows = rows - (cnn_conv1_kernel_size - 1)
        cols = cols - (cnn_conv1_kernel_size - 1)
        rows = rows // 2
        cols = cols // 2
        # Second convolutional layer with 10 input channels, 20 output channels, and kernel size of 5
        self.conv2 = nn.Conv2d(10, cnn_conv2_output_channels, kernel_size=cnn_conv2_kernel_size)
        rows = rows - (cnn_conv2_kernel_size - 1)
        cols = cols - (cnn_conv2_kernel_size - 1)
        rows = rows // 2
        cols = cols // 2
        # Dropout layer to prevent overfitting
        self.conv2_drop = nn.Dropout2d() 
        
        self.__neurons_in_first_fully_connected = cnn_conv2_output_channels * rows * cols
        # First fully connected layer with 6120 input features and 50 output features
        self.fc1 = nn.Linear(self.__neurons_in_first_fully_connected, 50)
        # Second fully connected layer with 50 input features and 1 output features
        self.fc2 = nn.Linear(50, 2)
        

    def forward(self, input_data):
        # Apply first convolutional layer, ReLU activation function, and max pooling with a 2x2 kernel size
        x = self.conv1(input_data)
        # print('x1.shape ', x.shape)
        x = F.relu(F.max_pool2d(x, 2)) 
        # print('x2.shape ', x.shape)
        # Apply second convolutional layer, dropout, ReLU activation function, and max pooling with a 2x2 kernel size
        x = self.conv2(x)
        # print('x3.shape ', x.shape)
        x = self.conv2_drop(x)
        # print('x3.1.shape ', x.shape)
        x = F.max_pool2d(x, 2)
        # print('x3.2.shape ', x.shape)
        x = F.relu(x)
        # print('x4.shape ', x.shape)
        # Reshape output from convolutional layers to fit fully connected layers
        x = x.view(-1, self.__neurons_in_first_fully_connected)
        # print('x5.shape ', x.shape)
        # Apply first fully connected layer
        x = self.fc1(x)
        # print('x6.shape ', x.shape)
        # Apply ReLU activation function to first fully connected layer
        x = F.relu(x)
        # print('x7.shape ', x.shape)
        # Apply dropout layer to prevent overfitting
        x = F.dropout(x, training=self.training)
        # print('x8.shape ', x.shape)
        # Apply second fully connected layer
        x = self.fc2(x)
        # print('x9.shape ', x.shape)

        # Apply log softmax activation function to output of the network
        return F.log_softmax(x)


    
