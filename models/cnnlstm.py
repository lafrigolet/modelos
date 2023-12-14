# CNN PLUS TRANSFORMER DEFINITION

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
#from torch.utils.tensorboard import SummaryWriter

from PIL import Image

class CNNLSTM(nn.Module):
        
    def __init__(self, rows, cols):
        super(CNNLSTM, self).__init__()

        ################################
        # CNN                          #
        ################################

        cnn_conv1_output_channels = 10
        cnn_conv1_kernel_size     = 5
        cnn_conv2_output_channels = 20
        cnn_conv2_kernel_size     = 5
        
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
        self.fc2 = nn.Linear(50, 10)

        ################################
        # LSTM                         #
        ################################

        # Dummy data for illustration
        lstm_input_size  = 10
        lstm_hidden_size = 20
        output_size = 2

        """
        - lstm_input_size: Represents the number of features in each time step of the input
        sequence. It is the dimensionality of the input at each time step.

        - lstm_hidden_size (or number of neurons): Represents the number of features in the
        hidden state of the LSTM layer. Each LSTM cell in the layer has this number of neurons.

        - batch_first: By default, RNN layers in PyTorch expect the input data to have the shape
        (sequence_length, batch_size, lstm_input_size). However, when batch_first=True
        is set, the expected shape becomes (batch_size, sequence_length, lstm_input_size).

        """
        
        # LSTM layer
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, batch_first=True)
        
        # Fully connected layer to map the hidden state to the output
        self.fc3 = nn.Linear(lstm_hidden_size, output_size)
        

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
        # print('x.shape ', x.shape)
        # Apply LSTM layer
        x, (final_hidden_state, final_cell_state) = self.lstm(x)
        # Apply third fully connected layer
        # Use only the output from the last time step for the fully connected layer
        x = self.fc3(x)


        # Apply log softmax activation function to output of the network
        return F.log_softmax(x)



    
