# CNN PLUS TRANSFORMER DEFINITION

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter

from PIL import Image

class CNNTrans(nn.Module):
        
    def __init__(self):
        super(CNNTrans, self).__init__()

        ################################
        # CNN                          #
        ################################

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

        ################################
        # Transformer                  #
        ################################

        ntoken  = 40000  # ntokens = len(vocab)  # size of vocabulary
        d_model = 200    # emsize = 200  # embedding dimension
        nhead   = 2      # number of heads in ``nn.MultiheadAttention``
        d_hid   = 200    # dimension of the feedforward network model in ``nn.TransformerEncoder``
        nlayers = 2      # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
        dropout = 0.2    # dropout probability
        
        # super().__init__()
        # self.model_type = 'Transformer'
        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers           = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        # self.embedding = nn.Embedding(ntoken, d_model)
        # self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
        

        # TODO Ver si self.embedding es necesario. Aparece en el constructor y en init_weights, asi que ten cuidado!!
        
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

        # Apply log softmax activation function to output of the network
        return F.log_softmax(x)


def train(model, train_loader, test_loader, n_epochs, learning_rate):
    # tensorboard inititialization writer
    writer = SummaryWriter('logs')
    
    # optimizer         = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    model.optimizer     = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer         = optim.Rprop(network.parameters(), lr=learning_rate)
    
    # Train the model
    for epoch in range(1, n_epochs + 1):
        model.train()
        datos_pasados = 0
        batch_idx = 0
        loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            datos_pasados += len(data)
            model.optimizer.zero_grad()
            data = data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            output = model(data)
            target = target.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            loss = F.nll_loss(output, target)
            writer.add_scalar('loss', loss.item(), epoch) # tensorboard log
            loss.backward()
            model.optimizer.step()
            
            
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, datos_pasados, len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

        if epoch % 10 == 0:
            results, labels, test_loss, correct = test(model, test_loader)
            print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

            results, labels, test_loss, correct = test(model, train_loader)
            print('Train set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, len(train_loader.dataset),
                100. * correct / len(train_loader.dataset)))

    writer.close()

                
def test(model, loader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loss = 0
    correct   = 0
    results   = torch.Tensor([]).to(device)
    labels    = torch.Tensor([]).to(device)
    with torch.no_grad():
        for data, target in loader:
            data    = data.to(device)
            target  = target.to(device)
            output  = model(data)
            pred    = output.data.max(1, keepdim=True)[1]
            results = torch.cat((results, output))
            labels  = torch.cat((labels, target))
            test_loss += F.nll_loss(output, target, size_average=False).item()
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(loader.dataset)

    return results, labels, test_loss, correct

    
def eval(model, img, image_width, image_height):
    model.eval()

    assert isinstance(img, Image.Image)

    img = TF.to_tensor(img)
    
    img = img.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    output = model(img)
                       
    return torch.exp(output)
