# CNN DEFINITION

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

class CNN(nn.Module):
        
    def __init__(self):
        super(CNN, self).__init__()
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

        # Move the model to the GPU if available
        if torch.cuda.is_available():
            self.to(torch.device("cuda"))

        # Check the device of the model's parameters
        param_device = next(self.parameters()).device

        if param_device.type == "cuda":
            print("Model is using GPU.")
        else:
            print("Model is using CPU.")


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

    def load(self, pth):
        self.load_state_dict(torch.load(pth))

    def save(self, output_pth_file):
        torch.save(self.state_dict(), output_pth_file + '.pth')
        torch.save(self.optimizer.state_dict(), output_pth_file + '_optimizer.pth')

    def train_helper(self, loader, epoch):
        datos_pasados = 0
        super().train()
        for batch_idx, (data, target) in enumerate(loader):
            datos_pasados += len(data)
            self.optimizer.zero_grad()
            data = data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            output = self(data)
            target = target.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            
        return datos_pasados, batch_idx, loss

    def train_model(self, train_loader, test_loader, n_epochs, learning_rate):
        # optimizer         = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
        self.optimizer     = optim.Adam(self.parameters(), lr=learning_rate)
        # optimizer         = optim.Rprop(network.parameters(), lr=learning_rate)

        # Train the model
        for epoch in range(1, n_epochs + 1):
            datos_pasados, batch_idx, loss = self.train_helper(train_loader, epoch)
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, datos_pasados, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

            if epoch % 10 == 0:
                results, labels, test_loss, correct = self.test(test_loader)
                print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                    test_loss, correct, len(test_loader.dataset),
                    100. * correct / len(test_loader.dataset)))

                results, labels, test_loss, correct = self.test(train_loader)
                print('Train set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                    test_loss, correct, len(train_loader.dataset),
                    100. * correct / len(train_loader.dataset)))

                
    def test(self, loader):
        self.eval()
        test_loss = 0
        correct   = 0
        results   = []
        labels    = []
        with torch.no_grad():
            for data, target in loader:
                #print('data', data.shape)
                data = data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                target = target.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                output = self(data)
                pred = output.data.max(1, keepdim=True)[1]
                results += output.tolist()
                labels  += target
                #for t in torch.exp(output):
                #    print('[{:.4f}, {:.4f}]'.format(t[0], t[1]))
                test_loss += F.nll_loss(output, target, size_average=False).item()
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(loader.dataset)

        return results, labels, test_loss, correct

    
    def eval_image(self, img, image_width, image_height):
        self.eval()

        assert isinstance(img, Image.Image)

        img = TF.to_tensor(img)

        img = img.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        output = self(img)
                       
        return torch.exp(output)
