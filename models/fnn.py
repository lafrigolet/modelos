from torchtext.datasets import WikiText2
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import dataset
from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optim
import copy
import models.utils as U
import torch
import torch.nn as nn

class FNN(nn.Module):
    # layers_sizes = [input_size, hidden_layer1, hidden_layer2, ..., hidden_layern, num_classes]
    def __init__(self, layer_sizes):
        super(FNN, self).__init__()

        print(f"layers: {layer_sizes}")
        
        layers = []

        input_size = layer_sizes[0]
        hidden_sizes = layer_sizes[1:-1]
        output_size = layer_sizes[-1]

        # Input layer

        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.layers = nn.Sequential(*layers)

        # Move the model to the GPU if available
        if torch.cuda.is_available():
            self.to(torch.device("cuda"))

        # Check the device of the model's parameters
        param_device = next(self.parameters()).device

        if param_device.type == "cuda":
            print("Model is using GPU.")
        else:
            print("Model is using CPU.")



    def forward(self, x):
        return self.layers(x)

            
def load(model, pth):
    model.load_state_dict(torch.load(pth))


def save(model, output_pth_file):
    torch.save(model.state_dict(), output_pth_file + '.pth')
    #torch.save(model.optimizer.state_dict(), output_pth_file + '_optimizer.pth')

        
def train(model, train_loader, test_loader, n_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.Rprop(network.parameters(), lr=learning_rate)
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            _, expected = torch.max(labels, 1) 
            # print(inputs.to(torch.int), expected)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch}/{n_epochs}], Loss: {running_loss / len(train_loader)}")

        if epoch % 10 == 0:
            """
            results, labels, test_loss, correct = model.test(test_loader)
            print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
            """
            correct = test(model, train_loader)
            accuracy = correct / len(train_loader.dataset) * 100
            print(f"Train set Accuracy: {accuracy:.2f}%")

            correct = test(model, test_loader)
            accuracy = correct / len(test_loader.dataset) * 100
            print(f"Test set Accuracy: {accuracy:.2f}%")


def test(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            _, targets = torch.max(labels, 1)
            correct += (predicted == targets).sum().item()
    
    return correct



