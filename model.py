import torch
import os
from PIL import Image
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
import custom_dataset
from helpers import normalize_images as NI
import net


class Model():
    def __init__(self):
        self.network       = net.Net()
        self.train_losses  = []
        self.train_counter = []
        self.test_losses   = []
        self.test_counter  = []

    def load(self, pth):
        self.network.load_state_dict(torch.load(pth))

    def train_counter(self):
        return self.train_counter

    def train_losses(self):
        return self.train_losses
        
    def test_losses(self):
        return self.test_losses

    def test_counter(self):
        return self.test_counter

    def cook_images(self, path, label, image_width, image_height): # template method
        raise NotImplementedError()
    
    def train(self, output_pth_file, learning_rate, train_path, test_path, batch_size_train, batch_size_test, n_epochs, image_width, image_height):
        # optimizer         = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
        self.optimizer     = optim.Adam(self.network.parameters(), lr=learning_rate)
        # optimizer         = optim.Rprop(network.parameters(), lr=learning_rate)

        def train_helper(loader, epoch):
            datos_pasados = 0
            self.network.train()
            for batch_idx, (data, target) in enumerate(loader):
                datos_pasados += len(data)
                self.optimizer.zero_grad()
                output = self.network(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                self.optimizer.step()
            #if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, datos_pasados, len(loader.dataset),
                    100. * batch_idx / len(loader), loss.item()))
            self.train_losses.append(loss.item())
            self.train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(loader.dataset)))
        
        # Usar la base de datos construida
        # Usar la base de datos construida
        train_dataset = custom_dataset.CustomDataset()
        train_dataset.append_images(self.cook_images(train_path + '/0', 0, image_width, image_height), 0)
        train_dataset.append_images(self.cook_images(train_path + '/1', 1, image_width, image_height), 1)

        test_dataset = custom_dataset.CustomDataset()
        test_dataset.append_images(self.cook_images(test_path + '/0', 0, image_width, image_height), 0)
        test_dataset.append_images(self.cook_images(test_path + '/1', 1, image_width, image_height), 1)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size_train, 
                                                   shuffle=True)
        
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                  batch_size=batch_size_test, 
                                                  shuffle=False)

        # Train the model
        
        for epoch in range(1, n_epochs + 1):
            train_helper(train_loader, epoch)

            if epoch % 10 == 0:
                self.test(test_loader, 'Test set', n_epochs)
                self.test(train_loader, 'Train set', n_epochs)

        torch.save(self.network.state_dict(), output_pth_file + '.pth')
        torch.save(self.optimizer.state_dict(), output_pth_file + '_optimizer.pth')

    def test(self, loader, msg, n_epochs):
        self.test_counter = [i*len(loader.dataset) for i in range(n_epochs + 1)]
        self.network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in loader:
                #print('data', data.shape)
                output = self.network(data)
                #for t in torch.exp(output):
                #    print('[{:.4f}, {:.4f}]'.format(t[0], t[1]))
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(loader.dataset)
        self.test_losses.append(test_loss)
        print('{}: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            msg, test_loss, correct, len(loader.dataset),
            100. * correct / len(loader.dataset)))

    def eval(self, pth, path, image_width, image_height):
        dataset = custom_dataset.CustomDataset()
        dataset.append_images(self.cook_images(path, 0, image_width, image_height), 0) # label doesn't matter
        
        self.network.load_state_dict(torch.load(pth))
        
        self.network.eval()

        with torch.no_grad():
            result = [torch.exp(self.network(data)) for data, target in dataset]
            
        return result
