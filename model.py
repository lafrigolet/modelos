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
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_curve
import math

class Model():
    def __init__(self):
        self.network       = net.Net()

        # Move the model to the GPU if available
        if torch.cuda.is_available():
            self.network.to(torch.device("cuda"))

        # Check the device of the model's parameters
        param_device = next(self.network.parameters()).device

        if param_device.type == "cuda":
            print("Model is using GPU.")
        else:
            print("Model is using CPU.")

    def load(self, pth):
        self.network.load_state_dict(torch.load(pth))

    def cook_images(self, path, label, image_width, image_height): # template method
        raise NotImplementedError()

    def build_loader(self, path, batchsize, shuffle, image_width, image_height):
        # Usar la base de datos construida
        dataset = custom_dataset.CustomDataset()
        dataset.append_images(self.cook_images(path + '/0', image_width, image_height), 0)
        dataset.append_images(self.cook_images(path + '/1', image_width, image_height), 1)

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batchsize, 
                                             shuffle=shuffle)

        return loader


    def save(self, output_pth_file):
        torch.save(self.network.state_dict(), output_pth_file + '.pth')
        torch.save(self.optimizer.state_dict(), output_pth_file + '_optimizer.pth')


    def train_helper(self, loader, epoch):
        datos_pasados = 0
        self.network.train()
        for batch_idx, (data, target) in enumerate(loader):
            datos_pasados += len(data)
            self.optimizer.zero_grad()
            data = data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            output = self.network(data)
            target = target.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

        return datos_pasados, batch_idx, loss
    
        
    def train(self, train_loader, test_loader, n_epochs, learning_rate):
        # optimizer         = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
        self.optimizer     = optim.Adam(self.network.parameters(), lr=learning_rate)
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



    def roc_curve(self, loader):
        self.network.eval()
        
        results, labels, test_loss, correct = self.test(loader)

        x = [math.exp(result[1]) for result in results]
        y = labels
        
        fpr, tpr, thresholds = roc_curve(y,x)
        plt.plot(fpr,tpr,color="blue")
        plt.grid()
        plt.xlabel("FPR (especifidad)", fontsize=12, labelpad=10)
        plt.ylabel("TPR (sensibilidad, Recall)", fontsize=12, labelpad=10)
        plt.title("ROC de suicidios", fontsize=14)
        
        nlabels = int(len(thresholds) / 5)
 
        for cont in range(0,len(thresholds)):
            if not cont % nlabels:
                plt.text(fpr[cont], tpr[cont], "  {:.2f}".format(thresholds[cont]),color="blue")
                plt.plot(fpr[cont], tpr[cont],"o",color="blue")

        plt.show()

        return x, y
        

    def test(self, loader):
        self.network.eval()
        test_loss = 0
        correct   = 0
        results   = []
        labels    = []
        with torch.no_grad():
            for data, target in loader:
                #print('data', data.shape)
                data = data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                target = target.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                output = self.network(data)
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
        self.network.eval()

        assert isinstance(img, Image.Image)

        img = TF.to_tensor(img)

        img = img.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                       
        return torch.exp(self.network(img))
                         
    def eval(self, path, image_width, image_height):
        dataset = custom_dataset.CustomDataset()
        dataset.append_images(self.cook_images(path, image_width, image_height), 0) # label doesn't matter
        self.network.eval()

        with torch.no_grad():
            data = data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            result = [torch.exp(self.network(data)) for data, target in dataset]
            
        return result
