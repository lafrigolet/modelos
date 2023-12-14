from PIL import Image
import math
import sklearn.metrics as SKL
import matplotlib.pyplot as PLT

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import transforms


# Given a list of files it split in nbatches batches
def batchify(files, nbatches):
    batches = [[] for _ in range(0, nbatches)]
    
    for i, file in enumerate(files):
        batches[i % nbatches].append(file)

    print('len(batches) ', len(batches))
    return batches


class CustomDataset(Dataset):
    def __init__(self, data):
        # self.data = [(tensor.to(device), target) for tensor, target in data]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


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


def train(model, train_loader, test_loader, n_epochs, learning_rate):
    # tensorboard inititialization writer
#    writer = SummaryWriter('logs')
    
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
#            writer.add_scalar('loss', loss.item(), epoch) # tensorboard log
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

#    writer.close()


def eval(model, img):
    model.eval()

    assert isinstance(img, Image.Image)

    img = TF.to_tensor(img)
    
    img = img.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    output = model(img)
                       
    return torch.exp(output)


def roc_curve(title, results, labels):
    x = [math.exp(result[1]) for result in results]
    y = labels

    fpr, tpr, thresholds = SKL.roc_curve(y,x)
    PLT.plot(fpr,tpr,color="blue")
    PLT.grid()
    PLT.xlabel("FPR (especifidad)", fontsize=12, labelpad=10)
    PLT.ylabel("TPR (sensibilidad, Recall)", fontsize=12, labelpad=10)
    PLT.title(title, fontsize=14)
    
    nlabels = int(len(thresholds) / 5)

    """
    for cont in range(0,len(thresholds)):
        if not cont % nlabels:
            PLT.text(fpr[cont], tpr[cont], "  {:.2f}".format(thresholds[cont]),color="blue")
            PLT.plot(fpr[cont], tpr[cont],"o",color="blue")
    """
    
    PLT.show()

    return x, y

