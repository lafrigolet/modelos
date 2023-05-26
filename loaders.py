import torch
import os
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys

def normalize_png_file(file, target_size):
    # Load the image
    img = Image.open(file)

    # Get the width and height of the image
    width, height = img.size
    tw, th = target_size

    # Calculate the new width and height while keeping aspect ratio
    if tw/width > th/height:
        new_width  = int(th * width / height)
        new_height = th
    else:
        new_width  = tw
        new_height = int(tw * height / width)

    #plt.imshow(img, cmap="gray"); plt.show();
        
    # Convert the image to grayscale
    img_gray = TF.to_grayscale(img, num_output_channels=1)
    #plt.imshow(img_gray, cmap="gray"); plt.show();
    
    #img_crop = TF.crop(img_gray, 0, 0, 50, 150)
    #plt.imshow(img_crop, cmap="gray"); plt.show();

    # Resize the image while keeping aspect ratio
    img_resized = TF.resize(img_gray, (new_height, new_width))
    #plt.imshow(img_resized, cmap="gray"); plt.show();
    
    # Pad the image if necessary to match the target size
    img_padded = TF.pad(img_resized, (0, 0, target_size[0] - new_width, target_size[1] - new_height))
    #img_padded = img_padded / 255
    # plt.imshow(img_padded, cmap="gray"); plt.show();

    # Convert the image to a PyTorch tensor 
    # (height, width, channel) -> (channel, height, width)
    img_tensor = TF.to_tensor(img_padded)
    """
    # Plot the tensor as an image 
    plt.imshow(img_tensor.permute(1, 2, 0), cmap="gray")
    plt.imshow(img_tensor[0], cmap="gray")
    plt.show()
    """
    """
    # Set a threshold value
    threshold = 0.8

    # Convert the grayscale image to a black and white image
    img_bw = torch.where(img_tensor < threshold, torch.zeros_like(img_tensor), torch.ones_like(img_tensor))

    # Plot the tensor as an image
    plt.imshow(img_bw[0], cmap="gray")
    plt.show()
    """
    
    return img_tensor

#normalize_png_file('OsborneSubdataset/1883-L119.M29_12_IMG_0021_fila_0_posicion_0_4c6f73.png', (image_width, image_height))


# Construcción de la base de datos.
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):

        # 1. Inicialización de rutas y nombres de ficheros
        self.lista_imagenes = []
        self.list_img       = []
        self.list_label     = []

        # print(self.lista_imagenes)
        # print(len(self.lista_imagenes))
        # print(len(self.list_img))
        # print(len(self.list_label))
        
        pass

    def append_path(self, path, label, image_width, image_height):
        lista_imagenes = os.listdir(path)
        self.lista_imagenes += lista_imagenes 
        for file in lista_imagenes:
            normalized_img = normalize_png_file(path + '/' + file, (image_width, image_height))
            self.list_img.append(normalized_img)
            self.list_label.append(label)
        
    
    def __getitem__(self, index):
        return (self.list_img[index],self.list_label[index], self.lista_imagenes[index])
        
    def __len__(self):
        return len(self.lista_imagenes)



# TRAINING THE MODEL

class NetTrainer:

    def __init__(self, loader):
        self.train_losses  = []
        self.train_counter = []
        self.loader        = loader

    def train_counter(self):
        return self.train_counter

    def train_losses(self):
        return self.train_losses
        
    def train(self, network, optimizer,epoch):
        datos_pasados = 0
        network.train()
        for batch_idx, (data, target, filename) in enumerate(self.loader):
            datos_pasados += len(data)
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            #if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, datos_pasados, len(self.loader.dataset),
            100. * batch_idx / len(self.loader), loss.item()))
        self.train_losses.append(loss.item())
        self.train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(self.loader.dataset)))
        filename = os.path.basename(sys.argv[0])
        filename_without_extension = os.path.splitext(filename)[0]
        modelname = filename + '_model.pth'
        optimizername = filename + '_optimizer.pth'
        torch.save(network.state_dict(), modelname)
        torch.save(optimizer.state_dict(), optimizername)

class NetTester:

    def __init__(self, loader, n_epochs):
        self.test_losses  = []
        self.test_counter = [i*len(loader.dataset) for i in range(n_epochs + 1)]
        self.loader       = loader;

    def test_losses(self):
        return self.test_losses

    def test_counter(self):
        return self.test_counter
        
    def test(self, network, msg):
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target, filename in self.loader:
                #print('data', data.shape)
                output = network(data)
                #print('output', output)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(self.loader.dataset)
        self.test_losses.append(test_loss)
        print('{}: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            msg, test_loss, correct, len(self.loader.dataset),
            100. * correct / len(self.loader.dataset)))


class NetEval:

    def __init__(self, loader, n_epochs):
        self.test_losses  = []
        self.test_counter = [i*len(loader.dataset) for i in range(n_epochs + 1)]
        self.loader       = loader;

    def test_losses(self):
        return self.test_losses

    def test_counter(self):
        return self.test_counter
        
    def eval(self, network, msg):
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.loader:
                #print('data', data.shape)
                output = network(data)
                #print('output', output)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(self.loader.dataset)
        self.test_losses.append(test_loss)
        print('{}: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            msg, test_loss, correct, len(self.loader.dataset),
            100. * correct / len(self.loader.dataset)))


        
