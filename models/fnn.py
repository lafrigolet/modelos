from torchtext.datasets import WikiText2
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import dataset
from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optim
import torch
import copy

class FNNModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes, learning_rate):
        super(FNNModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size2, num_classes)
        self.softmax = nn.Softmax(dim=1)

        
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu1(out)
        out = self.layer2(out)
        out = self.relu2(out)
        out = self.output_layer(out)
        out = self.softmax(out)
        return out



    def train_model(self, train_loader, n_epochs, learning_rate):
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # optimizer = optim.Rprop(network.parameters(), lr=learning_rate)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
        for epoch in range(n_epochs):
            self.train()
            running_loss = 0.0
        
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                inputs = inputs.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                labels = labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
                running_loss += loss.item()
        
            print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {running_loss / len(train_loader)}")

"""
    TO DELETE
    def train_model(self, train_loader, epoch):
        datos_pasados = 0
        super().train()
        for batch_idx, (data, target) in enumerate(train_loader):
            datos_pasados += len(data)
            self.optimizer.zero_grad()
            data = data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            output = self(data)
            target = target.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            
        return datos_pasados, batch_idx, loss
"""

# Define your custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, context_size):
        self.data = data
        self.len  = len(data) - context_size
        self.context_size = context_size
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        data = self.data[index:index + self.context_size]
        labels = torch.FloatTensor(vocab_len).fill_(0)
        next_token = self.data[index + self.context_size]
        labels[int(next_token)] = 1.

        return data, labels


"""
TO DELETE
def train(model, train_loader, test_loader, n_epochs):
    # Train the model
    for epoch in range(1, n_epochs + 1):
        datos_pasados, batch_idx, loss = model.train_model(train_loader, epoch)
            
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, datos_pasados, len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
"""


"""
        if epoch % 10 == 0:
            results, labels, test_loss, correct = model.test(test_loader)
            print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

            results, labels, test_loss, correct = model.test(train_loader)
            print('Train set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, len(train_loader.dataset),
                100. * correct / len(train_loader.dataset)))
"""



    
train_iter = WikiText2(root='data/wikitext-2', split='train')

"""
for item in train_iter:
    print('--------------------------------------------------------------------------')
    print(item)
    print('--------------------------------------------------------------------------')
"""

tokenizer = get_tokenizer('basic_english')
tokenized_train_iter = map(tokenizer, train_iter)

"""
for token in tokenized_train_iter:
    print(token)
"""

vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

"""
# Inspect callable methods for a python object
for name in dir(vocab):
    if callable(getattr(vocab, name)):
        print(name)
"""

# print(vocab.lookup_token(780))
      

"""
# Iterate through the vocabulary items
for token, index in vocab.get_stoi().items():
    print(f"Token: {token}, Index: {index}")

# Access the vocabulary size
vocab_size = len(vocab)
print(f"Vocabulary Size: {vocab_size}")

# Access the index of a specific token
index_of_unk = vocab['<unk>']
print(f"Index of <unk>: {index_of_unk}")
"""

def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.float32) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

# ``train_iter`` was "consumed" by the process of building the vocab,
# so we have to create it again
#train_iter, val_iter, test_iter = WikiText2()
train_iter, val_iter, test_iter = WikiText2(root='data/wikitext-2', split=('train', 'valid', 'test'))
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

"""
# Assuming train_data is a torch.Tensor
print(len(train_data))
print("Train Data (First 100 Elements):")
print(train_data[:100])
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





# Prepare Train Data Set ############################################################

# Example data and labels
#data = torch.randn(100, 10)  # Example: 100 samples, each with 10 features
#labels = torch.randint(0, 5, (100,))  # Example: 100 labels, 5 classes

context_size = 1000 # number of tokens to use as context
vocab_size = len(vocab) # len of vocab

train_data = train_data[0:100000] # Reduce the train_data set
vocab_len = len(vocab)
train_data_len = len(train_data)

"""
TO DELETE
data = torch.FloatTensor(train_data_len - context_size, context_size).fill_(0)
labels = torch.FloatTensor(train_data_len - context_size, vocab_len).fill_(0)

for i in range(train_data_len - context_size):
    data[i] = train_data[i:i + context_size]
    next_token = train_data[i + context_size]
    labels[i, int(next_token)] = 1.
    
print(len(data))
print(len(labels))
#print(len(labels[55]))
#print(labels[55])
"""
# Instantiate your custom dataset
custom_dataset = CustomDataset(train_data, context_size)

# Create a DataLoader
batch_size = 64  # Number of samples in each batch
train_loader = DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=True)
    

# Build and train model ######################################################################

# Define model parameters
input_size =  context_size  # Specify the input size or number of features
hidden_size1 = 200  # Number of neurons in the first hidden layer
hidden_size2 = 100  # Number of neurons in the second hidden layer
num_classes = vocab_size   # Number of output classes
learning_rate = 0.0001
n_epochs = 100

# Instantiate the model
model = FNNModel(input_size, hidden_size1, hidden_size2, num_classes, learning_rate)
print(model)

model.train_model(train_loader, n_epochs, learning_rate)
