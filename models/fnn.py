from torchtext.datasets import WikiText2
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import dataset
from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optim
import torch
import copy
import utils as U
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

        
def train_model(model, train_loader, test_loader, n_epochs, learning_rate):
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
    def __init__(self, data, context_size, vocab_size):
        self.data = data
        self.data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.len  = len(data) - context_size
        self.context_size = context_size
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        #data = self.data[index:index + self.context_size]
        data = self.data.narrow(0, index, self.context_size)
        labels = torch.FloatTensor(self.vocab_size).fill_(0)
        next_token = self.data[index + self.context_size]
        #print(data.to(torch.int), next_token)
        labels[int(next_token)] = 1.

        data = data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        labels = labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

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


def train():

    tokenizer, vocab = U.load_embeddings()

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

    context_size = 100 # number of tokens to use as context
    vocab_size = len(vocab) # len of vocab

    #train_data = train_data[0:100000] # Reduce the train_data set
    train_data_len = len(train_data)

    """
TO DELETE
data = torch.FloatTensor(train_data_len - context_size, context_size).fill_(0)
labels = torch.FloatTensor(train_data_len - context_size, vocab_size).fill_(0)

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
    train_custom_dataset = CustomDataset(train_data, context_size, vocab_size)
    test_custom_dataset = CustomDataset(test_data, context_size, vocab_size)

    # Create a DataLoader
    batch_size = 64  # Number of samples in each batch
    train_loader = DataLoader(dataset=train_custom_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(dataset=test_custom_dataset, batch_size=batch_size, shuffle=False)

    # Build and train model ######################################################################

    # Define model parameters
    input_size =  context_size  # Specify the input size or number of features
    num_classes = vocab_size   # Number of output classes
    learning_rate = 0.0001
    n_epochs = 50


    print(f"Hiperparameters: context_size {context_size}, learning_rate {learning_rate}, batch_size {batch_size}, n_epochs {n_epochs}");
    # Instantiate the model
    
    model = FNN([input_size, 1000, 1000, 1000, 1000, 1000, num_classes])

    print("torch.cuda.device_count ", torch.cuda.device_count())
    """
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    """
    print(model)

    train_model(model, train_loader, test_loader, n_epochs, learning_rate)
    save(model, 'fnn')

class TokensDataSet(Dataset):
    def __init__(self, data):
        self.data = data
#        self.data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.len = len(data)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.data[index]
        
    
def chat():
    tokenizer, vocab = U.load_embeddings()

    def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
        """Converts raw text into a flat Tensor."""
        data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.float32) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    vocab_size = len(vocab)
    context_size = 100 # number of tokens to use as context

    tokens  = ['<unk>' for _ in range(context_size)]
    tokens = (tokens + 'Once upon a time in hollywood, a man found a lover and the reason why you should keep searching is that the lover is always there, just searching for you, so dont hesitate and be patience. Luck helps to those keeping patience. Once upon a time in hollywood, a man found a lover and the reason why you should keep searching is that the lover is always there, just searching for you, so dont hesitate and be patience. Luck helps to those keeping patience. Once upon a time in hollywood, a man found a lover and the reason why you should keep searching is that the lover is always there, just searching for you, so dont hesitate and be patience. Luck helps to those keeping patience. Once upon a time in hollywood, a man found a lover and the reason why you should keep searching is that the lover is always there, just searching for you, so dont hesitate and be patience. Luck helps to those keeping patience. Once upon a time in hollywood, a man found a lover and the reason why you should keep searching is that the lover is always there, just searching for you, so dont hesitate and be patience. Luck helps to those keeping patience. Once upon a time in hollywood, a man found a lover and the reason why you should keep searching is that the lover is always there, just searching for you, so dont hesitate and be patience. Luck helps to those keeping patience. Once upon a time in hollywood, a man found a lover and the reason why you should keep searching is that the lover is always there, just searching for you, so dont hesitate and be patience. Luck helps to those keeping patience. '.lower().split())[-context_size:]

    print(tokens, len(tokens))

    tokens_iter = TokensDataSet(tokens)
    tokens_data = data_process(tokens_iter)[-context_size:]

    print(tokens_data, len(tokens_data))

    input_size =  context_size  # Specify the input size or number of features
    hidden_size1 = 10  # Number of neurons in the first hidden layer
    hidden_size2 = 5  # Number of neurons in the second hidden layer
    num_classes = vocab_size   # Number of output classes

    #model = FNN([input_size, 100, 100, num_classes])
    model = FNN([10, 5, 2, 4])
    print(model)
    load(model, 'fnn.pth')

    model.eval()
    outputs = model(tokens_data.view(1,-1))
    _, predicted = torch.max(outputs, 1)
    print(vocab.lookup_token(predicted[0]))

    exit()

    while True:
        user_input = input("You: ")
        print("Bot:", user_input)
        tokens = (tokens + user_input.lower().split())[-context_size:]

        print(tokens, len(tokens))
    
        if user_input.lower() == "exit":
            print("Bot: Goodbye!")
            break



#chat()

train()



