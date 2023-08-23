from torch.utils.data import DataLoader, Dataset
from torch.utils.data import dataset
from torchtext.datasets import WikiText2
import torch
import models.fnn as M
import models.utils as U

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


def train():

    tokenizer, vocab = U.load_embeddings()

    def data_process(raw_text_iter: dataset.IterableDataset) -> torch.Tensor:
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
    
    model = M.FNN([input_size, 1000, 1000, 1000, 1000, 1000, num_classes])

    print("torch.cuda.device_count ", torch.cuda.device_count())
    """
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    """
    print(model)

    M.train(model, train_loader, test_loader, n_epochs, learning_rate)
    M.save(model, 'fnn')



train()
