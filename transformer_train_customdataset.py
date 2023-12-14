import models.transformer as M
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import dataset
from torch import nn, Tensor
import torch
from tempfile import TemporaryDirectory
import os
import time
import math
from torch.utils.data import DataLoader, Dataset

# Define your custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, bsz):
        seq_len = data.size(0) // bsz
        data = data[:seq_len * bsz]
        self.data = data.view(bsz, seq_len).t().contiguous()
#        print(f"self.data.shape {self.data.shape}")
        self.data = self.data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def __len__(self):
        return len(self.data) - 1
    
    def __getitem__(self, i):
#        bptt = 35
#        seq_len = min(bptt, len(self.data) - 1 - i)
#        data = self.data[i:i+seq_len]
#        target = self.data[i+1:i+1+seq_len].reshape(-1)
        data = self.data[i]
        target = self.data[i+1]
        return data, target


def train(model, train_loader, test_loader, criterion, optimizer, epochs):
#    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
#    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.Rprop(network.parameters(), lr=learning_rate)
    
    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.
        log_interval = 200
        start_time = time.time()
        num_batches = len(train_loader) 
        batch = 0
        
        for inputs, labels in train_loader:
#            _, expected = torch.max(labels, 1) 
#            print(f"inputs {inputs.shape}, labels {labels.shape}")
            optimizer.zero_grad()
            outputs = model(inputs)
#            print(f"outputs {outputs.shape}, labels {labels.shape}")
            outputs_flat = outputs.view(-1, ntokens)
            labels = labels.reshape(-1)
#            print(f"outputs_flat {outputs_flat.shape}, labels {labels.shape}")
            loss = criterion(outputs_flat, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            batch += len(inputs[0])
            if batch % log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                ppl = math.exp(cur_loss)
                print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                      f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                      f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
                total_loss = 0
                start_time = time.time()


#            print(f"Epoch [{epoch}/{epochs}], Loss: {running_loss / len(train_loader)}")

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
            
    

train_iter = WikiText2(root='data/wikitext-2', split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

# ``train_iter`` was "consumed" by the process of building the vocab,
# so we have to create it again
#train_iter, val_iter, test_iter = WikiText2()
train_iter, val_iter, test_iter = WikiText2(root='data/wikitext-2', split=('train', 'valid', 'test'))
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Instantiate your custom dataset
bsz = 40
batch_size = 35
eval_batch_size = 10

train_custom_dataset = CustomDataset(train_data, bsz)
test_custom_dataset = CustomDataset(test_data, bsz)

# Create a DataLoader
train_loader = DataLoader(dataset=train_custom_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(dataset=test_custom_dataset, batch_size=eval_batch_size, shuffle=False)


ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 2  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability
model = M.TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)


criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
best_val_loss = float('inf')
epochs = 10

train(model, train_loader, test_loader, criterion, optimizer, epochs)

exit()

"""
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        M.train(model, train_data, criterion, lr, optimizer, scheduler, epoch, ntokens)
        val_loss = M.evaluate(model, val_data, criterion, ntokens)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)

        scheduler.step()
        
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states
"""

test_loss = M.evaluate(model, test_data, criterion, ntokens)
test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)    
