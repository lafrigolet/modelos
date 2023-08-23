from torch.utils.data import DataLoader, Dataset
from torch.utils.data import dataset
from torchtext.datasets import WikiText2
import torch
import models.fnn as M
import models.utils as U

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
    model = M.FNN([10, 5, 2, 4])
    print(model)
    M.load(model, 'fnn.pth')

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


