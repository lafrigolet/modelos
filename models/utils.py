from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import random

# Seguramente este fichero sea para borrar completamente, me falta comprobar si load_embeddings
# se usa para algo

def load_embeddings():
    # Load Text Processing Embeddings ############################################################
    
    train_iter = WikiText2(root='data/wikitext-2', split='train')

    """
    for item in train_iter:
    print('--------------------------------------------------------------------------')
    print(item)
    print('--------------------------------------------------------------------------')
    """

    tokenizer = get_tokenizer('basic_english')
    
    """
    tokenized_train_iter = map(tokenizer, train_iter)
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

    return tokenizer, vocab
