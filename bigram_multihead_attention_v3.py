import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import requests

## Hyperparameters
batch_size = 32     # Number of independent sequences to process in parallel
block_size = 8      # Maximum context length for prediction
max_iters = 13000
eval_interval = 300
learning_rate = 1e-3 # Decreased learning rate because self-attention can't tolerate high learning rates
# In case there's a GPU available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
## ----------------------------------

## Reprodudibility
torch.manual_seed(1337)

## Load Data
# Download and read data
# Code used in Collab notebook
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# Code snippet to download data from: https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare/prepare.py
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)
with open(input_file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Character mappings
stoi = { ch:i for i, ch in enumerate(chars) } # From characters to integers
itos = { i:ch for i, ch in enumerate(chars) } # From integers to characters
# Encoder
encode = lambda s: [stoi[c] for c in s]
# Decoder
decode = lambda l: ''.join([itos[i] for i in l])
# Encode entire dataset
data = torch.tensor(encode(text), dtype=torch.long)

## Data Splits
n = int(0.9*len(data)) # First 90% will be used for training, the rest for validation
train_data = data[:n]
val_data = data[n:]

## Helper Functions
def get_batch(split):
    # Generate small batch of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) # Moved to GPU when available
    return x, y

@torch.no_grad() # For more efficient memory use
def estimate_loss():
    out = {}
    model.eval() # Set model to eval phase
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Set model back to train phase
    return out

class Head(nn.Module): # Self-attention head
    """Single head of self-attention"""
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        # Query, Key, Value vectors
        k = self.key(x)     # (B, T, C)
        q = self.query(x)   # (B, T, C)
        v = self.value(x)   # (B, T, C)
        # Compute attention scores, affinities
        wei = q @ k.transpose(-2, -1) * (C**-0.5) # (B, T, C) @ (B, C, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v # (B, T, C)
        return out

class MultiHeadAttention(nn.Module): # Multi head attention
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

## Bigram Language Model Class
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # vocab_size and n_embd are defined as global variables
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # Positional information
        self.sa_heads = MultiHeadAttention(4, n_embd//4) # Multi head attention, i.e. 4 heads of 8-dimensional self-attentio blocks
        self.lm_head = nn.Linear(n_embd, vocab_size) # lm_head: short for language model head

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C) where C is n_embd
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.sa_heads(x) # Apply one head of self-attention (B, T,C)
        logits = self.lm_head(x) # (B, T, vocab_size)
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            # B: batch size
            # T: block size
            # C: vocab size
            logits = logits.view(B*T, C) # Stretch out logits array
            targets = targets.view(B*T) # Stretch out targets array
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        #idx is (B, T) array of indices in the corrent context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # Obtain predictions
            logits, loss = self(idx_cond)
            # Last time step, what comes next
            logits = logits[:, -1, :] # (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1) # (B, C)
            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

## Create Bigram Model
model = BigramLanguageModel(vocab_size)
m = model.to(device) # Moved to GPU when available

## Pytorch Optimizer Object
optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

## Training Loop
for iter in range(max_iters):
    # Evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

## Generate from the Model
context = torch.zeros((1, 1), dtype=torch.long, device=device) # Created on device
gen = m.generate(context, max_new_tokens=100)[0].tolist() # Calls generate function and unplucks the 0th batch dimension
print(decode(gen))