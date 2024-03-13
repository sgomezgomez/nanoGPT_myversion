import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import requests
from datetime import datetime

## ----------------------------------
## Hyperparameters
batch_size = 16     # Number of independent sequences to process in parallel
block_size = 16      # Maximum context length for prediction
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4 # Decreased learning rate because self-attention can't tolerate high learning rates
device = 'cuda' if torch.cuda.is_available() else 'cpu' # In case there's a GPU available
eval_iters = 200
n_embd = 32
n_layer = 8
n_head = 8
dropout = 0.2
## ----------------------------------
## Model
model_name = 'NanoGPT_v11'
model_log = './output/model_log.txt'
## ----------------------------------

# Log to terminal and model log file
def log_text(text):
    text = str(model_name) + '_' + str(datetime.now()) + ': ' + str(text)
    open(model_log, 'a').write((text + '\n')) # Log file
    print(text) # Terminal

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
log_text('Mini Shakespeare data loaded.')

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
log_text('Training and dev sets partitioned.')

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

## Model classes
class BatchedSelfAttention(nn.Module):

    def __init__(self, n_head, head_size):
        super().__init__()
        self.n_head = n_head
        self.head_size = head_size
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)) # (1, 1, block_size, block_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.ln_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        assert C == self.n_head * self.head_size, f"dimension mismatch between C: {C} and n_head * head_size {self.n_head * self.head_size}"
        # Key, Query, and Value vectors
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, n_head, T, head_size)
        q = self.query(x).view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # (B, n_head, T, head_size)
        v = self.value(x).view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # (B, n_head, T, head_size)
        # Compute attention scores, affinities
        wei = q @ k.transpose(-2, -1) * (C**-0.5) # (B, n_head T, head_size) @ (B, n_head, head_size, T) --> (B, n_head, T, T)
        wei = wei.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)
        out = wei @ v # (B, num_heads, T, T) @ (B, n_head, T, head_size) --> (B, n_head, T, head_size)
        out = out.transpose(1, 2).contiguous().view(B, T, C) # Reshape to (B, T, C)
        # Output projection
        out = self.ln_proj(out)
        out = self.resid_dropout(out)
        return out

class FeedForward(nn.Module): # MLP
    """Single linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        #self.net = nn.Sequential(
        #    nn.Linear(n_embd, 4 * n_embd),
        #    nn.ReLU(),
        #    nn.Linear(4 * n_embd, n_embd), # Projection layer, Linear transformation of the outcome from the previous layer
        #    nn.Dropout(dropout),
        #)
        self.ln1 = nn.Linear(n_embd, 4 * n_embd)
        self.relu = nn.ReLU()
        self.ln_proj = nn.Linear(4 * n_embd, n_embd) # Projection layer, Linear transformation of the outcome from the previous layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        #return self.net(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.ln_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """Transformer block: communication + computation (feed forward)"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.ln1 = nn.LayerNorm(n_embd) # Layer norm
        #self.sa = MultiHeadAttention(n_head, head_size) # Multi head attention
        self.sa = BatchedSelfAttention(n_head, head_size) # Batched self attention block
        self.ln2 = nn.LayerNorm(n_embd) # Layer norm
        self.feedforward = FeedForward(n_embd) # Feed forward
    
    def forward(self, x):
        # Residual connection x + ...
        x = self.ln1(x) # Layer norm before transformation
        x = x + self.sa(x) # Apply one head of self-attention (B, T, C)
        x = self.ln2(x) # Layer norm before transformation
        x = x + self.feedforward(x) # Apply feed forward
        return x

## GPT Language Model Class
class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # vocab_size and n_embd are defined as global variables
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # Positional information
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_final = nn.LayerNorm(n_embd) # Final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # lm_head: short for language model head
        # Init Linear or Embedding weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('ln_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C) where C is n_embd
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
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

## Create GPT Model
model = GPTLanguageModel(vocab_size)
m = model.to(device) # Moved to GPU when available
log_text('Model initialized.')
log_text((str (sum(p.numel() for p in m.parameters())/1e6) + ' Million parameters'))

## Pytorch Optimizer Object
optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)

## Training Loop
for iter in range(max_iters):
    # Evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        #print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        log_text(('Step  ' + str(iter) + '; Train loss ' + str(losses['train'].item()) + ', Val loss ' + str(losses['val'].item())))
    
    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
log_text('Model optimization completed.')

## Generate from the Model
context = torch.zeros((1, 1), dtype=torch.long, device=device) # Created on device
gen = m.generate(context, max_new_tokens=500)[0].tolist() # Calls generate function and unplucks the 0th batch dimension
log_text(decode(gen))

open('./output/more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist())) # Save output to file
log_text('more.txt file populated with generated text from the model.')

# Save the model
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_file = './model/' + str(model_name) + '_' + timestamp + '.pth'
torch.save(m, model_file)
log_text(str(model_file) + ' saved.')

# Later the model
#m2 = torch.load(model_file)