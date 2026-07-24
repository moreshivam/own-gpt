import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    # pick batch_size random starting offsets into the data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # x = the block_size chars starting at each offset; y = the SAME window
    # shifted one char to the right, so y[t] is always the "next char" for x[t]
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # no backprop needed here, saves memory/time
def estimate_loss():
    # average loss over many random batches instead of trusting one noisy
    # batch's loss -- gives a much more stable train/val curve to read
    out = {}
    model.eval() # switches off training-only behavior (e.g. dropout, later)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # switch back to training mode
    return out

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # A (vocab_size, vocab_size) table: row i IS the next-token logits for
        # "the current token is i". No hidden layers, no context beyond the
        # current char -- this table literally *is* the entire model.
        # Row i, column j = an (unnormalized) score for "char j follows char i".
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        # Embedding lookup: for every position, fetch its row from the table.
        # Note idx itself is never used as "context" beyond this row lookup --
        # positions 0..T-1 are all scored independently, in parallel.
        logits = self.token_embedding_table(idx) # (B,T,C) where C = vocab_size

        if targets is None:
            loss = None
        else:
            # cross_entropy wants (N, C) logits and (N,) targets, so flatten
            # the batch and time dimensions into one N = B*T axis of examples.
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context.
        # Autoregressive loop: predict one char, append it, repeat.
        for _ in range(max_new_tokens):
            # get the predictions for every position we have so far
            logits, loss = self(idx)
            # only the LAST time step's logits matter for "what comes next" --
            # everything earlier is discarded (bigram only conditions on the
            # single most recent char anyway, so this is all it ever needed)
            logits = logits[:, -1, :] # becomes (B, C)
            # turn scores into a probability distribution over the vocab
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample (not argmax!) so generation isn't deterministic/repetitive
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence, feed back in next loop
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # forward pass: get loss, then the standard PyTorch backprop cycle
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True) # clear old gradients
    loss.backward()                       # compute new gradients
    optimizer.step()                      # nudge weights to reduce loss

# generate from the model, starting from a single "newline" token (idx 0)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
