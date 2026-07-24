import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters (kept small for now while we build up the architecture --
# we'll scale these up once everything works)
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
# ------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        # three separate linear projections of the SAME input x.
        # key   = "what do I contain"      (what this token offers, if asked)
        # query = "what am I looking for"  (what this token wants from others)
        # value = "what I'll actually communicate" if attended to
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # tril is NOT a learnable parameter -- register_buffer keeps it on the
        # model (moves with .to(device), saved in state_dict) but excludes it
        # from gradients/optimizer updates
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)

        # affinity scores: how much does each query "match" each key?
        # this replaces the hardcoded zeros from attention_math.py's wei --
        # now the raw scores are DATA-DEPENDENT, learned from x itself.
        wei = q @ k.transpose(-2, -1)  # (B,T,hs) @ (B,hs,T) -> (B,T,T)
        # scale by 1/sqrt(head_size): without this, wei's variance grows with
        # head_size, softmax saturates towards one-hot, and gradients vanish
        wei = wei * C**-0.5
        # causal mask: same trick as attention_math.py's Version 3 --
        # a token can only attend to itself and the past, never the future
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # (B,T,T), each row sums to 1

        # weighted aggregation of VALUES (not the raw x!) using those weights
        v = self.value(x)  # (B,T,head_size)
        out = wei @ v       # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # token identity: which character is this? -> n_embd-dim vector
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # token position: where in the block (0..block_size-1) is this? -> n_embd-dim vector
        # needed because attention itself has no built-in notion of order --
        # it treats its input as an unordered set unless we inject position info
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # single self-attention head, same width as n_embd for now (before
        # we split into multiple smaller heads in the next step)
        self.sa_head = Head(n_embd)
        # project from the n_embd "thinking space" back out to vocab-sized logits
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)                        # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,n_embd)
        x = tok_emb + pos_emb  # (B,T,n_embd) -- broadcasts pos_emb across the batch
        x = self.sa_head(x)    # (B,T,n_embd) -- each token now mixes in info from its past
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop to the last block_size tokens -- position_embedding_table
            # only has entries for positions 0..block_size-1
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
