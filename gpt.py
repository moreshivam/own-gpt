import torch
from model import GPTLanguageModel
from vocab import build_vocab, make_encoder_decoder, save_vocab

# hyperparameters -- CPU-friendly sizing (benchmarked at ~0.07s/iter on an
# 8-core CPU, ~6 min for the full run). The video's GPU-sized config
# (n_embd=384, block_size=256, ...) measured ~23 HOURS on the same machine,
# so we deliberately stay here instead.
batch_size = 16
block_size = 32
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.1
# ------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    shakespeare_text = f.read()
with open('finance_qa.txt', 'r', encoding='utf-8') as f:
    finance_text = f.read()

# vocab is built from BOTH corpora even though this script only trains on
# Shakespeare -- so the character ids stay identical when we later load this
# checkpoint into finetune_finance.py and continue training on finance_qa.txt
stoi, itos, vocab_size = build_vocab(shakespeare_text, finance_text)
encode, decode = make_encoder_decoder(stoi, itos)
save_vocab(stoi, 'vocab.json')
print(f'vocab_size = {vocab_size}')

data = torch.tensor(encode(shakespeare_text), dtype=torch.long)
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

model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout)
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# checkpoint the pretrained weights + the exact hyperparameters needed to
# reconstruct this architecture, so finetune_finance.py can load it exactly
torch.save({
    'model_state_dict': model.state_dict(),
    'config': {
        'vocab_size': vocab_size, 'n_embd': n_embd, 'n_head': n_head,
        'n_layer': n_layer, 'block_size': block_size, 'dropout': dropout,
    },
}, 'gpt_pretrained.pt')
print('saved checkpoint to gpt_pretrained.pt')
