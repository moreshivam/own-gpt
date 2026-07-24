import torch
from model import GPTLanguageModel
from vocab import load_vocab, make_encoder_decoder

# --- fine-tuning hyperparameters ---
# Lower learning rate than pretraining, and far fewer iterations: SFT is
# meant to nudge an already-competent model toward a new format/behavior,
# not relearn everything from scratch. Too high an LR here risks
# "catastrophic forgetting" -- wrecking the pretrained weights instead of
# adapting them.
batch_size = 8
max_iters = 800
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
# ------------

torch.manual_seed(1337)

stoi, itos, vocab_size = load_vocab('vocab.json')
encode, decode = make_encoder_decoder(stoi, itos)

checkpoint = torch.load('gpt_pretrained.pt', map_location=device)
config = checkpoint['config']
block_size = config['block_size']  # fixed by the pretrained checkpoint's position embeddings

model = GPTLanguageModel(**config)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
print(f"loaded pretrained checkpoint: {sum(p.numel() for p in model.parameters())/1e6:.3f}M params")

with open('finance_qa.txt', 'r', encoding='utf-8') as f:
    finance_text = f.read()

data = torch.tensor(encode(finance_text), dtype=torch.long)
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

@torch.no_grad()
def sample(prompt, max_new_tokens=150):
    model.eval()
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens)[0].tolist()
    model.train()
    return decode(out)

@torch.no_grad()
def char_accuracy(split='val', n_batches=100):
    # Teacher-forced next-character accuracy: feed the TRUE context (not the
    # model's own generated output), and check how often its top prediction
    # matches the actual next character. This is a real, comparable % metric
    # -- unlike loss, it doesn't require any math background to interpret.
    model.eval()
    correct, total = 0, 0
    for _ in range(n_batches):
        X, Y = get_batch(split)
        logits, _ = model(X, Y)          # (B*T, vocab_size) -- flattened by forward()
        preds = logits.argmax(dim=-1)    # top predicted character id per position
        correct += (preds == Y.view(-1)).sum().item()
        total += Y.numel()
    model.train()
    return correct / total

prompt = "Q: What is a dividend?\nA:"

losses_before = estimate_loss()
acc_before = char_accuracy('val')
print("\n=== BEFORE fine-tuning (base Shakespeare-pretrained model) ===")
print(f"finance val loss:     {losses_before['val']:.4f}  (perplexity {torch.exp(losses_before['val']):.2f})")
print(f"next-char accuracy:   {acc_before*100:.2f}%  (on held-out finance_qa.txt)")
print(sample(prompt))

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("\n=== fine-tuning on finance_qa.txt ===")
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

losses_after = estimate_loss()
acc_after = char_accuracy('val')
print("\n=== AFTER fine-tuning ===")
print(f"finance val loss:     {losses_after['val']:.4f}  (perplexity {torch.exp(losses_after['val']):.2f})")
print(f"next-char accuracy:   {acc_after*100:.2f}%  (on held-out finance_qa.txt)")
print(sample(prompt))

print("\n=== IMPROVEMENT ===")
print(f"accuracy: {acc_before*100:.2f}% -> {acc_after*100:.2f}%  "
      f"({(acc_after-acc_before)*100:+.2f} percentage points, "
      f"{(acc_after/acc_before - 1)*100:+.1f}% relative)")
print(f"val loss: {losses_before['val']:.4f} -> {losses_after['val']:.4f}  "
      f"({(1 - losses_after['val']/losses_before['val'])*100:.1f}% reduction)")

torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
}, 'gpt_finetuned.pt')
print('\nsaved checkpoint to gpt_finetuned.pt')
