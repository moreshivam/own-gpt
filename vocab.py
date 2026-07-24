import json

# Vocab is built ONCE from the union of every corpus the model will ever see
# across all training stages (pretrain + fine-tune), so character-to-id
# mappings stay identical everywhere. If we instead rebuilt the vocab
# separately per file, the same integer could mean a different character in
# each stage, silently corrupting a fine-tuned checkpoint loaded on top of
# pretrained embeddings.

def build_vocab(*texts):
    chars = sorted(list(set(''.join(texts))))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos, vocab_size

def make_encoder_decoder(stoi, itos):
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return encode, decode

def save_vocab(stoi, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(stoi, f, ensure_ascii=False)

def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        stoi = json.load(f)
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos, len(stoi)
