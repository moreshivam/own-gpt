import os
import torch
from model import GPTLanguageModel
from vocab import load_vocab, make_encoder_decoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# prefer the fine-tuned checkpoint; fall back to the base pretrained one if
# finetune_finance.py hasn't been run yet
checkpoint_path = 'gpt_finetuned.pt' if os.path.exists('gpt_finetuned.pt') else 'gpt_pretrained.pt'
print(f"loading {checkpoint_path} ...")

stoi, itos, vocab_size = load_vocab('vocab.json')
encode, decode = make_encoder_decoder(stoi, itos)

checkpoint = torch.load(checkpoint_path, map_location=device)
model = GPTLanguageModel(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

known_chars = set(stoi.keys())

print("Type a finance question (or 'quit' to exit).")
print("Note: this is a ~200K-parameter char-level model trained on ~100 Q&A")
print("pairs -- expect plausible-looking text, not reliable factual answers.\n")

while True:
    question = input("Q: ").strip()
    if question.lower() in ('quit', 'exit'):
        break
    if not question:
        continue

    unknown = set(question) - known_chars
    if unknown:
        print(f"(skipping characters not in vocab: {unknown})\n")
        question = ''.join(c for c in question if c in known_chars)

    prompt = f"Q: {question}\nA:"
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=200)[0].tolist()
    generated = decode(out)[len(prompt):]

    # stop at the next "Q:" if the model starts hallucinating a new question,
    # so output looks like one answer instead of running on indefinitely
    if '\nQ:' in generated:
        generated = generated[:generated.index('\nQ:')]

    print(f"A:{generated}\n")
