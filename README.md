# Financial Q&A GPT — A Transformer Language Model Built From Scratch

A GPT-style transformer language model implemented from first principles in PyTorch — no HuggingFace, no pretrained weights, no external tokenizer. Built by implementing every component by hand (tokenizer, self-attention, multi-head attention, transformer blocks, training loop), then extended into a two-stage **pretrain → fine-tune** pipeline: general-language pretraining on Shakespeare, followed by domain-specific fine-tuning on a curated financial-markets Q&A dataset.

Architecture follows Andrej Karpathy's ["Let's build GPT: from scratch, in code, spelled out."](https://www.youtube.com/watch?v=kCc8FmEb1nY), extended with model checkpointing, a shared reusable architecture module, a second training stage, and a quantitative before/after evaluation.

## What's implemented

- **Custom character-level tokenizer** (`vocab.py`) — no external tokenizer library, vocabulary built from the union of both training corpora so token IDs stay valid across both training stages
- **Self-attention from first principles** (`attention_math.py`) — derives the causal-averaging trick (loop → triangular matmul → masked softmax) before it's used in the real model
- **Full transformer architecture** (`model.py`) — multi-head self-attention, position-wise feedforward, residual connections, pre-norm LayerNorm, dropout, GPT-2-style weight initialization
- **Two-stage training pipeline** — pretrain on general text (`gpt.py`), then fine-tune the same checkpoint on a new domain (`finetune_finance.py`), with model checkpointing so the second stage resumes from the first
- **Quantitative evaluation** — next-character prediction accuracy and validation loss/perplexity, measured before and after fine-tuning (not just eyeballed sample text)
- **Interactive CLI** (`chat.py`) — query the fine-tuned model with your own questions

## Results

| Stage | Params | Val loss | Notes |
|---|---:|---:|---|
| Bigram baseline (no attention) | 4.2K | plateaus ~2.47 | can only ever see 1 prior character |
| Single self-attention head | 7.5K | ~2.40 | first model to use context beyond 1 char |
| Multi-head + transformer blocks (dev config) | 55K | ~2.06 | 4 heads x 4 blocks, residuals + LayerNorm |
| **Pretrained on Shakespeare** (final config) | **210.6K** | **1.9586** | 4 heads x 4 layers, block_size=32, dropout |

**Fine-tuning on financial Q&A** (same checkpoint, continued training on `finance_qa.txt`):

| Metric | Before fine-tuning | After fine-tuning | Change |
|---|---:|---:|---:|
| Next-character accuracy (held-out) | 31.24% | 50.55% | **+19.3 points (+61.8% relative)** |
| Validation loss | 2.4348 | 1.6342 | **-32.9%** |
| Validation perplexity | 11.41 | 5.13 | more than 2x lower |

Accuracy is measured **teacher-forced** (true context in, top-1 prediction vs. true next character) on a held-out 10% split of the financial Q&A data never seen during fine-tuning — not a subjective read of generated text.

## Project structure

```
model.py              shared transformer architecture (Head, MultiHeadAttention,
                       FeedForward, Block, GPTLanguageModel) -- used by both
                       training stages so checkpoints load identically
vocab.py               tokenizer: build/save/load a shared character vocabulary
bigram.py               baseline bigram language model (no attention), for comparison
attention_math.py       derivation of the self-attention math trick from scratch
gpt.py                  stage 1: pretrains on input.txt, saves gpt_pretrained.pt
finetune_finance.py     stage 2: fine-tunes on finance_qa.txt, saves gpt_finetuned.pt
chat.py                 interactive CLI -- ask the fine-tuned model questions
input.txt               tinyshakespeare dataset (pretraining corpus)
finance_qa.txt          ~100 hand-curated financial markets Q&A pairs (fine-tuning corpus)
vocab.json              saved character-to-id mapping shared by both stages
```

## Setup & usage

Requires Python 3.10+ and PyTorch (CPU is enough — this project is deliberately sized to train in minutes on a CPU, no GPU required):

```bash
pip install torch

python gpt.py               # stage 1: pretrain on Shakespeare (~3-4 min on CPU)
python finetune_finance.py  # stage 2: fine-tune on financial Q&A (~1 min on CPU)
python chat.py               # ask it your own finance questions interactively
```

## Design notes

- **Why character-level tokenization?** Keeps the entire pipeline (vocab, embeddings, generation) simple enough to implement and reason about from scratch, at the cost of longer sequences than a subword tokenizer would need.
- **Why a shared `model.py`?** Both training stages must produce byte-identical architectures for a checkpoint saved in stage 1 to load correctly in stage 2 — factoring the model out once removes an entire class of bugs from keeping two copies in sync.
- **Why measure accuracy, not just read generated samples?** Generated text is easy to eyeball-judge optimistically. A held-out, teacher-forced accuracy number is the same style of metric used to evaluate real language models, and gives an honest, reproducible before/after comparison.
- **CPU-only by design.** All hyperparameters were benchmarked on an 8-core CPU and deliberately kept in a range that trains in minutes; the original lecture's GPU-sized configuration was benchmarked at ~23 hours on the same hardware and intentionally not used.

## Limitations

This is a from-scratch architecture and training pipeline demonstration, not a production financial assistant. At ~210K parameters and ~100 fine-tuning examples, the model learns the `Q:`/`A:` format and financial vocabulary well enough to show a clear, measurable improvement from fine-tuning, but it does not reliably produce factually accurate answers, especially for questions phrased differently from the training examples. Scaling data, parameters, and vocabulary (subword tokenization) would be the next steps toward higher answer quality.

## Credits

Built while following Andrej Karpathy's [Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=kCc8FmEb1nY) lecture and the [ng-video-lecture](https://github.com/karpathy/ng-video-lecture) repository, then extended with a checkpointing pipeline and financial-domain fine-tuning stage.
