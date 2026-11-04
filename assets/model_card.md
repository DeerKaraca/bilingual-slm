---
license: apache-2.0
language:
  - tr
  - en
datasets:
  - epfml/FineWeb2-HQ
  - HuggingFaceFW/fineweb-edu
library_name: transformers
pipeline_tag: text-generation
tags:
  - turkish
  - bilingual
  - from-scratch
  - small-language-model
---

# bilingual-slm-42m-tr050

A 42.5M-parameter Turkish–English **base** language model trained from scratch on 600M tokens,
half Turkish and half English. It is the `tr050` run of a research project on how the
Turkish/English ratio of the training mix affects each language:
**[github.com/DeerKaraca/bilingual-slm](https://github.com/DeerKaraca/bilingual-slm)**
(code, data pipeline, all experiments, analysis and a hand-written inference server).

> **Not instruction-tuned.** This is a plain next-token predictor: it continues text, it does not
> follow instructions or chat, and it has no chat template. It is a research artifact and a small
> model: expect fluent-looking but often wrong or incoherent text.

## Usage

With `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "DeerKaraca/bilingual-slm-42m-tr050"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo)

inputs = tokenizer("Türkiye'nin en kalabalık şehri", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=40, do_sample=True, temperature=0.7, top_p=0.9)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

With the project's own OpenAI-compatible server (KV cache, continuous batching, SSE streaming):

```bash
git clone https://github.com/DeerKaraca/bilingual-slm && cd bilingual-slm
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -e ".[server]"
python -m slm.server --model DeerKaraca/bilingual-slm-42m-tr050
curl localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"prompt": "The history of science is", "max_tokens": 40, "stream": true}'
```

## Model

Decoder-only transformer, Llama layout (stored as `LlamaForCausalLM`):

| | |
|---|---|
| Parameters | 42.5M (25.7M without the embedding table) |
| Layers / width / heads | 8 / 512 / 8 (full multi-head attention), head dim 64 |
| MLP | SwiGLU, hidden size 1408 |
| Normalization | RMSNorm (pre-norm), eps 1e-6 |
| Positions | RoPE, theta 10,000, context 1,024 tokens |
| Embeddings | tied input/output, vocabulary 32,768 |

Checked against the training code: the exported weights loaded in `transformers` reproduce the
original model's logits to within 6e-6.

## Tokenizer

One byte-level BPE tokenizer (32,768 tokens) trained on equal amounts of Turkish and English text,
shared by every run of the project. Its pre-tokenizer has no English contraction rules, so Turkish
suffixes after an apostrophe stay intact (`İstanbul'da` → `İstanbul` `'` `da`); digits are split
one by one. On the validation sets:

| Tokenizer | Language | Tokens / word | Bytes / token |
|---|---|---|---|
| this model (32k) | Turkish | 1.77 | 4.73 |
| this model (32k) | English | 1.42 | 4.35 |
| GPT-2 (50k) | Turkish | 3.78 | 2.21 |
| GPT-2 (50k) | English | 1.32 | 4.69 |

Special tokens: `<|endoftext|>` (id 0, document separator, used as BOS/EOS) and the reserved,
unused `<|im_start|>` / `<|im_end|>`.

## Training

| | |
|---|---|
| Data | 300M Turkish tokens from FineWeb2-HQ (`tur_Latn`) + 300M English tokens from FineWeb-Edu (`sample-10BT`), no window repeated |
| Cleaning | NFC normalization, length and letter-ratio filters, exact deduplication |
| Mixing | every sequence is one language; the 50/50 ratio is exact and spread evenly over training |
| Batch / steps | 262,144 tokens per step (sequences of 1,024), 2,288 steps |
| Optimizer | AdamW (0.9, 0.95), weight decay 0.1, gradient clipping 1.0 |
| Learning rate | 200 warmup steps to 1e-3, cosine decay to 1e-4 |
| Precision / hardware | fp16 mixed precision, one NVIDIA T4 (Kaggle), ~4.2 hours |
| Code | tag `train-v1` of the GitHub repository |

## Evaluation

Validation bits per byte (lower is better) on held-out Turkish and English text (10.5 MB each,
never used for training). Bits per byte, unlike perplexity, does not depend on the tokenizer, so
the two languages and other models can be compared.

All five mixtures of the project, same architecture, tokenizer and 600M training tokens:

| Turkish share of the training mix | Turkish BPB | English BPB |
|---|---|---|
| 0% (`tr000`, English only) | 2.733 | **1.118** |
| 10% (`tr010`) | 1.316 | 1.125 |
| 30% (`tr030`) | 1.216 | 1.147 |
| **50% (`tr050`, this model)** | 1.177 | 1.179 |
| 100% (`tr100`, Turkish only) | **1.120** | 1.746 |

This run is the one published because it is the most balanced: with half the data of each
language it stays 5.1% (Turkish) and 5.5% (English) behind the monolingual models.

<!-- TODO before publishing: one sentence on the main finding, updated with the result of the
compute-matched control (reports/analysis_plan_addendum.md). -->
See the repository for the full analysis, its pre-registered decision rules and its caveats.

## Limitations

- Small and undertrained by current standards: factual statements are unreliable, longer outputs
  drift and repeat.
- Trained on filtered web text and inherits its biases; no safety tuning or filtering of outputs.
- Only Turkish and English; quality filters differ between the two sources (FineWeb-Edu uses an
  educational-value classifier, FineWeb2-HQ an embedding-based quality classifier).
- Not suitable for any decision-making or user-facing application without further work.

## Data attribution

Trained on content from:

- **FineWeb2-HQ** ([epfml/FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ)), released under the
  [Open Data Commons Attribution License (ODC-By) v1.0](https://opendatacommons.org/licenses/by/1-0/) and subject
  to [Common Crawl's Terms of Use](https://commoncrawl.org/terms-of-use).
- **FineWeb-Edu** ([HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)), released
  under the [Open Data Commons Attribution License (ODC-By) v1.0](https://opendatacommons.org/licenses/by/1-0/) and
  subject to [Common Crawl's Terms of Use](https://commoncrawl.org/terms-of-use).

The model weights and code are released under the Apache License 2.0.
