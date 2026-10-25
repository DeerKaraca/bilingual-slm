# bilingual-slm

A ~40M-parameter Turkish–English language model trained from scratch, and a hand-written
inference server for it (no vLLM / TGI).

**Research question:** how does the Turkish/English ratio in the pre-training mix affect
performance in each language? How much does English transfer to Turkish, and when does one
language start to hurt the other?

> 🚧 Work in progress. Results and benchmark numbers will be filled in as the stages complete.

## Plan

| Part | What |
|---|---|
| Data | FineWeb-2 (Turkish) + FineWeb-Edu (English), streamed, cleaned, tokenized into shards |
| Tokenizer | One shared 32k byte-level BPE for all runs; fertility reported per language |
| Model | Decoder-only transformer: RoPE, RMSNorm, SwiGLU, fp16 mixed precision |
| Experiments | 0 / 10 / 30 / 50 / 100 % Turkish at 600M tokens each, plus a Turkish-only control matched to the Turkish tokens of the 30 % run |
| Metric | Per-language validation **bits-per-byte** (comparable across languages, unlike perplexity) |
| Server | KV cache, continuous batching, SSE streaming, int8 weight-only quantization, OpenAI-compatible API, Docker, load tests |

## Repository layout

```
slm/            Python package (config, data, tokenizer, model, train, eval, server)
configs/        YAML configs; experiments/ holds one file per run
notebooks/      Kaggle notebooks (clone repo -> train with resume)
benchmarks/     Load tests and server benchmarks
reports/        Generated plots and research report
tests/          Unit tests (run on CPU)
```

## Development

Developed on Linux (WSL2) with Python 3.13, matching Kaggle.

```bash
uv venv
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[dev,data,server]"
.venv/bin/python -m pytest
```

## Pipeline

| Step | Where | Command / notebook |
|---|---|---|
| 1. Stream + clean text | Kaggle CPU | `notebooks/01_prepare_data.ipynb` (`python -m slm.data.prepare`) |
| 2. Tokenizer, fertility, token shards | Kaggle CPU | `notebooks/02_tokenize.ipynb` (`slm.tokenizer`, `slm.eval.fertility`, `slm.data.tokenize`) |
| 3. Train (two runs per session, auto-resume) | Kaggle T4 x2 | `notebooks/03_train.ipynb` (`python -m slm.train --config configs/experiments/<run>.yaml`) |

## Inference server

Written from scratch on PyTorch + FastAPI (no vLLM / TGI). It serves our checkpoints and any
Llama-architecture HF model; development uses [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct),
whose layout (RMSNorm, RoPE, SwiGLU, tied embeddings) matches ours plus grouped-query attention.

```bash
python -m slm.server --model HuggingFaceTB/SmolLM2-135M-Instruct   # or: --model runs/tr030/final.pt --tokenizer assets/tokenizer.json
curl localhost:8000/v1/chat/completions -H "Content-Type: application/json"   -d '{"messages": [{"role": "user", "content": "What is the capital of France?"}], "stream": true}'
```

- **Engine:** one thread owns the model and a preallocated slot-based KV cache; every step decodes one token for all
  active requests, each at its own position. Attention is cut at the longest active sequence, not at the cache size.
- **Modes** (`--mode`): `no_cache`, `sequential`, `static` batching, `continuous` batching (default).
- **API:** `/v1/completions`, `/v1/chat/completions` (ChatML via the model's own chat template; models without
  instruction tuning get a clear 400), `/v1/models`, `/health`. SSE streaming, `temperature`, `top_p`, `top_k`,
  `stop`, `seed`. Disconnected clients free their slot immediately.
- **Correctness:** `benchmarks/parity.py` loads SmolLM2-135M into our model and compares with `transformers`:
  max logit difference 5e-5, identical greedy output.

## Known limitations

- **Quality filters differ between languages.** English comes from FineWeb-Edu (an educational-value
  classifier), Turkish from FineWeb2-HQ (an embedding-based quality classifier). Both are
  "high-quality" subsets, but not filtered identically, so part of any cross-lingual effect may come
  from differences in data quality rather than language alone.
