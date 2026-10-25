"""Quantization benchmark: weight memory, speed and quality of every variant against float32 on CPU.

    python benchmarks/quantization.py --model HuggingFaceTB/SmolLM2-135M --text-dir data/text

Quality uses the same validation text for every variant: bits-per-byte, and how often the
next-token argmax agrees with the float32 model (top-1 agreement).
"""

import argparse
import copy
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from slm.data.prepare import read_texts
from slm.eval.bpb import token_byte_lengths
from slm.model import KVCache
from slm.server.loading import load_model
from slm.server.quant import VARIANTS, quantize, weight_bytes


def eval_windows(tok, text_dir: Path, lang: str, n_tokens: int, seq_len: int) -> torch.Tensor:
    """First n_tokens of the validation split as (N, seq_len + 1) windows, documents separated by EOS."""
    eos = tok.token_to_id("<|endoftext|>")
    ids = []
    for text in read_texts(text_dir / lang, "val"):
        ids += [eos, *tok.encode(text).ids]
        if len(ids) > n_tokens:
            break
    n = (min(len(ids), n_tokens) - 1) // seq_len
    idx = np.arange(n)[:, None] * seq_len + np.arange(seq_len + 1)
    return torch.tensor(np.array(ids)[idx])


@torch.inference_mode()
def quality(model, windows: torch.Tensor, byte_lengths: torch.Tensor, batch_size: int = 2):
    """BPB over the windows and the argmax prediction at every position (special-token targets excluded)."""
    nats, n_bytes, preds = 0.0, 0, []
    for batch in windows.split(batch_size):
        x, y = batch[:, :-1], batch[:, 1:]
        logits = model(x).float()
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none")
        y_bytes = byte_lengths[y.flatten()]
        nats += loss[y_bytes > 0].sum().item()
        n_bytes += y_bytes.sum().item()
        preds.append(logits.argmax(-1))
    return nats / math.log(2) / n_bytes, torch.cat(preds)


@torch.inference_mode()
def time_prefill(model, prompt_len: int) -> float:
    """Seconds to process a prompt of prompt_len tokens into an empty KV cache (after one warm-up run)."""
    ids = torch.randint(0, model.cfg.vocab_size, (1, prompt_len))
    for _ in range(2):
        cache = KVCache(model.cfg, 1, prompt_len)
        start = time.perf_counter()
        model(ids, cache.view([0]))
    return time.perf_counter() - start


@torch.inference_mode()
def decode_throughput(model, batch: int, new_tokens: int = 32, prompt_len: int = 32) -> float:
    """Generated tokens per second when decoding `batch` sequences together with the KV cache."""
    cache = KVCache(model.cfg, batch, prompt_len + new_tokens + 1)
    view = cache.view(list(range(batch)))
    logits = model(torch.randint(0, model.cfg.vocab_size, (batch, prompt_len)), view)
    model(logits[:, -1:].argmax(-1), cache.view(list(range(batch))))  # warm-up step
    start = time.perf_counter()
    for _ in range(new_tokens):
        logits = model(logits[:, -1:].argmax(-1), cache.view(list(range(batch))))
    return batch * new_tokens / (time.perf_counter() - start)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--tokenizer", help="for .pt checkpoints")
    p.add_argument("--text-dir", type=Path, default=Path("data/text"))
    p.add_argument("--langs", nargs="+", default=["tr", "en"])
    p.add_argument("--eval-tokens", type=int, default=32_768, help="per language")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--rounds", type=int, default=5, help="interleaved timing rounds")
    p.add_argument("--out", type=Path, default=Path("reports/quantization"))
    args = p.parse_args()

    loaded = load_model(args.model, args.tokenizer, 2048, torch.device("cpu"), torch.float32)
    tok, base = loaded.tokenizer, loaded.model
    byte_lengths = token_byte_lengths(tok)
    windows = {lang: eval_windows(tok, args.text_dir, lang, args.eval_tokens, args.seq_len) for lang in args.langs}

    models = {v: quantize(copy.deepcopy(base), v) for v in VARIANTS}
    # Laptop CPUs throttle as they heat up: measure in interleaved rounds and keep the median,
    # so no variant is penalized for running last.
    timings = {v: {"prefill_256_ms": [], "decode_tok_s_b1": [], "decode_tok_s_b8": []} for v in VARIANTS}
    for _ in range(args.rounds):
        for v, model in models.items():
            timings[v]["prefill_256_ms"].append(time_prefill(model, 256) * 1000)
            for batch in (1, 8):
                timings[v][f"decode_tok_s_b{batch}"].append(decode_throughput(model, batch))

    rows, reference = [], {}
    for v, model in models.items():
        row = {"variant": v, "weights_mb": weight_bytes(model) / 2**20}
        row |= {k: statistics.median(t) for k, t in timings[v].items()}
        for lang, w in windows.items():
            bpb, preds = quality(model, w, byte_lengths)
            reference.setdefault(lang, preds)  # the first variant is float32
            row[f"{lang}_bpb"] = bpb
            row[f"{lang}_top1_agreement"] = (preds == reference[lang]).float().mean().item()
        rows.append(row)
        print(json.dumps(row), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"model": loaded.name, "threads": torch.get_num_threads(), "eval_tokens_per_lang": args.eval_tokens,
            "seq_len": args.seq_len, "timing_rounds": args.rounds, "rows": rows}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    args.out.with_suffix(".md").write_text(to_markdown(meta))
    print(to_markdown(meta))


def to_markdown(meta: dict) -> str:
    langs = [k.removesuffix("_bpb") for k in meta["rows"][0] if k.endswith("_bpb")]
    head = ["Variant", "Weights (MB)", "Prefill 256 tok (ms)", "Decode tok/s B=1", "Decode tok/s B=8"]
    head += [f"{lang.upper()} BPB" for lang in langs] + [f"{lang.upper()} top-1 agree" for lang in langs]
    lines = [f"# Quantization benchmark: {meta['model']}", "",
             f"CPU, {meta['threads']} threads; speed = median of {meta['timing_rounds']} interleaved rounds; quality on {meta['eval_tokens_per_lang']:,} validation tokens per language "
             f"in windows of {meta['seq_len']}. Top-1 agreement = share of positions where the next-token argmax "
             "matches float32.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in meta["rows"]:
        cells = [r["variant"], f"{r['weights_mb']:.0f}", f"{r['prefill_256_ms']:.0f}",
                 f"{r['decode_tok_s_b1']:.1f}", f"{r['decode_tok_s_b8']:.1f}"]
        cells += [f"{r[f'{lang}_bpb']:.4f}" for lang in langs] + [f"{r[f'{lang}_top1_agreement']:.1%}" for lang in langs]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
