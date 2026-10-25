"""Why does torch's quantize_dynamic hurt quality while weight-only int8 does not?

    python benchmarks/dynamic_quant_analysis.py --model HuggingFaceTB/SmolLM2-135M --text-dir data/text

1. Activation outliers: max |x| / median |x| at the input of each linear type. quantize_dynamic
   gives every activation matrix a single int8 scale, set by its largest value.
2. Which change recovers quality: per-channel weights (weight granularity) vs keeping mlp.down,
   the layer with the largest outliers, in float (activation quantization).
3. Batch dependence: the activation scale is shared by the whole batch, so a sequence's
   quality depends on what it is batched with.
"""

import argparse
import copy
import warnings
from pathlib import Path

import torch
from torch import nn
from torch.ao.quantization import default_dynamic_qconfig, per_channel_dynamic_qconfig, quantize_dynamic

from quantization import eval_windows, quality
from slm.eval.bpb import token_byte_lengths
from slm.server.loading import load_model

warnings.simplefilter("ignore")  # torch.ao.quantization deprecation notices


def activation_outliers(model, x) -> dict[str, list[float]]:
    ratios: dict[str, list[float]] = {}

    def hook(kind):
        def record(module, inputs, output):
            a = inputs[0].abs().flatten()
            ratios.setdefault(kind, []).append((a.max() / a.median()).item())
        return record

    handles = [m.register_forward_hook(hook(name.split(".")[-1])) for name, m in model.named_modules()
               if isinstance(m, nn.Linear) and name != "lm_head"]
    with torch.inference_mode():
        model(x)
    for h in handles:
        h.remove()
    return ratios


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--text-dir", type=Path, default=Path("data/text"))
    p.add_argument("--lang", default="en")
    p.add_argument("--eval-tokens", type=int, default=4096)
    args = p.parse_args()

    loaded = load_model(args.model, None, 2048, torch.device("cpu"), torch.float32)
    model, byte_lengths = loaded.model, token_byte_lengths(loaded.tokenizer)
    windows = eval_windows(loaded.tokenizer, args.text_dir, args.lang, args.eval_tokens, 512)

    print("## Activation outliers (max |x| / median |x| at linear inputs)\n")
    print("| Layer | median over layers | worst layer |\n|---|---|---|")
    for kind, r in activation_outliers(model, windows[:1, :-1]).items():
        print(f"| {kind} | {sorted(r)[len(r) // 2]:.0f} | {max(r):.0f} |")

    ref_bpb, ref = quality(model, windows, byte_lengths)
    linears = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n != "lm_head"]
    setups = {
        "per-tensor weights (default)": {n: default_dynamic_qconfig for n in linears},
        "per-channel weights": {n: per_channel_dynamic_qconfig for n in linears},
        "per-tensor, mlp.down kept in float": {n: default_dynamic_qconfig for n in linears if not n.endswith("down")},
    }
    print(f"\n## quantize_dynamic variants ({args.lang}, float32 BPB {ref_bpb:.4f})\n")
    print("| Setup | Eval batch | BPB | Top-1 agree |\n|---|---|---|---|")
    for name, spec in setups.items():
        q = quantize_dynamic(copy.deepcopy(model), spec, dtype=torch.qint8)
        for batch in (1, 4) if name.startswith("per-tensor weights") else (2,):
            bpb, preds = quality(q, windows, byte_lengths, batch_size=batch)
            print(f"| {name} | {batch} | {bpb:.4f} | {(preds == ref).float().mean().item():.1%} |", flush=True)


if __name__ == "__main__":
    main()
