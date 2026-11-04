"""Exploratory (not in the analysis plan): does the compute-matched comparison change when validation
documents that share a 13-gram with the repeated training windows are removed?

    python benchmarks/clean_val_eval.py --tokens-dir data/tokens --runs-dir runs

The x3 control sees its training windows ~3.3 times, so validation text overlapping them could favour
it. Every model is evaluated on CPU in float32 on the full Turkish validation set and on the subset of
documents free of overlap with the windows of both seeds, so all models share one clean subset.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from slm.config import ModelConfig
from slm.eval.bpb import evaluate_bpb, token_byte_lengths
from slm.eval.overlap import contaminated_documents
from slm.model import GPT
from slm.special_tokens import EOT
from tokenizers import Tokenizer

RUNS = ["tr030", "tr030_seed1", "tr_only_control_x3", "tr_only_control_x3_seed1", "tr_only_control"]


def load_gpt(path: Path) -> GPT:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(ModelConfig(**state["config"]["model"]))
    model.load_state_dict(state["model"])
    return model.eval().requires_grad_(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens-dir", type=Path, default=Path("data/tokens"))
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--tokenizer", type=Path, default=Path("assets/tokenizer.json"))
    p.add_argument("--out", type=Path, default=Path("reports/research/clean_val"))
    args = p.parse_args()

    tok = Tokenizer.from_file(str(args.tokenizer))
    eot = tok.token_to_id(EOT)
    val = np.fromfile(args.tokens_dir / "tr" / "val.bin", dtype=np.uint16)
    doc_start, contaminated = contaminated_documents(args.tokens_dir, "tr", eot, seeds=[0, 1], unique_sequences=175_718)
    doc_end = np.append(doc_start[1:], len(val))
    clean = np.concatenate([val[s:e] for s, e, c in zip(doc_start, doc_end, contaminated) if not c])
    summary = {"documents": int(len(doc_start)), "contaminated_documents": int(contaminated.sum()),
               "val_tokens": int(len(val)), "clean_tokens": int(len(clean))}
    print(json.dumps(summary), flush=True)

    byte_lengths = token_byte_lengths(tok)
    rows = {}
    for name in RUNS:
        model = load_gpt(args.runs_dir / name / "final.pt")
        rows[name] = {split: evaluate_bpb(model, tokens, byte_lengths, model.cfg.context_len, batch_size=8)
                      for split, tokens in (("full", val), ("clean", clean))}
        print(name, json.dumps(rows[name]), flush=True)

    mean = lambda a, b, split: (rows[a][split] + rows[b][split]) / 2  # noqa: E731
    diffs = {split: mean("tr_only_control_x3", "tr_only_control_x3_seed1", split) - mean("tr030", "tr030_seed1", split)
             for split in ("full", "clean")}
    lines = ["# Exploratory: compute-matched comparison on overlap-free validation documents", "",
             "Not part of the analysis plan. Validation documents sharing any 13-gram with the repeated training "
             "windows of either seed are removed; every model is re-evaluated on CPU in float32.", "",
             f"Removed {summary['contaminated_documents']:,} of {summary['documents']:,} documents "
             f"({summary['val_tokens'] - summary['clean_tokens']:,} of {summary['val_tokens']:,} tokens).", "",
             "| Run | TR BPB, full validation | TR BPB, overlap-free documents |", "|---|---|---|"]
    lines += [f"| {name} | {r['full']:.4f} | {r['clean']:.4f} |" for name, r in rows.items()]
    lines += ["", "x3 control minus tr030 (seed means): "
                  f"full {diffs['full']:+.4f}, overlap-free {diffs['clean']:+.4f} TR BPB."]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps({**summary, "bpb": rows, "x3_minus_tr030": diffs}, indent=2))
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
