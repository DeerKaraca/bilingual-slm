"""Exploratory (not in the analysis plan): does the compute-matched comparison change when validation
documents that overlap the repeated training windows are removed?

    python benchmarks/clean_val_eval.py --tokens-dir data/tokens --runs-dir runs --device cuda

The x3 control sees its training windows ~3.3 times, so validation text overlapping them could favour
it. Each document gets its share of 13-grams that occur in the training windows of either seed
(slm.eval.overlap.document_overlap); every number below comes from that one computation. Each model
is run once over the full Turkish validation set, and the BPB of every subset is summed from the
same per-token losses, restricted to the documents the subset keeps.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from slm.config import ModelConfig
from slm.eval.bpb import token_byte_lengths
from slm.eval.overlap import document_overlap
from slm.model import GPT
from slm.special_tokens import EOT

RUNS = ["tr030", "tr030_seed1", "tr_only_control_x3", "tr_only_control_x3_seed1", "tr_only_control"]
# Documents kept by each subset, from their overlap share. 20% was picked after seeing the
# distribution; 50% shows whether the result depends on that choice.
SUBSETS = {
    "full validation": lambda share: share >= 0,
    "without documents >50% overlapping": lambda share: share <= 0.5,
    "without documents >20% overlapping": lambda share: share <= 0.2,
    "without any overlapping document": lambda share: share == 0,
}
BINS = [(0, 0), (0, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1)]


def load_gpt(path: Path, device: torch.device) -> GPT:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(ModelConfig(**state["config"]["model"]))
    model.load_state_dict(state["model"])
    return model.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def token_losses(model: GPT, tokens: np.ndarray, batch_size: int = 16) -> np.ndarray:
    """Cross-entropy (nats) of every target token, non-overlapping windows as in training evaluation.
    Entry i is the loss of predicting tokens[i + 1]; the tail shorter than one window gets NaN."""
    device = next(model.parameters()).device
    seq = model.cfg.context_len
    n_windows = (len(tokens) - 1) // seq
    out = np.full(len(tokens) - 1, np.nan)
    for first in range(0, n_windows, batch_size):
        idx = np.arange(first, min(first + batch_size, n_windows))[:, None] * seq + np.arange(seq + 1)
        batch = torch.from_numpy(tokens[idx].astype(np.int64)).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.float().flatten(0, 1), batch[:, 1:].flatten(), reduction="none")
        out[first * seq:(first + len(idx)) * seq] = loss.cpu().numpy()
    return out


def subset_bpb(losses: np.ndarray, target_doc: np.ndarray, target_bytes: np.ndarray, keep_doc: np.ndarray) -> float:
    """BPB over the targets that belong to kept documents; special tokens carry no bytes and are skipped."""
    m = keep_doc[target_doc] & (target_bytes > 0) & ~np.isnan(losses)
    return float(losses[m].sum() / math.log(2) / target_bytes[m].sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens-dir", type=Path, default=Path("data/tokens"))
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--tokenizer", type=Path, default=Path("assets/tokenizer.json"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=Path("reports/research/clean_val"))
    args = p.parse_args()

    tok = Tokenizer.from_file(str(args.tokenizer))
    val = np.fromfile(args.tokens_dir / "tr" / "val.bin", dtype=np.uint16)
    doc_start, share = document_overlap(args.tokens_dir, "tr", tok.token_to_id(EOT), seeds=[0, 1],
                                        unique_sequences=175_718)
    doc_len = np.diff(np.append(doc_start, len(val)))
    target_doc = np.searchsorted(doc_start, np.arange(1, len(val)), side="right") - 1  # document of tokens[i + 1]
    target_bytes = token_byte_lengths(tok).numpy()[val[1:]]

    bpb_table = {}
    for name in RUNS:
        losses = token_losses(load_gpt(args.runs_dir / name / "final.pt", torch.device(args.device)), val)
        bpb_table[name] = {subset: subset_bpb(losses, target_doc, target_bytes, keep(share))
                           for subset, keep in SUBSETS.items()}
        print(name, json.dumps(bpb_table[name]), flush=True)

    density = [{"share": [lo, hi], "documents": int(m.sum()), "val_tokens": int(doc_len[m].sum()),
                "mean_document_tokens": float(doc_len[m].mean()) if m.any() else 0.0}
               for lo, hi in BINS for m in [(share == 0) if hi == 0 else (share > lo) & (share <= hi)]]
    kept = {s: {"documents": int(keep(share).sum()), "tokens": int(doc_len[keep(share)].sum())} for s, keep in SUBSETS.items()}
    seed_mean = lambda a, b, s: (bpb_table[a][s] + bpb_table[b][s]) / 2  # noqa: E731
    x3_minus_tr030 = {s: seed_mean("tr_only_control_x3", "tr_only_control_x3_seed1", s) - seed_mean("tr030", "tr030_seed1", s)
                      for s in SUBSETS}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(
        {"density": density, "kept": kept, "bpb": bpb_table, "x3_minus_tr030": x3_minus_tr030}, indent=2))
    args.out.with_suffix(".md").write_text(to_markdown(density, kept, bpb_table, x3_minus_tr030, len(val)))
    print(args.out.with_suffix(".md").read_text())


def to_markdown(density, kept, bpb_table, diffs, n_tokens) -> str:
    lines = [
        "# Exploratory: the compute-matched comparison on validation documents without overlap", "",
        "Not part of the analysis plan; the pre-registered decision rule uses the full validation set "
        "(see results.md). Each Turkish validation document gets the share of its 13-grams that occur in the "
        "repeated training windows of either seed; subsets drop documents above a threshold. The 20% threshold "
        "was chosen after seeing the distribution below, so 50% is shown as well.", "",
        "The subsets are clean for every model listed: the loader reads Turkish windows in one seed-fixed order, "
        "so tr030 and tr030_seed1 saw exactly the x3 pools of their seeds and tr_only_control saw the first 175,616 "
        "of the seed-0 pool (English windows of the tr030 runs are not scanned).", "",
        "## Overlap per document", "",
        "| Share of the document's 13-grams in the training windows | Documents | Validation tokens | Mean document length |",
        "|---|---|---|---|",
    ]
    for d in density:
        lo, hi = d["share"]
        label = "none" if hi == 0 else f"{lo:.0%} < share <= {hi:.0%}"
        lines.append(f"| {label} | {d['documents']:,} | {d['val_tokens'] / n_tokens:.1%} | {d['mean_document_tokens']:,.0f} |")
    lines += ["", "Dropping every document with any overlap removes long documents disproportionately (a single shared "
                  "phrase is enough), so that subset also shifts the validation distribution towards short documents.", "",
              "## Turkish BPB per subset", "",
              "| Run | " + " | ".join(kept) + " |", "|---|" + "---|" * len(kept),
              "| documents kept | " + " | ".join(f"{k['documents']:,}" for k in kept.values()) + " |",
              "| tokens kept | " + " | ".join(f"{k['tokens'] / n_tokens:.0%}" for k in kept.values()) + " |"]
    for name, row in bpb_table.items():
        lines.append(f"| {name} | " + " | ".join(f"{v:.4f}" for v in row.values()) + " |")
    lines.append("| **x3 control minus tr030 (seed means)** | " + " | ".join(f"**{v:+.4f}**" for v in diffs.values()) + " |")
    lines += ["", "All values from one float16 evaluation pass per model; the full-validation column can differ from "
                  "the training logs in the last digit because of evaluation order and precision."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
