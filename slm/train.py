"""Training loop: gradient accumulation, fp16 autocast + GradScaler, warmup + cosine LR,
periodic per-language validation BPB and resumable checkpoints.

    python -m slm.train --config configs/experiments/tr030.yaml --set hub_repo=USER/bilingual-slm-runs
"""

import argparse
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tokenizers import Tokenizer

from slm import checkpoint
from slm.config import Config, load_config
from slm.data.loader import MixtureLoader, load_tokens
from slm.eval.bpb import evaluate_bpb, token_byte_lengths
from slm.model import GPT

LOG_FIELDS = ["step", "tokens", "lr", "train_loss", "tr_bpb", "en_bpb", "tokens_per_sec", "elapsed_min"]
SNAPSHOTS = (0.25, 0.5, 0.75)  # fractions of training at which weights are kept


def lr_at(step: int, cfg: Config) -> float:
    t = cfg.train
    if step < t.warmup_steps:
        return t.lr * (step + 1) / t.warmup_steps
    progress = (step - t.warmup_steps) / max(1, cfg.total_steps - t.warmup_steps)
    return t.min_lr + 0.5 * (t.lr - t.min_lr) * (1 + math.cos(math.pi * progress))


def make_optimizer(model: GPT, cfg: Config, device: torch.device):
    # Decay matrices and embeddings, not norm gains.
    params = list(model.parameters())
    groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg.train.weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.train.lr, betas=cfg.train.betas, fused=device.type == "cuda")


def use_compile(requested: bool, device: torch.device) -> bool:
    """torch.compile generates Triton kernels, which need compute capability >= 7.0 (T4 yes, P100 no)."""
    if requested and device.type == "cuda" and torch.cuda.get_device_capability(device) < (7, 0):
        print("torch.compile disabled: GPU compute capability < 7.0")
        return False
    return requested


def truncate_log(path: Path, step: int):
    """Drop rows logged after the checkpoint we resume from, so they are not duplicated."""
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if int(r["step"]) <= step]
    write_log(path, rows, mode="w")


def write_log(path: Path, rows: list[dict], mode: str = "a"):
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, LOG_FIELDS)
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def train(cfg: Config, device: torch.device):
    t = cfg.train
    cuda = device.type == "cuda"
    run_dir = Path(cfg.out_dir) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.csv"

    torch.manual_seed(t.seed)
    model = GPT(cfg.model).to(device)
    optimizer = make_optimizer(model, cfg, device)
    scaler = torch.amp.GradScaler(device.type, enabled=cuda)
    n_sequences = cfg.total_steps * t.grad_accum_steps * t.micro_batch_size
    loader = MixtureLoader(
        cfg.data.tokens_dir, cfg.data.tr_fraction, t.micro_batch_size, cfg.model.context_len, t.seed, n_sequences
    )
    val = {lang: load_tokens(cfg.data.tokens_dir, lang, "val") for lang in ("tr", "en")}
    byte_lengths = token_byte_lengths(Tokenizer.from_file(cfg.data.tokenizer))
    print(f"{cfg.name}: {model.num_params() / 1e6:.1f}M params "
          f"({model.num_params(non_embedding=True) / 1e6:.1f}M non-embedding), {cfg.total_steps} steps")

    step = 0
    if cfg.hub_repo:
        checkpoint.pull(cfg.hub_repo, run_dir)
    if (run_dir / "latest.pt").exists():
        state = checkpoint.load(run_dir / "latest.pt")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        loader.load_state_dict(state["loader"])
        step = state["step"]
        truncate_log(log_path, step)
        print(f"resumed from step {step}")
    else:
        write_log(log_path, [], mode="w")

    def save(names: list[str]):
        if cfg.hub_repo:
            checkpoint.push(cfg.hub_repo, run_dir, names)

    def save_weights(name: str):
        checkpoint.save(run_dir / name, {"model": model.state_dict(), "config": cfg.to_dict(), "step": step})
        save([name])

    fwd = torch.compile(model) if use_compile(t.compile, device) else model
    snapshot_steps = {int(cfg.total_steps * s) for s in SNAPSHOTS}
    start_time = last_save = window_start = time.time()
    window_steps = 0

    while step < cfg.total_steps:
        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        loss_sum = torch.zeros((), device=device)  # summed on device to avoid a sync per micro-batch
        for _ in range(t.grad_accum_steps):
            x, y = (b.to(device, non_blocking=True) for b in loader.next_batch())
            with torch.autocast(device.type, dtype=torch.float16, enabled=cuda):
                loss = F.cross_entropy(fwd(x).flatten(0, 1), y.flatten())
            scaler.scale(loss / t.grad_accum_steps).backward()
            loss_sum += loss.detach()
        scaler.unscale_(optimizer)  # so clipping sees true gradient norms
        torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        window_steps += 1

        if step % t.eval_every == 0 or step == cfg.total_steps:
            tokens_per_sec = window_steps * cfg.tokens_per_step / (time.time() - window_start)
            row = {
                "step": step,
                "tokens": step * cfg.tokens_per_step,
                "lr": f"{lr:.3e}",
                "train_loss": f"{loss_sum.item() / t.grad_accum_steps:.4f}",
                **{f"{lang}_bpb": f"{evaluate_bpb(fwd, tokens, byte_lengths, cfg.model.context_len, t.micro_batch_size):.4f}"
                   for lang, tokens in val.items()},
                "tokens_per_sec": f"{tokens_per_sec:.0f}",
                "elapsed_min": f"{(time.time() - start_time) / 60:.1f}",
            }
            write_log(log_path, [row])
            print(row, flush=True)
            window_start, window_steps = time.time(), 0

        if step in snapshot_steps:
            save_weights(f"step_{step:06d}.pt")
        if time.time() - last_save > t.checkpoint_every_minutes * 60 or step == cfg.total_steps:
            checkpoint.save(run_dir / "latest.pt", {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "loader": loader.state_dict(),
                "step": step,
                "config": cfg.to_dict(),
            })
            save(checkpoint.RESUME_FILES)
            last_save = time.time()

    save_weights("final.pt")


def parse_overrides(items: list[str]) -> dict:
    """["train.lr=3e-4", "hub_repo=user/runs"] -> {"train": {"lr": 0.0003}, "hub_repo": "user/runs"}"""
    out = {}
    for item in items:
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        if isinstance(value, str):  # YAML reads "3e-4" as a string
            try:
                value = float(value)
            except ValueError:
                pass
        *parents, leaf = key.split(".")
        node = out
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    train(load_config(args.config, parse_overrides(args.set)), torch.device(args.device))


if __name__ == "__main__":
    main()
