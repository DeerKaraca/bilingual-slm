"""Checkpoints under runs/{name}/, mirrored to a private HF model repo so that training can resume
in a fresh Kaggle session.

Files per run: latest.pt (full training state), log.csv, and weights-only snapshots
step_XXXXXX.pt / final.pt for later analysis and serving.
"""

import os
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

RESUME_FILES = ["latest.pt", "log.csv"]


def save(path: Path, state: dict):
    # Write to a temp file first so an interrupted save never leaves a corrupt checkpoint.
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def pull(hub_repo: str, run_dir: Path):
    """Download resume files of this run from the Hub, if any exist."""
    for name in RESUME_FILES:
        try:
            hf_hub_download(hub_repo, f"{run_dir.name}/{name}", local_dir=run_dir.parent)
        except (EntryNotFoundError, RepositoryNotFoundError):
            pass


def push(hub_repo: str, run_dir: Path, names: list[str]):
    api = HfApi()
    api.create_repo(hub_repo, private=True, exist_ok=True)
    for name in names:
        api.upload_file(path_or_fileobj=run_dir / name, path_in_repo=f"{run_dir.name}/{name}", repo_id=hub_repo)
