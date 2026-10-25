"""Experiment configuration.

Defaults live in the dataclasses below; experiment YAML files only list what differs.
"""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 32_768
    context_len: int = 1024
    n_layers: int = 8
    d_model: int = 512
    n_heads: int = 8
    ffn_hidden: int = 1408  # ~8/3 * d_model (SwiGLU keeps params close to a 4x MLP), rounded to 64
    n_kv_heads: int | None = None  # None = n_heads; fewer = grouped-query attention (e.g. SmolLM2)
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_heads(self) -> int:
        return self.n_kv_heads or self.n_heads


@dataclass
class DataConfig:
    tr_fraction: float = 0.5  # share of Turkish tokens in the training mix
    tokens_dir: str = "data/tokens"  # {lang}/{split}.bin from slm.data.tokenize
    tokenizer: str = "assets/tokenizer.json"


@dataclass
class TrainConfig:
    total_tokens: int = 600_000_000
    micro_batch_size: int = 16  # sequences per forward pass
    grad_accum_steps: int = 16  # 16 * 16 * 1024 = 262k tokens per optimizer step
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 200
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    eval_every: int = 100  # optimizer steps
    checkpoint_every_minutes: float = 30.0
    seed: int = 0
    compile: bool = True  # torch.compile; ~1 min warm-up, faster steps on GPU


@dataclass
class Config:
    name: str = "default"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    hub_repo: str | None = None  # HF Hub repo for checkpoints; None disables uploads
    out_dir: str = "runs"

    @property
    def tokens_per_step(self) -> int:
        t = self.train
        return t.micro_batch_size * t.grad_accum_steps * self.model.context_len

    @property
    def total_steps(self) -> int:
        return self.train.total_tokens // self.tokens_per_step

    def to_dict(self) -> dict:
        return asdict(self)


def _update(obj, values: dict):
    """Recursively apply a dict of overrides to a dataclass, rejecting unknown keys."""
    known = {f.name: f for f in fields(obj)}
    for key, value in values.items():
        if key not in known:
            raise KeyError(f"Unknown config key '{key}' for {type(obj).__name__}")
        current = getattr(obj, key)
        if is_dataclass(current):
            _update(current, value)
        elif isinstance(current, tuple):
            setattr(obj, key, tuple(value))
        else:
            setattr(obj, key, value)


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    cfg = Config()
    _update(cfg, yaml.safe_load(Path(path).read_text()) or {})
    _update(cfg, overrides or {})
    if not 0.0 <= cfg.data.tr_fraction <= 1.0:
        raise ValueError("data.tr_fraction must be in [0, 1]")
    if cfg.model.d_model % cfg.model.n_heads or cfg.model.n_heads % cfg.model.kv_heads:
        raise ValueError("model.d_model must be divisible by n_heads, and n_heads by n_kv_heads")
    return cfg
