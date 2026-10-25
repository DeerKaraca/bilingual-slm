"""Load a model for serving: an HF Llama-architecture model (e.g. SmolLM2) or our own checkpoint."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from tokenizers import Tokenizer

from slm.config import ModelConfig
from slm.model import GPT
from slm.special_tokens import EOT

HF_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "generation_config.json"]


@dataclass
class LoadedModel:
    model: GPT
    tokenizer: Tokenizer
    name: str
    eos_ids: set[int]
    chat_template: str | None  # None: no instruction tuning, so no chat endpoint


def config_from_hf(hf: dict, max_context: int) -> ModelConfig:
    """Translate an HF Llama config.json, rejecting variants our model does not implement."""
    head_dim = hf["hidden_size"] // hf["num_attention_heads"]
    # transformers >= 5 nests RoPE settings in rope_parameters; older files (e.g. SmolLM2) keep them top-level.
    rope = hf.get("rope_parameters") or {}
    unsupported = {
        "model_type": hf.get("model_type") != "llama",
        "hidden_act": hf.get("hidden_act", "silu") != "silu",
        "rope_scaling": hf.get("rope_scaling") is not None or rope.get("rope_type", "default") != "default",
        "rope_interleaved": hf.get("rope_interleaved", False),
        "attention_bias": hf.get("attention_bias", False),
        "mlp_bias": hf.get("mlp_bias", False),
        "head_dim": hf.get("head_dim", head_dim) != head_dim,
    }
    if bad := [k for k, v in unsupported.items() if v]:
        raise ValueError(f"unsupported model config: {', '.join(bad)}")
    return ModelConfig(
        vocab_size=hf["vocab_size"],
        context_len=min(hf["max_position_embeddings"], max_context),
        n_layers=hf["num_hidden_layers"],
        d_model=hf["hidden_size"],
        n_heads=hf["num_attention_heads"],
        n_kv_heads=hf.get("num_key_value_heads", hf["num_attention_heads"]),
        ffn_hidden=hf["intermediate_size"],
        rope_theta=rope.get("rope_theta", hf.get("rope_theta", 10_000.0)),
        norm_eps=hf.get("rms_norm_eps", 1e-6),
        tie_embeddings=hf.get("tie_word_embeddings", False),
    )


def state_dict_from_hf(w: dict[str, torch.Tensor], cfg: ModelConfig) -> dict[str, torch.Tensor]:
    """Rename HF Llama weights to ours, fusing q/k/v and gate/up projections into single matrices."""
    sd = {"embed.weight": w["model.embed_tokens.weight"], "norm.weight": w["model.norm.weight"]}
    for i in range(cfg.n_layers):
        hf, ours = f"model.layers.{i}.", f"blocks.{i}."
        attn, mlp = hf + "self_attn.", hf + "mlp."
        sd[ours + "attn_norm.weight"] = w[hf + "input_layernorm.weight"]
        sd[ours + "attn.qkv.weight"] = torch.cat([w[attn + f"{p}_proj.weight"] for p in "qkv"])
        sd[ours + "attn.out.weight"] = w[attn + "o_proj.weight"]
        sd[ours + "mlp_norm.weight"] = w[hf + "post_attention_layernorm.weight"]
        sd[ours + "mlp.gate_up.weight"] = torch.cat([w[mlp + "gate_proj.weight"], w[mlp + "up_proj.weight"]])
        sd[ours + "mlp.down.weight"] = w[mlp + "down_proj.weight"]
    sd["lm_head.weight"] = sd["embed.weight"] if cfg.tie_embeddings else w["lm_head.weight"]
    return sd


def load_hf_model(path: Path, max_context: int) -> GPT:
    cfg = config_from_hf(json.loads((path / "config.json").read_text()), max_context)
    model = GPT(cfg)
    model.load_state_dict(state_dict_from_hf(load_file(path / "model.safetensors"), cfg))
    return model


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def load_hf(repo_or_dir: str, max_context: int) -> LoadedModel:
    path = Path(repo_or_dir)
    if not path.is_dir():
        path = Path(snapshot_download(repo_or_dir, allow_patterns=HF_FILES))
    tok = Tokenizer.from_file(str(path / "tokenizer.json"))
    tok_cfg, gen_cfg = _read_json(path / "tokenizer_config.json"), _read_json(path / "generation_config.json")
    eos = gen_cfg.get("eos_token_id", [])
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    if tok_cfg.get("eos_token"):
        eos_ids.add(tok.token_to_id(tok_cfg["eos_token"]))
    return LoadedModel(load_hf_model(path, max_context), tok, repo_or_dir, eos_ids, tok_cfg.get("chat_template"))


def load_checkpoint(path: str, tokenizer_path: str) -> LoadedModel:
    """Our weights-only checkpoint (final.pt / step_*.pt). Base models only, so no chat template."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(ModelConfig(**state["config"]["model"]))
    model.load_state_dict(state["model"])
    tok = Tokenizer.from_file(tokenizer_path)
    return LoadedModel(model, tok, f"{state['config']['name']}-step{state['step']}", {tok.token_to_id(EOT)}, None)


def load_model(spec: str, tokenizer: str | None, max_context: int, device: torch.device, dtype: torch.dtype) -> LoadedModel:
    """spec: an HF repo id / local HF directory, or a .pt checkpoint (needs tokenizer)."""
    if spec.endswith(".pt"):
        if tokenizer is None:
            raise ValueError("a .pt checkpoint needs --tokenizer")
        loaded = load_checkpoint(spec, tokenizer)
    else:
        loaded = load_hf(spec, max_context)
    loaded.model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return loaded
