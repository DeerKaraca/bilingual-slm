"""Export one of our checkpoints as an HF Llama model directory (safetensors), loadable with
transformers' AutoModelForCausalLM and with our server.

    python -m slm.export --checkpoint runs/tr050/final.pt --tokenizer assets/tokenizer.json --out export/tr050

Our architecture is Llama with full multi-head attention (no GQA): the only work is splitting our
fused qkv and gate_up matrices back into Llama's separate projections.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from slm.config import ModelConfig
from slm.special_tokens import EOT


def hf_config(cfg: ModelConfig, eos_id: int) -> dict:
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.d_model,
        "intermediate_size": cfg.ffn_hidden,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.kv_heads,
        "head_dim": cfg.head_dim,
        "max_position_embeddings": cfg.context_len,
        "rms_norm_eps": cfg.norm_eps,
        "rope_theta": cfg.rope_theta,
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": cfg.tie_embeddings,
        "bos_token_id": eos_id,
        "eos_token_id": eos_id,
        "torch_dtype": "float32",
    }


def state_dict_to_hf(sd: dict[str, torch.Tensor], cfg: ModelConfig) -> dict[str, torch.Tensor]:
    """Inverse of slm.server.loading.state_dict_from_hf. Tied lm_head weights are stored once."""
    q, kv = cfg.n_heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim
    out = {"model.embed_tokens.weight": sd["embed.weight"], "model.norm.weight": sd["norm.weight"]}
    for i in range(cfg.n_layers):
        ours, hf = f"blocks.{i}.", f"model.layers.{i}."
        wq, wk, wv = sd[ours + "attn.qkv.weight"].split([q, kv, kv])
        gate, up = sd[ours + "mlp.gate_up.weight"].chunk(2)
        out |= {
            hf + "input_layernorm.weight": sd[ours + "attn_norm.weight"],
            hf + "self_attn.q_proj.weight": wq,
            hf + "self_attn.k_proj.weight": wk,
            hf + "self_attn.v_proj.weight": wv,
            hf + "self_attn.o_proj.weight": sd[ours + "attn.out.weight"],
            hf + "post_attention_layernorm.weight": sd[ours + "mlp_norm.weight"],
            hf + "mlp.gate_proj.weight": gate,
            hf + "mlp.up_proj.weight": up,
            hf + "mlp.down_proj.weight": sd[ours + "mlp.down.weight"],
        }
    if not cfg.tie_embeddings:
        out["lm_head.weight"] = sd["lm_head.weight"]
    return {k: v.contiguous() for k, v in out.items()}  # safetensors needs contiguous, unshared tensors


def export(checkpoint: Path, tokenizer: Path, out: Path):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**state["config"]["model"])
    vocab = json.loads(tokenizer.read_text())
    eos_id = next(t["id"] for t in vocab["added_tokens"] if t["content"] == EOT)
    out.mkdir(parents=True, exist_ok=True)
    save_file(state_dict_to_hf(state["model"], cfg), out / "model.safetensors", metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(hf_config(cfg, eos_id), indent=2))
    (out / "generation_config.json").write_text(json.dumps({"bos_token_id": eos_id, "eos_token_id": eos_id}, indent=2))
    shutil.copy(tokenizer, out / "tokenizer.json")
    # No chat_template: the model is not instruction-tuned, so our server refuses /v1/chat/completions.
    (out / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "bos_token": EOT, "eos_token": EOT,
        "model_max_length": cfg.context_len, "clean_up_tokenization_spaces": False,
    }, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, default=Path("assets/tokenizer.json"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    export(args.checkpoint, args.tokenizer, args.out)
    print(f"exported to {args.out}")


if __name__ == "__main__":
    main()
