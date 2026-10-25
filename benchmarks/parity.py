"""Check that our model reproduces transformers' reference implementation on real HF weights.

    python benchmarks/parity.py --model HuggingFaceTB/SmolLM2-135M-Instruct
"""

import argparse

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

from slm.server.loading import HF_FILES, load_hf

PROMPT = "The capital of Turkey is Ankara. Türkiye'nin başkenti Ankara'dır ve"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--new-tokens", type=int, default=20)
    args = p.parse_args()

    path = snapshot_download(args.model, allow_patterns=HF_FILES)
    ours = load_hf(path, max_context=2048)
    reference = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    ids = torch.tensor([ours.tokenizer.encode(PROMPT).ids])

    with torch.no_grad():
        expected, actual = reference(ids).logits, ours.model.eval()(ids)
        greedy_ref = reference.generate(ids, max_new_tokens=args.new_tokens, do_sample=False)[0, ids.shape[1]:]
        greedy_ours = ids
        for _ in range(args.new_tokens):  # plain full-recompute greedy decoding, independent of the engine
            greedy_ours = torch.cat([greedy_ours, ours.model(greedy_ours)[:, -1:].argmax(-1)], dim=1)
        greedy_ours = greedy_ours[0, ids.shape[1]:]

    print(f"max |logit diff|: {(actual - expected).abs().max().item():.2e}")
    print(f"next-token argmax agreement: {(actual.argmax(-1) == expected.argmax(-1)).float().mean().item():.1%}")
    print(f"greedy {args.new_tokens} tokens identical: {torch.equal(greedy_ref, greedy_ours)}")
    print("continuation:", repr(ours.tokenizer.decode(greedy_ours.tolist())))


if __name__ == "__main__":
    main()
