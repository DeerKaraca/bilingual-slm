"""Check our model and engine against transformers' reference implementation on real HF weights.

    python benchmarks/parity.py --model HuggingFaceTB/SmolLM2-135M-Instruct

1. Logits of one forward pass (no cache).
2. Greedy decoding through the engine (continuous batching, KV cache) with several prompts of
   different lengths in flight at once: some finish early (their slots are compacted) and one
   waits and joins mid-flight. The reference is a plain full-recompute greedy loop in transformers.

For every step up to the first divergence, the engine's logits are compared with the reference's,
which measures the KV-cache and batched path, not only a single forward pass.

If a sequence diverges, the first differing position is reported with the reference model's gap
between its two best logits there. A gap below NEAR_TIE means two candidates were practically tied
and float rounding can pick either; anything larger is treated as a bug (exit code 1).
"""

import argparse
import sys

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

from slm.server.engine import Engine, Request
from slm.server.loading import HF_FILES, load_hf
from slm.server.sampling import SamplingParams

NEAR_TIE = 1e-3  # ~20x the largest logit difference measured between the two implementations
LOGIT_PROMPT = "The capital of Turkey is Ankara. Türkiye'nin başkenti Ankara'dır ve"
# Different lengths and languages; max_tokens is chosen so that two requests finish early.
ENGINE_PROMPTS = [
    ("The capital of France is", 4),
    ("Photosynthesis is the process by which green plants and some other organisms use sunlight to "
     "synthesize foods from carbon dioxide and water. In plants it generally involves the green pigment "
     "chlorophyll and generates oxygen as a by-product. The process", 32),
    ("Türkiye'nin en kalabalık şehri", 8),
    ("def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n", 32),
]


@torch.no_grad()
def reference_greedy(model, prompt_ids: list[int], n: int) -> tuple[list[int], list[torch.Tensor]]:
    """Full-recompute greedy decoding; returns the tokens and the last-position logits of every step."""
    ids, steps = torch.tensor([prompt_ids]), []
    for _ in range(n):
        logits = model(ids).logits[0, -1]
        steps.append(logits)
        ids = torch.cat([ids, logits.argmax().view(1, 1)], dim=1)
    return ids[0, len(prompt_ids):].tolist(), steps


class RecordingEngine(Engine):
    """Engine that keeps the logits each request was sampled from, step by step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logits: dict[int, list[torch.Tensor]] = {}

    def _accept(self, req, logits):
        self.logits.setdefault(id(req), []).append(logits.clone())
        super()._accept(req, logits)


def run_engine(loaded, prompts: list[list[int]], lengths: list[int], max_batch: int):
    """Submit everything at once and step the engine until idle. Returns each request's tokens and
    per-step logits, and whether slots were compacted and a request joined while others were running."""
    engine = RecordingEngine(loaded, "continuous", max_batch=max_batch)
    reqs = [Request(ids, SamplingParams(max_tokens=n, temperature=0, ignore_eos=True), lambda e: None)
            for ids, n in zip(prompts, lengths)]
    for req in reqs:
        engine.submit(req)
    slots, seen = {}, {"compacted": False, "joined_mid_flight": False}
    first_step = True
    while engine.step():
        for req in engine.active.values():
            if id(req) in slots and slots[id(req)] != req.slot:
                seen["compacted"] = True
            if id(req) not in slots and not first_step:
                seen["joined_mid_flight"] = True
            slots[id(req)] = req.slot
        first_step = False
    return [req.output_ids for req in reqs], [engine.logits[id(req)] for req in reqs], seen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    args = p.parse_args()

    path = snapshot_download(args.model, allow_patterns=HF_FILES)
    ours = load_hf(path, max_context=2048)
    ours.model.eval().requires_grad_(False)
    reference = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    tok = ours.tokenizer

    ids = torch.tensor([tok.encode(LOGIT_PROMPT).ids])
    with torch.no_grad():
        diff = (ours.model(ids) - reference(ids).logits).abs().max().item()
    print(f"1. logits, one forward pass: max |difference| {diff:.2e}")

    prompts = [tok.encode(text).ids for text, _ in ENGINE_PROMPTS]
    lengths = [n for _, n in ENGINE_PROMPTS]
    outputs, engine_logits, seen = run_engine(ours, prompts, lengths, max_batch=3)
    print(f"2. engine greedy, {len(prompts)} prompts ({', '.join(str(len(x)) for x in prompts)} tokens), "
          f"max batch 3: slots compacted {seen['compacted']}, request joined mid-flight {seen['joined_mid_flight']}")

    bugs, worst = 0, 0.0
    for i, (prompt, n, got, steps) in enumerate(zip(prompts, lengths, outputs, engine_logits)):
        expected, ref_logits = reference_greedy(reference, prompt, n)
        pos = next((k for k, (a, b) in enumerate(zip(got, expected)) if a != b), None)
        # Up to and including the first divergence both sides decode from the same context.
        compared = n if pos is None else pos + 1
        step_diff = max((steps[k] - ref_logits[k]).abs().max().item() for k in range(compared))
        worst = max(worst, step_diff)
        if pos is None:
            print(f"   prompt {i}: {n} tokens identical; max |logit difference| over {compared} steps {step_diff:.2e}")
            continue
        top2 = ref_logits[pos].topk(2).values
        gap = (top2[0] - top2[1]).item()
        kind = "near-tie (float rounding)" if gap < NEAR_TIE else "BUG"
        bugs += kind == "BUG"
        print(f"   prompt {i}: diverges at token {pos} of {n}; reference top-2 logit gap {gap:.2e} -> {kind}; "
              f"max |logit difference| over {compared} steps {step_diff:.2e}")
    print(f"   engine (KV cache, batched): max |logit difference| over all compared steps {worst:.2e}")
    print("result:", "OK" if bugs == 0 and seen["compacted"] and seen["joined_mid_flight"] else "FAILED")
    sys.exit(1 if bugs or not (seen["compacted"] and seen["joined_mid_flight"]) else 0)


if __name__ == "__main__":
    main()
