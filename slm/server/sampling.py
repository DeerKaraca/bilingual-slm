from dataclasses import dataclass, field

import torch


@dataclass
class SamplingParams:
    max_tokens: int = 16
    temperature: float = 1.0  # 0 = greedy
    top_p: float = 1.0
    top_k: int = 0  # 0 = disabled
    stop: list[str] = field(default_factory=list)
    seed: int | None = None


def sample(logits: torch.Tensor, params: SamplingParams, generator: torch.Generator) -> int:
    """Pick the next token from one row of logits (V,)."""
    if params.temperature == 0:
        return int(logits.argmax())
    logits = logits.float() / params.temperature
    if params.top_k > 0:
        kth = torch.topk(logits, min(params.top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if params.top_p < 1.0:
        sorted_logits, order = logits.sort(descending=True)
        probs = sorted_logits.softmax(-1)
        # Drop tokens once the mass before them already exceeds top_p; the top token always stays.
        drop = probs.cumsum(-1) - probs > params.top_p
        logits = logits.scatter(0, order[drop], float("-inf"))
    return int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
