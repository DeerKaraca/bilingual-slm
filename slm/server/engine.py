"""Inference engine: one thread owns the model and the KV cache and advances every active request
by one token per step.

Modes (compared in the benchmarks):
  no_cache    one request at a time; the whole sequence is recomputed for every new token
  sequential  one request at a time, with the KV cache
  static      waiting requests form a batch; newcomers wait until the whole batch has finished
  continuous  waiting requests take a free slot at every step, so the batch never drains
"""

import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from slm.model import KVCache
from slm.server.loading import LoadedModel
from slm.server.sampling import SamplingParams, sample

MODES = ("no_cache", "sequential", "static", "continuous")
log = logging.getLogger(__name__)


@dataclass
class Event:
    text: str  # newly generated text since the previous event
    finish_reason: str | None = None  # "stop", "length" or "error" on the last event
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Request:
    prompt_ids: list[int]
    params: SamplingParams
    on_event: Callable[[Event], None]  # called on the engine thread
    cancelled: bool = False  # set by the client side, e.g. on disconnect
    output_ids: list[int] = field(default_factory=list)
    slot: int = -1
    sent: int = 0  # characters of the decoded output already emitted
    generator: torch.Generator | None = None


class Engine:
    def __init__(self, loaded: LoadedModel, mode: str = "continuous", max_batch: int = 8, max_len: int | None = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.model, self.tokenizer, self.eos_ids = loaded.model, loaded.tokenizer, loaded.eos_ids
        self.mode = mode
        cfg = self.model.cfg
        self.n_slots = max_batch if mode in ("static", "continuous") else 1
        self.max_len = min(max_len or cfg.context_len, cfg.context_len)
        weight = self.model.norm.weight  # float even in quantized models
        self.device = weight.device
        self.cache = None if mode == "no_cache" else KVCache(cfg, self.n_slots, self.max_len, self.device, weight.dtype)
        self.waiting: queue.Queue[Request] = queue.Queue()
        self.active: dict[int, Request] = {}  # slot -> request
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    def submit(self, req: Request):
        """Queue a request; safe to call from any thread."""
        n = len(req.prompt_ids)
        if not 0 < n < self.max_len:
            raise ValueError(f"prompt has {n} tokens; it must have between 1 and {self.max_len - 1}")
        req.generator = torch.Generator(device=self.device)
        if req.params.seed is None:
            req.generator.seed()
        else:
            req.generator.manual_seed(req.params.seed)
        self.waiting.put(req)

    def start(self):
        self._thread = threading.Thread(target=self._run, name="engine", daemon=True)
        self._thread.start()

    def stop(self):
        self._stopped.set()
        if self._thread:
            self._thread.join()

    def _run(self):
        while not self._stopped.is_set():
            try:
                self.step(timeout=0.05)
            except Exception:
                log.exception("engine step failed")
                for req in self.active.values():
                    req.on_event(Event("", "error"))
                self.active.clear()

    @torch.inference_mode()
    def step(self, timeout: float = 0.0) -> bool:
        """Admit waiting requests, then generate one token for every active request.

        When idle, waits up to `timeout` seconds for a request. Returns whether any work was done.
        """
        self._admit(timeout)
        if not self.active:
            return False
        self._decode()
        return True

    def _admit(self, timeout: float):
        if self.mode == "static" and self.active:
            return
        free = [s for s in range(self.n_slots) if s not in self.active]
        while free:
            try:
                block = timeout > 0 and not self.active
                req = self.waiting.get(timeout=timeout) if block else self.waiting.get_nowait()
            except queue.Empty:
                return
            if not req.cancelled:
                req.slot = free.pop(0)
                self._prefill(req)

    def _prefill(self, req: Request):
        ids = torch.tensor([req.prompt_ids], device=self.device)
        if self.cache is None:
            logits = self.model(ids)
        else:
            self.cache.lengths[req.slot] = 0
            logits = self.model(ids, self.cache.view([req.slot]))
        self.active[req.slot] = req
        self._accept(req, logits[0, -1])

    def _decode(self):
        reqs = list(self.active.values())
        if self.cache is None:  # no_cache runs one request: recompute prompt + everything generated so far
            seq = reqs[0].prompt_ids + reqs[0].output_ids
            logits = self.model(torch.tensor([seq], device=self.device))[:, -1]
        else:
            last = torch.tensor([[r.output_ids[-1]] for r in reqs], device=self.device)
            logits = self.model(last, self.cache.view([r.slot for r in reqs]))[:, -1]
        for req, row in zip(reqs, logits):
            self._accept(req, row)

    def _accept(self, req: Request, logits: torch.Tensor):
        """Sample the next token of req, emit the new text and free the slot if the request is done."""
        if req.cancelled:
            del self.active[req.slot]
            return
        token = sample(logits, req.params, req.generator)
        req.output_ids.append(token)
        is_eos = token in self.eos_ids and not req.params.ignore_eos
        reason = None
        if is_eos:
            reason = "stop"
        elif len(req.output_ids) >= req.params.max_tokens or len(req.prompt_ids) + len(req.output_ids) >= self.max_len:
            reason = "length"
        text, hit_stop = self._new_text(req, final=reason is not None, drop_last=is_eos)
        reason = "stop" if hit_stop else reason
        if text or reason:
            req.on_event(Event(text, reason, len(req.prompt_ids), len(req.output_ids)))
        if reason:
            del self.active[req.slot]

    def _new_text(self, req: Request, final: bool, drop_last: bool = False) -> tuple[str, bool]:
        """Text that can be emitted now, and whether a stop string was reached.

        Byte-level BPE can split one character over several tokens, so a trailing incomplete
        character ("�") is held back, as is a tail that could still grow into a stop string.
        """
        # The EOS token is counted in usage but its text is never emitted.
        text = self.tokenizer.decode(req.output_ids[:-1] if drop_last else req.output_ids)
        hits = [i for i in (text.find(s) for s in req.params.stop) if i >= 0]
        if hits:
            text = text[:min(hits)]
        elif not final:
            hold = max((len(s) for s in req.params.stop), default=1) - 1
            text = text.rstrip("�")[:len(text) - hold]
        new = text[req.sent:]
        req.sent = max(req.sent, len(text))
        return new, bool(hits)
