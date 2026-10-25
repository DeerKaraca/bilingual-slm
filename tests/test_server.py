import json

import pytest
import torch
from fastapi.testclient import TestClient

from slm.config import ModelConfig
from slm.model import GPT
from slm.server.api import create_app
from slm.server.engine import MODES, Engine, Request
from slm.server.loading import LoadedModel
from slm.server.sampling import SamplingParams
from slm.tokenizer import EOT, train_tokenizer

CORPUS = ["İstanbul'da hava çok güzel.", "The weather is nice today.", "Merhaba dünya, hello world!"] * 20
CHATML = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


@pytest.fixture(scope="module")
def loaded():
    tok = train_tokenizer(iter(CORPUS), vocab_size=300)
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=tok.get_vocab_size(), context_len=64, n_layers=2, d_model=32, n_heads=4,
                      n_kv_heads=2, ffn_hidden=64)
    model = GPT(cfg).eval().requires_grad_(False)
    return LoadedModel(model, tok, "tiny", {tok.token_to_id(EOT)}, chat_template=None)


def run(engine, prompts, params, log=None):
    """Submit all prompts, step the engine until idle; return (text, finish_reason) per prompt."""
    outputs = [{"text": "", "finish": None} for _ in prompts]

    def on_event(i):
        def handle(event):
            outputs[i]["text"] += event.text
            outputs[i]["finish"] = event.finish_reason
            if log is not None:
                log.append((i, event.finish_reason))
        return handle

    for i, (prompt, p) in enumerate(zip(prompts, params)):
        engine.submit(Request(engine.tokenizer.encode(prompt).ids, p, on_event(i)))
    while engine.step():
        pass
    return [(o["text"], o["finish"]) for o in outputs]


PROMPTS = ["İstanbul'da", "The weather", "Merhaba", "hello", "çok güzel"]
GREEDY = [SamplingParams(max_tokens=n, temperature=0) for n in (12, 3, 7, 10, 5)]


def test_all_modes_generate_the_same_greedy_text(loaded):
    results = {mode: run(Engine(loaded, mode, max_batch=2), PROMPTS, GREEDY) for mode in MODES}
    assert all(r == results["no_cache"] for r in results.values())
    assert all(finish == "length" for _, finish in results["continuous"])


def test_continuous_admits_while_static_waits(loaded):
    """With 2 slots: request 2 should start as soon as request 1 (short) ends; static waits for request 0 too."""
    params = [SamplingParams(max_tokens=n, temperature=0) for n in (10, 2, 3)]
    order = {}
    for mode in ("static", "continuous"):
        log = []
        run(Engine(loaded, mode, max_batch=2), PROMPTS[:3], params, log)
        first_of_2 = next(k for k, (i, _) in enumerate(log) if i == 2)
        end_of_0 = next(k for k, (i, f) in enumerate(log) if i == 0 and f)
        order[mode] = first_of_2 < end_of_0
    assert order == {"static": False, "continuous": True}


def test_stop_string_truncates(loaded):
    engine = Engine(loaded, "continuous")
    params = SamplingParams(max_tokens=30, seed=3)  # seeded sampling: varied text, same with or without stop
    [(full, _)] = run(engine, ["hello"], [params])
    # A 2-character stop string whose first occurrence is in the middle of the text.
    k = next(k for k in range(3, len(full) - 2) if full.find(full[k:k + 2]) == k)
    params.stop = [full[k:k + 2]]
    [(text, finish)] = run(engine, ["hello"], [params])
    assert finish == "stop" and text == full[:k]


def test_seeded_sampling_is_reproducible(loaded):
    params = [SamplingParams(max_tokens=10, temperature=1.0, top_p=0.9, top_k=50, seed=7)] * 2
    a, b = run(Engine(loaded, "continuous"), ["hello", "hello"], params)
    assert a == b


def test_cancel_frees_slot(loaded):
    engine = Engine(loaded, "continuous", max_batch=1)
    events = []
    req = Request(engine.tokenizer.encode("hello").ids, SamplingParams(max_tokens=30), events.append)
    engine.submit(req)
    engine.step()
    req.cancelled = True
    engine.step()
    assert not engine.active and len(events) <= 1


def test_rejects_too_long_prompt(loaded):
    engine = Engine(loaded, "continuous", max_len=8)
    with pytest.raises(ValueError):
        engine.submit(Request(list(range(8)), SamplingParams(), lambda e: None))


@pytest.fixture
def client(loaded, request):
    template = getattr(request, "param", None)
    model = LoadedModel(loaded.model, loaded.tokenizer, loaded.name, loaded.eos_ids, template)
    engine = Engine(model, "continuous", max_batch=4)
    engine.start()
    yield TestClient(create_app(engine, model))
    engine.stop()


def test_completion_and_streaming_agree(client):
    body = {"prompt": "hello", "max_tokens": 8, "temperature": 0}
    plain = client.post("/v1/completions", json=body).json()
    assert plain["object"] == "text_completion"
    assert plain["usage"]["completion_tokens"] == 8 and plain["choices"][0]["finish_reason"] == "length"

    with client.stream("POST", "/v1/completions", json={**body, "stream": True}) as r:
        lines = [line.removeprefix("data: ") for line in r.iter_lines() if line]
    assert lines[-1] == "[DONE]"
    chunks = [json.loads(line) for line in lines[:-1]]
    assert "".join(c["choices"][0]["text"] for c in chunks) == plain["choices"][0]["text"]
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_chat_needs_instruction_tuned_model(client):
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400 and "not instruction-tuned" in r.json()["error"]["message"]


@pytest.mark.parametrize("client", [CHATML], indirect=True)
def test_chat_with_template(client):
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
    message = r.json()["choices"][0]["message"]
    assert r.status_code == 200 and message["role"] == "assistant"


def test_bad_requests(client):
    assert client.post("/v1/completions", json={"prompt": "hi " * 100}).status_code == 400  # longer than context
    assert client.post("/v1/completions", json={"prompt": "hi", "temperature": 5}).status_code == 422
    assert client.get("/v1/models").json()["data"][0]["id"] == "tiny"
