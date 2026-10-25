"""OpenAI-compatible HTTP API on top of the engine: /v1/completions, /v1/chat/completions, /v1/models."""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator

import jinja2
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from slm.server.engine import Engine, Event, Request
from slm.server.loading import LoadedModel
from slm.server.sampling import SamplingParams


class GenerationFields(BaseModel):
    model: str | None = None
    max_tokens: int = Field(16, ge=1)
    temperature: float = Field(1.0, ge=0, le=2)
    top_p: float = Field(1.0, gt=0, le=1)
    top_k: int = Field(0, ge=0)
    stop: str | list[str] | None = None
    stream: bool = False
    seed: int | None = None
    ignore_eos: bool = False  # vLLM-style extension, used by the load test

    def sampling(self) -> SamplingParams:
        stop = [self.stop] if isinstance(self.stop, str) else self.stop or []
        return SamplingParams(self.max_tokens, self.temperature, self.top_p, self.top_k, stop, self.seed, self.ignore_eos)


class CompletionRequest(GenerationFields):
    prompt: str


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(GenerationFields):
    messages: list[Message] = Field(min_length=1)


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "invalid_request_error"}}, status_code=status)


def sse(data: dict | str) -> str:
    return f"data: {data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(engine: Engine, loaded: LoadedModel) -> FastAPI:
    app = FastAPI(title="bilingual-slm inference server")
    tokenizer = loaded.tokenizer
    chat_template = jinja2.Environment().from_string(loaded.chat_template) if loaded.chat_template else None

    def submit(prompt_ids: list[int], params: SamplingParams) -> tuple[Request, asyncio.Queue]:
        loop = asyncio.get_running_loop()
        events: asyncio.Queue[Event] = asyncio.Queue()
        # The engine runs on its own thread and asyncio queues are not thread-safe,
        # so events are handed over through the event loop.
        req = Request(prompt_ids, params, on_event=lambda ev: loop.call_soon_threadsafe(events.put_nowait, ev))
        engine.submit(req)  # raises ValueError for invalid prompts
        return req, events

    async def until_finished(events: asyncio.Queue) -> AsyncIterator[Event]:
        while True:
            event = await events.get()
            yield event
            if event.finish_reason:
                return

    async def respond(body: GenerationFields, prompt_ids: list[int], chat: bool):
        try:
            req, events = submit(prompt_ids, body.sampling())
        except ValueError as e:
            return error(400, str(e))
        meta = {"id": f"cmpl-{uuid.uuid4().hex}", "created": int(time.time()), "model": loaded.name}

        def choice(text: str, finish: str | None, streaming: bool) -> dict:
            if not chat:
                return {"index": 0, "text": text, "logprobs": None, "finish_reason": finish}
            key = "delta" if streaming else "message"
            return {"index": 0, key: {"role": "assistant", "content": text}, "finish_reason": finish}

        if body.stream:
            async def chunks():
                try:
                    async for event in until_finished(events):
                        chunk = {**meta, "object": "chat.completion.chunk" if chat else "text_completion",
                                 "choices": [choice(event.text, event.finish_reason, True)]}
                        yield sse(chunk)
                    yield sse("[DONE]")
                finally:
                    req.cancelled = True  # client disconnected: the engine frees the slot (no-op once finished)

            return StreamingResponse(chunks(), media_type="text/event-stream")

        try:
            collected = [e async for e in until_finished(events)]
        finally:
            req.cancelled = True
        last = collected[-1]
        usage = {"prompt_tokens": last.prompt_tokens, "completion_tokens": last.completion_tokens,
                 "total_tokens": last.prompt_tokens + last.completion_tokens}
        text = "".join(e.text for e in collected)
        return {**meta, "object": "chat.completion" if chat else "text_completion",
                "choices": [choice(text, last.finish_reason, False)], "usage": usage}

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest):
        return await respond(body, tokenizer.encode(body.prompt).ids, chat=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatRequest):
        if chat_template is None:
            return error(400, f"model '{loaded.name}' has no chat template: it was not instruction-tuned. "
                              "Use /v1/completions with a plain-text prompt instead.")
        prompt = chat_template.render(messages=[m.model_dump() for m in body.messages], add_generation_prompt=True)
        return await respond(body, tokenizer.encode(prompt, add_special_tokens=False).ids, chat=True)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": loaded.name, "object": "model", "owned_by": "bilingual-slm"}]}

    @app.get("/health")
    async def health():
        return {"status": "ok", "mode": engine.mode, "active": len(engine.active), "waiting": engine.waiting.qsize()}

    return app

