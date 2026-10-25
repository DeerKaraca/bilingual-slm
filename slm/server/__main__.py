"""Run the inference server.

    python -m slm.server --model HuggingFaceTB/SmolLM2-135M-Instruct
    python -m slm.server --model runs/tr030/final.pt --tokenizer assets/tokenizer.json
"""

import argparse
import logging

import torch
import uvicorn

from slm.server.api import create_app
from slm.server.engine import MODES, Engine
from slm.server.loading import load_model
from slm.server.quant import VARIANTS, quantize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct", help="HF repo id, HF directory or .pt checkpoint")
    p.add_argument("--tokenizer", help="tokenizer.json, required for .pt checkpoints")
    p.add_argument("--mode", default="continuous", choices=MODES)
    p.add_argument("--max-batch", type=int, default=8)
    p.add_argument("--max-len", type=int, default=2048, help="max prompt + completion tokens per request")
    p.add_argument("--quant", default="none", choices=VARIANTS, help="see slm/server/quant.py")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, help="torch intra-op threads (default: torch's choice)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.threads:
        torch.set_num_threads(args.threads)

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32  # fp16 matmuls are slow on most CPUs
    loaded = load_model(args.model, args.tokenizer, args.max_len, device, dtype)
    loaded.model = quantize(loaded.model, args.quant)
    engine = Engine(loaded, args.mode, args.max_batch, args.max_len)
    engine.start()
    try:
        uvicorn.run(create_app(engine, loaded), host=args.host, port=args.port)
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
