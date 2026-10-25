FROM python:3.13-slim

# CPU-only PyTorch keeps the image small (a CUDA base image would be needed for GPU serving).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY pyproject.toml README.md ./
COPY slm ./slm
RUN pip install --no-cache-dir ".[server]"

RUN useradd --create-home app
USER app
# Model weights are not baked into the image: they are downloaded on first start into this
# directory, which docker-compose mounts as a volume so later starts are fast.
ENV HF_HOME=/home/app/.cache/huggingface
RUN mkdir -p $HF_HOME

EXPOSE 8000
ENTRYPOINT ["python", "-m", "slm.server", "--port", "8000"]
CMD ["--model", "HuggingFaceTB/SmolLM2-135M-Instruct", "--threads", "3"]
