# Python Examples (autoagents-mistral-rs-cuda)

## Prerequisites

- Python 3.9+
- CUDA Toolkit and compatible NVIDIA driver
- Internet access for first model download from Hugging Face unless already cached

## Install

```bash
uv venv --python=3.12
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -U pip maturin pytest pytest-asyncio pytest-cov
make python-bindings-build-cuda
```

## Run

```bash
python bindings/python/autoagents-mistralrs-cuda/examples/mistral_rs_cuda_agent.py
```

## Package Import

```python
from autoagents_mistral_rs_cuda import MistralRsBuilder, backend_build_info
```
