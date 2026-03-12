# Python Examples (autoagents-llamacpp-cuda)

## Prerequisites

- Python 3.9+
- CUDA Toolkit and compatible NVIDIA driver
- A local GGUF model file or a downloadable GGUF model

## Install

```bash
uv venv --python=3.12
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -U pip maturin pytest pytest-asyncio pytest-cov
make python-bindings-build-cuda
```

## Run

```bash
python bindings/python/autoagents-llamacpp-cuda/examples/llamacpp_cuda_agent.py
```

## Package Import

```python
from autoagents_llamacpp_cuda import LlamaCppBuilder, backend_build_info
```
