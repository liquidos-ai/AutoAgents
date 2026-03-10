# Python Examples (autoagents-llamacpp-py)

## Prerequisites

- Python 3.9+
- A local GGUF model file

## Install

```bash
uv venv --python=3.12
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -U pip maturin pytest pytest-asyncio pytest-cov
make python-bindings-build
```

## Configure

Edit `examples/llamacpp_agent.py` and choose one source:

- GGUF: `.model_path("/path/to/model.gguf")`
- HuggingFace: `.repo_id("unsloth/phi-4-GGUF")` with optional
  `.hf_filename("phi-4-Q4_K_M.gguf")` and `.mmproj_filename("mmproj.gguf")`

The example uses modular composition:
- `AgentBuilder(...).llm(...)`
- `.memory(SlidingWindowMemory(...))`
- `.executor(ReActExecutor(...))`

## Run

```bash
python examples/llamacpp_agent.py
```

## Check Build Acceleration

```python
from autoagents_llamacpp import backend_build_info
print(backend_build_info())
```
