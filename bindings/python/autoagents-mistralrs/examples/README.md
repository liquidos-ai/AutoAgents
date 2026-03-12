# Python Examples (autoagents-mistral-rs-py)

## Prerequisites

- Python 3.9+
- Internet access for first model download from Hugging Face (unless model is already cached)

## Install

```bash
uv venv --python=3.12
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -U pip maturin pytest pytest-asyncio pytest-cov
make python-bindings-build
```

## Run

```bash
python examples/mistral_rs_agent.py
```

## Source Selection

- HuggingFace source (default): `.repo_id("microsoft/Phi-3.5-mini-instruct")`
  with optional `.revision("main")` and `.model_type("auto" | "text" | "vision")`
- GGUF source: `.model_dir("/models").gguf_files(["model.gguf"])`
  with optional `.tokenizer("repo-or-path")` and `.chat_template("/path/to/template")`

The examples use modular composition:
- `AgentBuilder(...).llm(...)`
- `.memory(SlidingWindowMemory(...))`
- `.executor(ReActExecutor(...))`

## Check Build Acceleration

```python
from autoagents_mistral_rs import backend_build_info
print(backend_build_info())
```
