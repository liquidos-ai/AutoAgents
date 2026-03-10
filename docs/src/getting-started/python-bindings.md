# Python Bindings

AutoAgents provides Python bindings built with [PyO3](https://pyo3.rs) and
[maturin](https://www.maturin.rs). The bindings are split into multiple
packages so users install only what they need:

| Package | What it provides |
|---------|-----------------|
| `autoagents-py` | Core API — cloud LLM providers (OpenAI, Anthropic, …) |
| `autoagents-guardrails-py` | Guardrails for wrapping Python `LLMProvider` instances |
| `autoagents-llamacpp-py` | Local llama.cpp CPU backend |
| `autoagents-llamacpp-cuda` | llama.cpp with CUDA acceleration |
| `autoagents-llamacpp-metal` | llama.cpp with Metal acceleration (macOS) |
| `autoagents-llamacpp-vulkan` | llama.cpp with Vulkan acceleration |
| `autoagents-mistral-rs-py` | Local mistral-rs CPU backend |
| `autoagents-mistral-rs-cuda` | mistral-rs with CUDA acceleration |
| `autoagents-mistral-rs-metal` | mistral-rs with Metal acceleration (macOS) |

Release wheels for all platforms are published to PyPI by CI on every version
tag. **For local development and testing, use the repository `Makefile`
targets below.**

---

## Architecture Contract

The stable Python API is intentionally narrow:

- Rust owns runtime, scheduling, streaming, concurrency, event transport,
  agent execution, and production memory behavior.
- Python owns agent specs, prompt/orchestration logic, hooks, and tool
  implementations.
- Python tools are first-class in the stable path.

Use the stable package surface for production:

- `BasicAgent`
- `ReActAgent`
- `SlidingWindowMemory`
- `AgentBuilder`
- `Agent` / `AgentHandle`
- runtime and event APIs

Extension paths are opt-in under `autoagents.experimental`:

- `ExperimentalAgentBuilder`
- `CustomExecutor`
- Python-backed memory adapters

Those APIs are experimental in the Python bindings because they introduce
Python-native execution or Python-native memory behavior. The project keeps
those paths out of the stable Python surface so Rust remains the production
owner of runtime, scheduling, concurrency, streaming, and memory lifecycle.

That does not mean the capability itself is immature. The equivalent
Rust-native implementations are the production-grade path. The experimental
label applies to the Python extension mechanism, not to the Rust architecture.

### Python tool execution semantics

- Rust owns tool dispatch and lifecycle.
- Sync Python tools run on Tokio's blocking pool.
- Async Python tools run through the shared PyO3 async bridge on the single
  shared Tokio runtime.
- Tool arguments are validated against the declared JSON schema before the
  Python callable is invoked.

---

## Prerequisites

| Tool | Minimum version | Install |
|------|----------------|---------|
| Python | 3.9 | [python.org](https://www.python.org/downloads/) |
| Rust | stable | `curl https://sh.rustup.rs -sSf \| sh` |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| maturin | 1.5 | `uv pip install maturin` |
| pytest + pytest-asyncio + pytest-cov | latest | `uv pip install pytest pytest-asyncio pytest-cov` |

---

## Local Development Build (Linux)

The recommended local workflow is the repository `Makefile`. It runs
`maturin develop`, installs the editable packages into the active virtual
environment, and cleans stale editable-install artifacts before every build so
local imports always resolve to freshly compiled extensions:

```bash
# From the repository root
uv venv --python=3.12
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -U pip maturin pytest pytest-asyncio pytest-cov

# Build and install: base + llamacpp CPU + mistralrs CPU
make python-bindings-build

# Also build CUDA variants (requires CUDA Toolkit 12.x on PATH)
make python-bindings-build-cuda

# Other local variants
make python-bindings-build-metal
make python-bindings-build-vulkan
```

### Manual per-package build

Use this only if you need low-level control over one package. The `make`
targets above are the supported day-to-day development workflow.

```bash
# 1. Base package (required first)
cd bindings/python/autoagents
uv pip install -U pip maturin pytest pytest-asyncio pytest-cov
maturin develop --release

# 2. llama.cpp CPU backend (optional)
cd bindings/python/autoagents-llamacpp
maturin develop --release

# 3. guardrails package (optional)
cd bindings/python/autoagents-guardrails
maturin develop --release

# 4. mistral-rs CPU backend (optional)
cd bindings/python/autoagents-mistralrs
maturin develop --release
```

### Verify the install

```bash
python -c "import autoagents; print('OK')"
```

---

## Running Tests Locally

Tests live in `bindings/python/autoagents/tests/`.

| File | Requires API key | What it tests |
|------|:---------------:|---------------|
| `test_unit.py` | No | Schema inference, `@tool` decorator, builder/output schema wiring, error types, `Topic` |

### Unit tests — no API key needed

Run from anywhere after `make python-bindings-build`:

```bash
PYTHONPATH=bindings/python/autoagents \
  pytest bindings/python/autoagents/tests/test_unit.py -v
```

Unit tests cover schema inference (`Optional`, `List`, `Dict`, `Literal`,
`Enum`, dataclass, Pydantic), structured output schema registration, and
exception hierarchy. All pass with no keys set.

## Protocol Events

Every agent run emits typed protocol events. Access them via the event stream
or from the `run()` result:

```python
import asyncio
from autoagents import (
    Agent,
    LLMBuilder,
    ToolCallCompleted,
    ToolCallRequested,
    TurnStarted,
    tool,
)
from autoagents.prebuilt import BasicAgent, SlidingWindowMemory

@tool(description="Add two numbers")
def add(a: float, b: float) -> float:
    return a + b

async def main():
    llm = LLMBuilder("openai").api_key("sk-...").model("gpt-4o-mini").build()
    executor = BasicAgent("calc", "Calculator").tools([add])
    agent = Agent(executor, llm=llm, memory=SlidingWindowMemory(window_size=20))

    # Option A: get events from run() result
    result = await agent.run("What is 2 + 3?")
    for event in result.get("events", []):
        if isinstance(event, TurnStarted):
            print(f"Turn {event.turn_number}/{event.max_turns}")
        elif isinstance(event, ToolCallRequested):
            print(f"Calling {event.tool_name}({event.arguments})")
        elif isinstance(event, ToolCallCompleted):
            print(f"Result: {event.result}")

    # Option B: async event stream (concurrent with run)
    async for event in agent.events():
        print(event)

asyncio.run(main())
```

Available event types (all importable from `autoagents`):

| Class | Key attributes |
|-------|---------------|
| `NewTask` | `actor_id`, `prompt`, `system_prompt` |
| `TaskStarted` | `sub_id`, `actor_id`, `actor_name`, `task_description` |
| `TaskComplete` | `sub_id`, `actor_id`, `actor_name`, `result` |
| `TaskError` | `sub_id`, `actor_id`, `error` |
| `SendMessage` | `actor_id`, `message` |
| `ToolCallRequested` | `sub_id`, `actor_id`, `id`, `tool_name`, `arguments` |
| `ToolCallCompleted` | `sub_id`, `actor_id`, `id`, `tool_name`, `result` |
| `ToolCallFailed` | `sub_id`, `actor_id`, `id`, `tool_name`, `error` |
| `TurnStarted` | `sub_id`, `actor_id`, `turn_number`, `max_turns` |
| `TurnCompleted` | `sub_id`, `actor_id`, `turn_number`, `final_turn` |
| `StreamChunk` | `sub_id`, `chunk` |
| `StreamToolCall` | `sub_id`, `tool_call` |
| `StreamComplete` | `sub_id` |

---

## Examples

```bash
# Cloud provider (requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-... python bindings/python/autoagents/examples/openai_agent.py
OPENAI_API_KEY=sk-... python bindings/python/autoagents/examples/actor_agent.py
OPENAI_API_KEY=sk-... python bindings/python/autoagents/examples/protocol_event_streaming.py
OPENAI_API_KEY=sk-... python bindings/python/autoagents/examples/custom_pipeline_layer.py
OPENAI_API_KEY=sk-... python bindings/python/autoagents-guardrails/examples/pipeline_guardrails.py

# llama.cpp (requires `make python-bindings-build`)
python bindings/python/autoagents-llamacpp/examples/llamacpp_agent.py

# mistral-rs (requires `make python-bindings-build`)
python bindings/python/autoagents-mistralrs/examples/mistral_rs_agent.py

# CUDA variants (requires `make python-bindings-build-cuda`)
python bindings/python/autoagents-llamacpp-cuda/examples/llamacpp_cuda_agent.py
python bindings/python/autoagents-mistralrs-cuda/examples/mistral_rs_cuda_agent.py
```

Experimental extension examples:

```bash
OPENAI_API_KEY=sk-... python bindings/python/autoagents/examples/custom_executor.py
OPENAI_API_KEY=sk-... python bindings/python/autoagents/examples/custom_memory.py
```

The actor example mirrors the Rust runtime model: create a `Runtime`, register
it with an `Environment`, build the actor against the runtime, listen on the
environment event stream, then publish a `Task` to the topic.

---

## Troubleshooting

**`ImportError: No module named 'autoagents'`**
The Rust extension has not been compiled yet. Run `make python-bindings-build`
from the repository root.

**`maturin develop` fails with linker errors on Linux**
Install build essentials: `sudo apt install build-essential pkg-config`

**CUDA build fails: `nvcc: command not found`**
CUDA Toolkit is not on `PATH`. Source the CUDA environment or add
`/usr/local/cuda/bin` to `PATH`.

**`block_in_place` panic on import / `register_runtime`**
Ensure you are using the bindings built from the current source. Older builds
may not include the fix. Rebuild with `make python-bindings-build`; the target
cleans stale editable-install binaries before compiling.
