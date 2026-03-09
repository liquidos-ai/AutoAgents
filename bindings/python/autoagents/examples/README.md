# Python Examples (autoagents-py)

## Prerequisites

- Python 3.9+
- `OPENAI_API_KEY` set for cloud examples

## Install

```bash
uv venv --python=3.12
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -U pip maturin pytest pytest-asyncio pytest-cov
make python-bindings-build
```

## Run

```bash
python examples/openai_agent.py
python examples/actor_agent.py
python examples/protocol_event_streaming.py
python examples/custom_pipeline_layer.py
```

Experimental extension examples:

```bash
python examples/custom_executor.py
python examples/custom_memory.py
```

## Streaming

Use `Agent.run_stream(task)` to stream output chunks while the run is in progress:

```python
async for chunk in agent.run_stream("Solve 15 * 9"):
    print(chunk)
```

## Hooks

`openai_agent.py` includes a `CustomHooks` implementation and demonstrates
how to dispatch typed protocol events (`TurnStarted`, `ToolCallRequested`, …)
from `result["events"]`.

## Custom Executor (Experimental)

`custom_executor.py` demonstrates custom execution with a plain Python callback
object:
- `CustomMathExecutor` implements `config/execute/execute_stream`.
- `autoagents.experimental.ExperimentalAgentBuilder(...)` keeps the extension path explicit.
- `autoagents.experimental.CustomExecutor(...)` injects that object into the Rust runtime.
- `ctx["llm"]` and `ctx["memory"]` are context-bound Rust runtime handles.

## Custom Memory (Experimental)

`custom_memory.py` demonstrates Python memory callbacks using a plain list.
Use `autoagents.experimental` to opt into Python-backed memory explicitly.

## Actor Runtime

`actor_agent.py` shows `build_actor(...)`, `Runtime`, `Environment`, `Topic`,
and environment event consumption.

## Protocol Event Stream

`protocol_event_streaming.py` consumes typed protocol events concurrently with
`handle.run(...)`.

## Custom Pipeline Layer

`custom_pipeline_layer.py` shows the Python pipeline API with
`PipelineBuilder.add_layer(...)` and a small Python `PipelineLayer`.
