# Python Bindings

Repository: https://github.com/liquidos-ai/AutoAgents

The Python bindings exist to make it easy to explore new ideas quickly without
giving up the Rust core that powers AutoAgents.

Python stays focused on agent definition, orchestration, experimentation, and
callback glue. Rust owns the runtime, scheduling, streaming, concurrency, and
core execution model.

## Stable vs Experimental

The production path is:

- Rust-owned executors (`BasicAgent`, `ReActAgent`)
- Rust-owned memory (`SlidingWindowMemory` and future Rust-backed providers)
- Python tools
- Python hooks

Explicit extension paths live under `autoagents.experimental`:

- `ExperimentalAgentBuilder`
- `CustomExecutor`
- Python-backed memory adapters

Those extension APIs are experimental in the Python bindings specifically
because they rely on Python-native execution or Python-native memory behavior.
That boundary exists to keep production runtime ownership in Rust rather than
re-centering the architecture on Python callbacks.

The same categories of capability are production-grade in the Rust-native
implementation. What is experimental here is the Python extension path, not the
underlying AutoAgents model when implemented natively in Rust.

## Why This Matters

- Rapid iteration in Python is useful when you are prototyping agent flows,
  hooks, memory adapters, and pipeline layers.
- Performance-sensitive and safety-critical behavior still comes from the Rust
  implementation underneath.
- Moving a successful prototype to production Rust is easier because the Python
  layer is orchestration-oriented rather than a separate runtime.

## Practical Outcome

You can prototype in Python with:

- the same `LLMProvider` model
- the same pipeline composition model
- the same agent builder structure
- the same runtime concepts used by the Rust crates

When you decide to harden something for production, the path to Rust is much
smaller because the architecture is already aligned.

## Installation from PyPI

Install the base package:

```bash
pip install autoagents-py
```

Install with a local backend or guardrails using extras:

```bash
pip install "autoagents-py[llamacpp]"           # llama.cpp CPU
pip install "autoagents-py[llamacpp-cuda]"      # llama.cpp CUDA
pip install "autoagents-py[llamacpp-metal]"     # llama.cpp Metal (macOS)
pip install "autoagents-py[llamacpp-vulkan]"    # llama.cpp Vulkan
pip install "autoagents-py[mistralrs]"          # mistral-rs CPU
pip install "autoagents-py[mistralrs-cuda]"     # mistral-rs CUDA
pip install "autoagents-py[mistralrs-metal]"    # mistral-rs Metal (macOS)
pip install "autoagents-py[guardrails]"         # Guardrails
pip install "autoagents-py[llamacpp-cuda,guardrails]"  # Multiple extras
```

## Local Development

From the repository root:

```bash
uv venv --python=3.12
source .venv/bin/activate
uv pip install -U pip maturin pytest pytest-asyncio pytest-cov

make python-bindings-build
```

For local backend work:

```bash
make python-bindings-build-llamacpp-only
make python-bindings-build-mistralrs-only
```

Backend-specific distributables now live in dedicated package directories under
`bindings/python/` such as `autoagents-llamacpp-cuda` and
`autoagents-mistralrs-metal`, so local builds and release builds use the same
checked-in package metadata.

For full setup and usage details, see:

- `README.md`
- `docs/src/getting-started/python-bindings.md`
