# llama.cpp Agent Example

This example runs a simple AutoAgents math agent using the `autoagents-llamacpp` backend.

## Requirements

- `llama-cpp-2` build dependencies (clang, cmake, and a C/C++ toolchain)

## Run

Use `--features cuda` flag if CUDA is available

```bash
cargo run -p llamacpp_agent -- \
  --prompt "What is 42 + 8?" \
  --max-tokens 256 \
  --temperature 0.2
```

To show reasoning/thinking events from llama.cpp:

```bash
cargo run -p llamacpp_agent -- \
  --thinking \
  --prompt "/think What is (20 + 30) * 10?"
```

Note: in thinking mode, many models can consume most generation budget in reasoning.
If you see reasoning events but little/no text output, increase `--max-tokens` (e.g. `1024`+).
