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
