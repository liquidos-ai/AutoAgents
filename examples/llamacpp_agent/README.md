# llama.cpp Agent Example

This example runs a simple AutoAgents math agent using the `autoagents-llamacpp` backend.

## Requirements

- A GGUF model file for Llama 3 3B (e.g. `models/llama3-3b.gguf`)
- `llama-cpp-2` build dependencies (clang, cmake, and a C/C++ toolchain)

## Run

```bash
cargo run -p llamacpp_agent -- \
  --model-path /path/to/llama3-3b.gguf \
  --prompt "What is 42 + 8?" \
  --max-tokens 256 \
  --temperature 0.2
```

If your GGUF includes a chat template, it is used automatically. To override it:

```bash
cargo run -p llamacpp_agent -- \
  --model-path /path/to/llama3-3b.gguf \
  --chat-template llama3
```

Use `--features cuda` flag if CUDA is available
