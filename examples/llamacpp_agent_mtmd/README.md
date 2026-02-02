# llama.cpp Agent Multimodal Example

This example runs a simple AutoAgents agent using the `autoagents-llamacpp` backend with image input.

## Requirements

- `llama-cpp-2` build dependencies (clang, cmake, and a C/C++ toolchain)

## Run

Use `--features cuda` flag if CUDA is available. For multimodal, also enable `--features mtmd`

### Multimodal (MTMD)

```bash
cargo run -p llamacpp_agent_mtmd --release --features "cuda mtmd" -- --prompt "What doyou see?" --image ./examples/llamacpp_agent_mtmd/test_img.jpg
```
