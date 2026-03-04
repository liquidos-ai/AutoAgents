# Safe Local Agent Example

This example shows a minimal local agent setup with an optimization pipeline in front of `LlamaCpp`.

## What It Demonstrates

- A single composed `LLMPipeline`: `Guardrails -> Cache -> LlamaCpp`
- Local model provider via:
  - `Qwen/Qwen3-VL-2B-Instruct-GGUF`
  - `Qwen3VL-2B-Instruct-Q8_0.gguf`
  - `mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf`
- A very small agent flow: read one prompt, run once, print response

## Safety + Optimization

- `RegexPiiRedactionGuard`: redacts common PII in input
- `PromptInjectionGuard`: blocks prompt-injection style instructions
- `CacheLayer`: uses `UserPromptOnly` key mode so repeated user prompts can hit
  cache even in a multi-turn conversation

## Run

```bash
cargo run -p safe_local_agent
```

Then enter your prompt in the terminal.

## Notes

- First run may download model files from Hugging Face.
- Ensure you have `llama-cpp-2` build dependencies installed (`clang`, `cmake`, C/C++ toolchain).
