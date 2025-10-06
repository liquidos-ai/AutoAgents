# MistralRs Example

This example demonstrates how to use the `autoagents-mistral-rs` crate to run local LLM models with AutoAgents.

## Features

The example showcases four different use cases:

1. **Text Models** - Standard text generation with HuggingFace models
2. **Vision Models** - Image understanding with vision-capable models
3. **GGUF Models** - Quantized models loaded from local files
4. **Tool Calling** - Agent with tools for mathematical operations

## Running the Examples

### Text Model (Default)

```bash
cargo run --package mistral_rs --release -- --model-type text
```

Uses Phi-3.5-mini-instruct with 8-bit ISQ quantization.

### Vision Model

```bash
cargo run --package mistral_rs --release -- --model-type vision
```

Uses SmolVLM-Instruct for image understanding. Make sure you have `test_img.jpg` in the example directory.

### GGUF Model

```bash
cargo run --package mistral_rs --release -- --model-type gguf --model-dir models/phi-3.5
```

Loads a quantized GGUF model from local files. You'll need to download the model files first.

### Tool Calling Model

```bash
cargo run --package mistral_rs --release -- --model-type tools
```

Uses Mistral-7B-Instruct with ReAct executor and math tools (addition, multiplication). Demonstrates:

- Simple calculations (42 + 58)
- Multiplication (15 × 8)
- Multi-step calculations ((10 + 5) × 3)

## Command-Line Options

- `-t, --model-type <TYPE>` - Model type: text, vision, gguf, or tools (default: text)
- `-d, --model-dir <PATH>` - Directory for GGUF models (default: examples/mistral_rs/models/phi-3.5)
- `-r, --repo-id <REPO>` - Override the default HuggingFace repository
- `-q, --quant <LEVEL>` - GGUF quantization level (default: q4-k-m)
- `--max-tokens <NUM>` - Maximum tokens to generate (default: 1024)
- `--temperature <TEMP>` - Sampling temperature (default: 0.2)
- `--paged-attention` - Enable paged attention (not compatible with GGUF + CUDA)
- `--verbose` - Enable detailed mistral.rs logging

## Hardware Acceleration

Build with CUDA support:

```bash
cargo build --package mistral_rs --release --features cuda
```

Build with Metal support (macOS):

```bash
cargo build --package mistral_rs --release --features metal
```