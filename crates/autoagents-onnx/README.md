# Onnx-ort - High-Performance Edge Inference Runtime

---

### ⚠️ Experimental Feature Notice

**Disclaimer:** This project is currently in an **experimental** phase and is under active development. Features, APIs,
and internal logic are subject to change without notice. Stability and performance are not yet guaranteed.

We welcome feedback and contributions, but please use with caution in production environments. Expect breaking changes
and incomplete functionality as we iterate and improve the inference engine.

---

Onnx is a inference runtime designed specifically for edge computing environments. It provides
high-performance LLM inference with multiple backend support, comprehensive tokenization capabilities, and optimized
memory management with AutoAgents LLM Provider Support.

### Model Directory Structure

```
models/my-model/
├── model.onnx              # ONNX model file
├── tokenizer.json          # HuggingFace tokenizer
├── config.json             # Model configuration
├── tokenizer_config.json   # Tokenizer configuration
├── special_tokens_map.json # Special tokens mapping
└── chat_template.jinja     # Chat template (optional)
```
