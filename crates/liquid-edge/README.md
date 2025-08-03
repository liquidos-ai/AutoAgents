# Liquid Edge - High-Performance Edge Inference Runtime

Liquid Edge is a production-ready inference runtime designed specifically for edge computing environments. It provides high-performance LLM inference with multiple backend support, comprehensive tokenization capabilities, and optimized memory management.

## 🚀 Features

- **🎯 Multiple Backends**: ONNX Runtime support with more backends planned
- **⚡ High Performance**: Optimized for edge devices with minimal latency
- **🔄 Async Support**: Full async/await support for non-blocking inference
- **💬 Chat Templates**: Jinja2 template support for conversational AI
- **📊 Comprehensive Monitoring**: Built-in metrics and performance tracking
- **🛡️ Production Ready**: Robust error handling and logging
- **🔧 Flexible Configuration**: Feature flags for customized builds


```sh
pipx install "optimum[onnxruntime]"
optimum-cli export onnx \                                                                                                                                               ✔ │ 7s   │ rust-onnx-chat 
      --model squeeze-ai-lab/TinyAgent-1.1B \
      --task text-generation ./models/tinyagen
```

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
