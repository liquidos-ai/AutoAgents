# LLM Providers Overview

AutoAgents supports a wide range of Large Language Model (LLM) providers, allowing you to choose the best fit for your
specific use case. This document provides an overview of the supported providers and how to use them.

## Supported Providers

AutoAgents currently supports the following LLM providers:

| Provider              | Status |
|-----------------------|--------|
| **LiquidEdge (ONNX)** | ✅      |
| **OpenAI**            | ✅      |
| **Anthropic**         | ✅      |
| **Ollama**            | ✅      |
| **DeepSeek**          | ✅      |
| **xAI**               | ✅      |
| **Phind**             | ✅      |
| **Groq**              | ✅      |
| **Google**            | ✅      |
| **Azure OpenAI**      | ✅      |

## Common Interface

All LLM providers implement the `LLMProvider` trait, providing a consistent interface:
