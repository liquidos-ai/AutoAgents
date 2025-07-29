# LLM Providers Overview

AutoAgents supports a wide range of Large Language Model (LLM) providers, allowing you to choose the best fit for your specific use case. This document provides an overview of the supported providers and how to use them.

## Supported Providers

AutoAgents currently supports the following LLM providers:

| Provider | Status | Models | Features |
|----------|--------|--------|----------|
| **OpenAI** | ✅ | GPT-4, GPT-3.5-turbo | Function calling, streaming, vision |
| **Anthropic** | ✅ | Claude 3.5 Sonnet, Claude 3 Opus/Haiku | Tool use, long context |
| **Ollama** | ✅ | Llama 3, Mistral, CodeLlama | Local inference, custom models |
| **Google** | ✅ | Gemini Pro, Gemini Flash | Multimodal, fast inference |
| **Groq** | ✅ | Llama 3, Mixtral | Ultra-fast inference |
| **DeepSeek** | ✅ | DeepSeek Coder, DeepSeek Chat | Code-specialized models |
| **xAI** | ✅ | Grok-1, Grok-2 | Real-time information |
| **Phind** | ✅ | CodeLlama variants | Developer-focused |
| **Azure OpenAI** | ✅ | GPT-4, GPT-3.5 | Enterprise features |

## Common Interface

All LLM providers implement the `LLMProvider` trait, providing a consistent interface:
