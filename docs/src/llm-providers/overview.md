# Overview

AutoAgents supports a wide range of LLM providers, allowing you to choose the best fit for your use case:

### Cloud Providers

| Provider         | Status |
|------------------|--------|
| **OpenAI**       | ✅      |
| **OpenRouter**   | ✅      |
| **Anthropic**    | ✅      |
| **DeepSeek**     | ✅      |
| **xAI**          | ✅      |
| **Phind**        | ✅      |
| **Groq**         | ✅      |
| **Google**       | ✅      |
| **Azure OpenAI** | ✅      |

### Local Providers

| Provider       | Status               |
| -------------- | -------------------- |
| **Ollama**     | ✅                   |
| **Mistral-rs** | ⚠️ Under Development |
| **Llama-Cpp**  | ⚠️ Under Development |

### Experimental Providers
Checkout https://github.com/liquidos-ai/AutoAgents-Experimental-Backends
| Provider       | Status               |
| -------------- | -------------------- |
| **Burn**       | ⚠️ Experimental      |
| **Onnx**       | ⚠️ Experimental      |

_Provider support is actively expanding based on community needs._

## Using Providers

Providers are accessed via `LLMBuilder` and enabled via `autoagents` crate features. Choose only what you need (e.g.,
`openai`, `anthropic`, `ollama`).

```rust
use autoagents::llm::builder::LLMBuilder;
use autoagents::llm::backends::openai::OpenAI;
use std::sync::Arc;

let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
    .api_key(std::env::var("OPENAI_API_KEY")?)
    .model("gpt-4o")
    .build()?;
```

Local providers like Ollama:

```rust
use autoagents::llm::backends::ollama::Ollama;
let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
    .base_url("http://localhost:11434")
    .model("llama3.2:3b")
    .build()?;
```

## Feature Flags

Enable providers on the `autoagents` crate:

```toml
autoagents = { version = "0.3.0", features = ["openai"] }
```

Common API key environment variables:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OPENROUTER_API_KEY`
- `GROQ_API_KEY`
- `GOOGLE_API_KEY`
- `AZURE_OPENAI_API_KEY`
- `XAI_API_KEY`

## Architecture

All LLM backends implement the unified `LLMProvider` trait; chat/completion/embedding/model listing are composed from
sub‑traits. This keeps agents provider‑agnostic.

## Capability Snapshot

This snapshot reflects the current code paths in `autoagents-llm` and may vary by specific model or provider changes.

- OpenAI
    - Chat + Streaming: Yes
    - Tool Calls: Yes
    - Structured Output (JSON Schema): Yes
    - Embeddings: Yes
    - Notes: Some options vary by model; check provider docs.

- Anthropic (Claude)
    - Chat + Streaming: Yes
    - Tool Calls: Yes (Anthropic tool-use format)
    - Structured Output: Not standardized; return text + tool events
    - Embeddings: No

- Groq (OpenAI-compatible)
    - Chat + Streaming: Yes
    - Tool Calls: Yes
    - Structured Output: Yes
    - Embeddings: No (not implemented)

- OpenRouter (OpenAI-compatible)
    - Chat + Streaming: Yes
    - Tool Calls: Yes
    - Structured Output: Yes
    - Embeddings: No (not implemented)

For other providers (Azure OpenAI, Google, XAI, DeepSeek, Ollama), consult their module docs and service documentation;
support can vary by model and API.
