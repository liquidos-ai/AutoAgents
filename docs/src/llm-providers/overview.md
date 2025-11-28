# Overview
AutoAgents supports a wide range of LLM providers, allowing you to choose the best fit for your use case:

### Cloud Providers

| Provider         | Status |
| ---------------- | ------ |
| **OpenAI**       | ✅     |
| **OpenRouter**   | ✅     |
| **Anthropic**    | ✅     |
| **DeepSeek**     | ✅     |
| **xAI**          | ✅     |
| **Phind**        | ✅     |
| **Groq**         | ✅     |
| **Google**       | ✅     |
| **Azure OpenAI** | ✅     |

### Local Providers

| Provider       | Status               |
| -------------- | -------------------- |
| **Mistral-rs** | ⚠️ Under Development |
| **Burn**       | ⚠️ Experimental      |
| **Onnx**       | ⚠️ Experimental      |
| **Ollama**     | ✅                   |

_Provider support is actively expanding based on community needs._

## Using Providers

Providers are accessed via `LLMBuilder` and enabled via `autoagents` crate features. Choose only what you need (e.g., `openai`, `anthropic`, `ollama`).

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
autoagents = { version = "0.2.4", features = ["openai"] }
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

All LLM backends implement the unified `LLMProvider` trait; chat/completion/embedding/model listing are composed from sub‑traits. This keeps agents provider‑agnostic.
