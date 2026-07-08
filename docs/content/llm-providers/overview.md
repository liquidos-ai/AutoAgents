# Overview

AutoAgents supports cloud and local LLM providers behind the same agent-facing
interfaces. Provider support can still vary by model and upstream API behavior,
so this matrix describes the current AutoAgents code paths.

### Cloud Providers

| Provider | Feature | Chat | Streaming | Tool Calls | Structured Output | Vision / Multimodal |
| --- | --- | --- | --- | --- | --- | --- |
| OpenAI | `openai` | Yes | Yes | Yes | Yes | Image URLs and inline images; PDFs are rejected with a typed error. |
| OpenRouter | `openrouter` | Yes | Yes | Yes | Yes | OpenAI-compatible image inputs; PDFs are rejected with a typed error. |
| Anthropic | `anthropic` | Yes | Yes | Yes | Yes | Images, image URLs, and PDFs use Anthropic content blocks. |
| DeepSeek | `deepseek` | Yes | Yes | Yes | Yes | OpenAI-compatible image inputs; PDFs are rejected with a typed error. |
| xAI | `xai` | Yes | Yes | No | Model-dependent | Text-only chat; multimodal input returns `LLMError::InvalidRequest`. |
| Phind | `phind` | Yes | No* | No | No | Text-only chat; multimodal input returns `LLMError::InvalidRequest`. |
| Groq | `groq` | Yes | Yes | Yes | Yes | OpenAI-compatible image inputs; PDFs are rejected with a typed error. |
| Google | `google` | Yes | Yes | Yes | Yes | Inline images and PDFs; image URLs are rejected with a typed error. |
| Azure OpenAI | `azure_openai` | Yes | No* | Yes | Yes | Image URLs; PDFs and raw inline images are rejected with typed errors. |
| MiniMax | `minimax` | Yes | Yes | Yes | No | OpenAI-compatible image inputs; PDFs are rejected with a typed error. |

### Local Providers

| Provider | Crate / Feature | Chat | Streaming | Tool Calls | Structured Output | Vision / Multimodal | Local Inference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ollama | `ollama` | Yes | Yes | Yes | Yes | Model-dependent | Yes, via Ollama server |
| Mistral-rs | `autoagents-mistral-rs` | Yes | Yes | Yes | Yes | Vision models supported | Yes, embedded runtime |
| Llama-Cpp | `autoagents-llamacpp` | Yes | Yes | Yes | Yes | Vision models supported with projector files | Yes, embedded runtime |

### Experimental Providers

Checkout https://github.com/liquidos-ai/AutoAgents-Experimental-Backends

| Provider | Status |
| --- | --- |
| Burn | Experimental |
| Onnx | Experimental |

_Provider support is actively expanding based on community needs._

\* Providers marked **No** for streaming use the default `ChatProvider::chat_stream` implementation, which returns `LLMError::Generic("Streaming not supported for this provider")` rather than panicking.

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

Unless you set `.timeout_seconds(...)`, providers apply a default HTTP timeout of **120 seconds** at the reqwest client level. This bounds the full request lifecycle, including reading a streaming response body. For long-running generations, increase the timeout explicitly:

```rust
LLMBuilder::<OpenAI>::new()
    .model("gpt-4o")
    .timeout_seconds(300)
    .build()?;
```

### Streaming timeout semantics

- The configured timeout applies from request start through completion of the HTTP body read.
- `RetryLayer` retries only the initial stream-establishment call; mid-stream chunk errors are not retried automatically.
- There is no separate idle/chunk-gap timeout in the default client configuration.

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
autoagents = { version = "0.4.0", features = ["openai"] }
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

For optimization layers (cache/retry/fallback), see [Optimization Pipelines](./optimization-pipelines.md).
