# LLM Optimization Pipelines

AutoAgents provides a composable LLM pipeline layer in `autoagents-llm` to optimize inference latency, reliability, and cost without changing agent code.

This feature is available through:

- `autoagents::llm::pipeline::PipelineBuilder`
- `autoagents::llm::optim::{CacheLayer, RetryLayer, FallbackLayer}`

## Enable Feature Flag

Enable the `optim` feature on `autoagents`:

```toml
autoagents = { git = "https://github.com/liquidos-ai/AutoAgents", features = ["openai", "optim"] }
```

Or directly on `autoagents-llm`:

```toml
autoagents-llm = { git = "https://github.com/liquidos-ai/AutoAgents", features = ["optim"] }
```

## Why Pipelines

Pipelines let you keep your agent code provider-agnostic while adding operational behavior:

- Response caching to reduce repeated network calls
- Retry with backoff on transient errors
- Fallback routing to alternate providers on failure

The final built value is still an `Arc<dyn LLMProvider>`, so existing `AgentBuilder` code remains unchanged.

## Basic Composition

```rust
use autoagents::llm::LLMProvider;
use autoagents::llm::optim::{CacheConfig, CacheLayer, FallbackLayer, RetryConfig, RetryLayer};
use autoagents::llm::pipeline::PipelineBuilder;
use std::sync::Arc;
use std::time::Duration;

let llm: Arc<dyn LLMProvider> = PipelineBuilder::new(primary_provider)
    .add_layer(CacheLayer::new(CacheConfig {
        ttl: Some(Duration::from_secs(3600)),
        max_size: Some(1000),
        ..CacheConfig::default()
    }))
    .add_layer(RetryLayer::new(RetryConfig::default()))
    .add_layer(FallbackLayer::new(vec![fallback_provider]))
    .build();
```

## Layer Order

Layers are applied so that the first added layer is the outermost interceptor.

For:

```rust
PipelineBuilder::new(base)
    .add_layer(LayerA)
    .add_layer(LayerB)
    .build()
```

request flow is:

`LayerA -> LayerB -> base provider`

This is important for behavior:

- Place cache outside retry/fallback if you want cache hits to bypass all network logic.
- Place retry outside fallback if you want one global retry around the whole fallback chain.
- Place retry inside each fallback provider if you need per-provider retry policy.

## Built-in Optimization Layers

## CacheLayer

`CacheLayer` is an in-memory cache for chat, completion, embedding, and streaming responses.

`CacheConfig`:

- `ttl`: entry freshness duration
- `max_size`: per-cache-bucket maximum entries
- `cache_completions`: enable completion caching
- `cache_embeddings`: enable embedding caching
- `cache_streaming`: enable stream replay caching

Behavior notes:

- Non-streaming requests use single-flight on cache miss to coalesce identical concurrent calls.
- Streaming cache stores chunks only after a successful stream completion.
- `chat_with_web_search` is intentionally not cached.

## RetryLayer

`RetryLayer` adds automatic retry with exponential backoff and optional jitter.

`RetryConfig`:

- `max_attempts`
- `initial_backoff`
- `max_backoff`
- `jitter`
- `retryable` predicate

Default policy retries transient/provider/network-style failures and avoids retrying deterministic errors (for example auth/invalid-request style failures).

## FallbackLayer

`FallbackLayer` routes requests to backup providers when errors are fallbackable.

`FallbackConfig`:

- `fallbackable` predicate

Behavior notes:

- Providers are tried in declared order.
- Non-fallbackable errors stop the chain immediately.
- Fallback providers are used as passed to `FallbackLayer::new` (they are not automatically wrapped by other inner pipeline layers around the primary provider).

## Production Recommendations

- Set explicit `ttl` and `max_size` for predictable memory usage.
- Tune retry/backoff for your provider SLOs and rate limits.
- Keep fallback providers model-compatible with your prompt/tooling expectations.
- Use structured logging/telemetry around provider failures and fallback hops.

## Example Crate

A runnable end-to-end example is available at:

- `examples/pipeline`

Run it with:

```bash
OPENAI_API_KEY=... cargo run -p pipeline-example
```
