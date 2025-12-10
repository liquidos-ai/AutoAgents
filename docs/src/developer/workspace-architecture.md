# Workspace Architecture

This page maps the crates (excluding the CLI and Serve crates) and how they compose so contributors can keep boundaries clear.

## Crate Layout
- `autoagents`: Facade crate that re-exports the public surface from `autoagents-core`, `autoagents-llm`, and derive macros; holds only feature wiring and logging initialization.
- `autoagents-core`: Agent engine (agent config, executors, memory, protocol/events, vector store traits, runtime abstractions). Uses `ractor` only on non-WASM targets; compiled out for wasm via `cfg`.
- `autoagents-llm`: Provider-agnostic LLM traits plus concrete backend implementations (OpenAI, Anthropic, Ollama, etc.) and the `LLMBuilder` to configure them. Purely networking + request/response normalization, no agent logic.
- `autoagents-derive`: Proc macros for `#[agent]`, `#[tool]`, and derive helpers (`AgentOutput`, `ToolInput`, `AgentHooks`) that generate glue code while keeping downstream code ergonomic.
- `autoagents-toolkit`: Shared, reusable tools and MCP helpers. Feature-gated (`filesystem`, `search`, `mcp`) so downstream crates only pull what they need.
- `autoagents-qdrant`: Vector store implementation backed by Qdrant. Implements the `VectorStoreIndex` trait from `autoagents-core` and depends on an embedding provider via `SharedEmbeddingProvider`.
- Inference crates (optional): `autoagents-onnx`, `autoagents-burn`, and `autoagents-mistral-rs` provide local/runtime-specific inference backends. They plug into the LLM traits but are isolated to keep the core light.
- `autoagents-test-utils`: Shared fixtures and helpers for integration tests across crates.
- `examples/*`: Runnable end-to-end examples that demonstrate wiring agents, executors, and providers; each example is its own crate to keep dependencies scoped.

## Layering and Dependencies
- Top-level dependency direction is `autoagents` → (`autoagents-core`, `autoagents-llm`, `autoagents-derive`).
- `autoagents-core` depends on `autoagents-llm` for message/LLM types but keeps provider-specific details out of the core execution logic.
- `autoagents-toolkit` and `autoagents-qdrant` depend on the core traits and optionally on `autoagents-llm`/providers for embeddings.
- Inference crates implement the LLM traits so they can be swapped with remote providers without changing agent code.
- Examples pull only the crates they exercise (e.g., `autoagents-qdrant` for vector store examples), which keeps build times predictable and dependencies modular.

## Agent/Runtime Flow (non-Serve/CLI)
1. Agent definition (usually via `#[agent]` from `autoagents-derive`) describes tools, output type, and hooks.
2. Executors in `autoagents-core` (Basic, ReAct, direct or actor-backed) drive the conversation loop, calling into:
   - Memory providers (sliding window, etc.) from `autoagents-core`.
   - Tools (from `autoagents-toolkit` or custom example-local tools).
   - LLM providers implementing the `LLMProvider` traits (`autoagents-llm` backends or local inference crates).
3. Optional vector store operations go through `VectorStoreIndex` (e.g., `autoagents-qdrant`).
4. On non-WASM targets, the actor runtime (`ractor`) manages multi-agent orchestration; on WASM targets those pieces are `cfg`-gated out.

## Modularity Guidelines
- Keep provider concerns inside `autoagents-llm` (or inference crates); avoid leaking HTTP/provider structs into `autoagents-core`.
- Add reusable tools to `autoagents-toolkit`; example-specific tools should stay within their example crate.
- Prefer feature flags on extension crates (`autoagents-toolkit`, `autoagents-llm`, inference crates) so downstream users can opt in without pulling heavy dependencies.
- When adding new storage or provider integrations, implement the existing traits (`VectorStoreIndex`, `EmbeddingProvider`, `LLMProvider`) to preserve swappability and testability.
