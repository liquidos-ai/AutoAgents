# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Building and Testing
```bash
# Build entire workspace
cargo build --release

# Build with all features
cargo build --all-features

# Run tests
cargo test --all-features

# Run tests with coverage
cargo install cargo-tarpaulin
cargo tarpaulin --all-features --out html

# Run specific example
cargo run --package basic-example -- --usecase simple
cargo run --package coding_agent -- --usecase interactive
cargo run --package wasm-runner
```

### Code Quality (Automated via LeftHook)
```bash
# Format code
cargo fmt

# Run linter
cargo clippy --all-features --all-targets -- -D warnings

# Check compilation
cargo check --all-features --all-targets

# Generate documentation
cargo doc --all-features --no-deps

# Run pre-commit hooks manually
lefthook run pre-commit
```

### Running Single Tests
```bash
# Run specific test by name
cargo test test_name --all-features

# Run tests in specific crate
cargo test -p autoagents-core --all-features

# Run tests with output
cargo test test_name --all-features -- --nocapture
```

## Architecture Overview

AutoAgents is a multi-agent framework built in Rust using a modular workspace architecture:

### Core Components

**Agent System**: The framework uses a trait-based agent system where agents implement `AgentDeriveT` and can be enhanced with executors like `ReActExecutor`. Agents are built using `AgentBuilder` and can have structured outputs via `AgentOutputT`.

**Runtime Management**: The system uses a `Runtime` trait with implementations like `SingleThreadedRuntime`. Runtimes manage agent lifecycle, message passing, and event handling through pub/sub topics.

**Environment**: The `Environment` struct orchestrates multiple runtimes and provides the execution context. It manages the working directory and coordinates between different runtime instances.

**Memory Systems**: Pluggable memory providers implementing `MemoryProvider` trait, with `SlidingWindowMemory` as the default implementation for conversation history management.

**Tool System**: Tools implement the `ToolT` trait and can be executed in different contexts, including WebAssembly runtimes for sandboxed execution. Tools are integrated via derive macros.

**LLM Integration**: Provider-agnostic LLM support through the `llm` crate with backends for OpenAI, Anthropic, Ollama, and others. Uses feature flags for conditional compilation.

### Key Patterns

**Event-Driven Architecture**: Communication happens through `Event` types with `TaskResult` outcomes. The system uses tokio streams for async event handling.

**Builder Pattern**: Most components use builder patterns for configuration (e.g., `AgentBuilder`, `LLMBuilder`).

**Feature Gates**: LLM providers and capabilities are feature-gated (e.g., `openai`, `anthropic`, `wasm`, `full`).

**Derive Macros**: The `autoagents-derive` crate provides procedural macros for `#[agent]`, `#[tool]`, `#[AgentOutput]`, and `#[ToolInput]` to reduce boilerplate.

### Workspace Structure

- `crates/autoagents`: Main library entry point with re-exports
- `crates/core`: Core agent framework with runtime, memory, and tool systems  
- `crates/llm`: LLM provider implementations and chat/completion APIs
- `crates/derive`: Procedural macros for agents and tools
- `crates/liquid-edge`: Edge runtime for local LLM inference
- `examples/`: Reference implementations including basic agents, coding agents, and WASM tool execution

The framework supports both local (ONNX via liquid-edge) and remote LLM providers, with WASM-based tool execution for security and cross-platform compatibility.
