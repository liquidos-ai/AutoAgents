<div align="center">
  <img src="assets/logo.png" alt="AutoAgents Logo" width="200" height="200">

# AutoAgents

**A production-grade multi-agent framework in Rust**

[![Crates.io](https://img.shields.io/crates/v/autoagents.svg)](https://crates.io/crates/autoagents)
[![Documentation](https://docs.rs/autoagents/badge.svg)](https://liquidos-ai.github.io/AutoAgents)
[![License](https://img.shields.io/crates/l/autoagents.svg)](https://github.com/liquidos-ai/AutoAgents#license)
[![Build Status](https://github.com/liquidos-ai/AutoAgents/workflows/coverage/badge.svg)](https://github.com/liquidos-ai/AutoAgents/actions)
[![codecov](https://codecov.io/gh/liquidos-ai/AutoAgents/graph/badge.svg)](https://codecov.io/gh/liquidos-ai/AutoAgents)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/liquidos-ai/AutoAgents)

[Documentation](https://liquidos-ai.github.io/AutoAgents/) | [Examples](examples/) | [Contributing](CONTRIBUTING.md)

<br />
<strong>Like this project?</strong> <a href="https://github.com/liquidos-ai/AutoAgents">Star us on GitHub</a>
</div>

---

## Overview

AutoAgents is a modular, multi-agent framework for building intelligent systems in Rust. It combines a type-safe agent
model with structured tool calling, configurable memory, and pluggable LLM backends. The architecture is designed for
performance, safety, and composability across server, edge.

---

## Key Features

- **Agent execution**: ReAct and basic executors, streaming responses, and structured outputs
- **Tooling**: Derive macros for tools and outputs, plus a sandboxed WASM runtime for tool execution
- **Memory**: Sliding window memory with extensible backends
- **LLM providers**: Cloud and local backends behind a unified interface
- **Multi-agent orchestration**: Typed pub/sub communication and environment management
- **Speech-Processing**: Local TTS and STT support
- **Observability**: OpenTelemetry tracing and metrics with pluggable exporters

---

## Supported LLM Providers

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
| **MiniMax**      | ✅     |

### Local Providers

| Provider       | Status |
| -------------- | ------ |
| **Ollama**     | ✅     |
| **Mistral-rs** | ✅     |
| **Llama-Cpp**  | ✅     |

### Experimental Providers

See https://github.com/liquidos-ai/AutoAgents-Experimental-Backends

| Provider | Status          |
| -------- | --------------- |
| **Burn** | ⚠️ Experimental |
| **Onnx** | ⚠️ Experimental |

Provider support is actively expanding based on community needs.

---

## Benchmarks

![Benchmark](./assets/Benchmark.png)

More info at [GitHub](https://github.com/liquidos-ai/autoagents-bench)

---

## Installation

### Prerequisites

- **Rust** (latest stable recommended)
- **Cargo** package manager
- **LeftHook** for Git hooks management

### Install LeftHook

macOS (Homebrew):

```bash
brew install lefthook
```

Linux/Windows (npm):

```bash
npm install -g lefthook
```

### Clone and Build

```bash
git clone https://github.com/liquidos-ai/AutoAgents.git
cd AutoAgents
lefthook install
cargo build --workspace --all-features
```

### Run Tests

```bash
cargo test --workspace --features default --exclude autoagents-burn --exclude autoagents-mistral-rs --exclude wasm_agent
```

---

## Quick Start

```rust
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::{ReActAgent, ReActAgentOutput};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentDeriveT, AgentOutputT, DirectAgent};
use autoagents::core::error::Error;
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents::llm::LLMProvider;
use autoagents::llm::backends::openai::OpenAI;
use autoagents::llm::builder::LLMBuilder;
use autoagents_derive::{agent, tool, AgentHooks, AgentOutput, ToolInput};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct AdditionArgs {
    #[input(description = "Left Operand for addition")]
    left: i64,
    #[input(description = "Right Operand for addition")]
    right: i64,
}

#[tool(
    name = "Addition",
    description = "Use this tool to Add two numbers",
    input = AdditionArgs,
)]
struct Addition {}

#[async_trait]
impl ToolRuntime for Addition {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        println!("execute tool: {:?}", args);
        let typed_args: AdditionArgs = serde_json::from_value(args)?;
        let result = typed_args.left + typed_args.right;
        Ok(result.into())
    }
}

#[derive(Debug, Serialize, Deserialize, AgentOutput)]
pub struct MathAgentOutput {
    #[output(description = "The addition result")]
    value: i64,
    #[output(description = "Explanation of the logic")]
    explanation: String,
    #[output(description = "If user asks other than math questions, use this to answer them.")]
    generic: Option<String>,
}

#[agent(
    name = "math_agent",
    description = "You are a Math agent",
    tools = [Addition],
    output = MathAgentOutput,
)]
#[derive(Default, Clone, AgentHooks)]
pub struct MathAgent {}

impl From<ReActAgentOutput> for MathAgentOutput {
    fn from(output: ReActAgentOutput) -> Self {
        let resp = output.response;
        if output.done && !resp.trim().is_empty() {
            if let Ok(value) = serde_json::from_str::<MathAgentOutput>(&resp) {
                return value;
            }
        }
        MathAgentOutput {
            value: 0,
            explanation: resp,
            generic: None,
        }
    }
}

pub async fn simple_agent(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent_handle = AgentBuilder::<_, DirectAgent>::new(ReActAgent::new(MathAgent {}))
        .llm(llm)
        .memory(sliding_window_memory)
        .build()
        .await?;

    let result = agent_handle.agent.run(Task::new("What is 1 + 1?")).await?;
    println!("Result: {:?}", result);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or("".into());

    let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(api_key)
        .model("gpt-4o")
        .max_tokens(512)
        .temperature(0.2)
        .build()
        .expect("Failed to build LLM");

    let _ = simple_agent(llm).await?;
    Ok(())
}
```

### AutoAgents CLI

AutoAgents CLI helps in running Agentic Workflows from YAML configurations and serves them over HTTP. You can check it out at https://github.com/liquidos-ai/AutoAgents-CLI.

---

## Examples

Explore the examples to get started quickly:

### [Basic](examples/basic/)

Demonstrates various examples like Simple Agent with Tools, Very Basic Agent, Edge Agent, Chaining, Actor Based Model,
Streaming and Adding Agent Hooks.

### [MCP Integration](examples/mcp/)

Demonstrates how to integrate AutoAgents with the Model Context Protocol (MCP).

### [Local Models](examples/mistral_rs)

Demonstrates how to integrate AutoAgents with the Mistral-rs for Local Models.

### [Design Patterns](examples/design_patterns/)

Demonstrates various design patterns like Chaining, Planning, Routing, Parallel and Reflection.

### [Providers](examples/providers/)

Contains examples demonstrating how to use different LLM providers with AutoAgents.

### [WASM Tool Execution](examples/wasm_runner/)

A simple agent which can run tools in WASM runtime.

### [Coding Agent](examples/coding_agent/)

A sophisticated ReAct-based coding agent with file manipulation capabilities.

### [Speech](examples/speech/)

Run AutoAgents Speech Example with realtime TTS and STT.

### [Android Local Agent](https://github.com/liquidos-ai/AutoAgents-Android-Example)

Example App that runs AutoAgents with Local models in Android using AutoAgents-llamacpp backend

---

## Components

AutoAgents is built with a modular architecture:

```
AutoAgents/
├── crates/
│   ├── autoagents/                # Main library entry point
│   ├── autoagents-core/           # Core agent framework
│   ├── autoagents-protocol/       # Shared protocol/event types
│   ├── autoagents-llm/            # LLM provider implementations
│   ├── autoagents-telemetry/      # OpenTelemetry integration
│   ├── autoagents-toolkit/        # Collection of ready-to-use tools
│   ├── autoagents-mistral-rs/     # LLM provider implementations using Mistral-rs
│   ├── autoagents-llamacpp/       # LLM provider implementation using LlamaCpp
│   ├── autoagents-speech/         # Speech model support for TTS and STT
│   ├── autoagents-qdrant/         # Qdrant vector store
│   └── autoagents-derive/         # Procedural macros
├── examples/                      # Example implementations
```

### Core Components

- **Agent**: The fundamental unit of intelligence
- **Environment**: Manages agent lifecycle and communication
- **Memory**: Configurable memory systems
- **Tools**: External capability integration
- **Executors**: Different reasoning patterns (ReAct, Chain-of-Thought)

---

## Development

### Running Tests

```bash
cargo test --workspace --features default --exclude autoagents-burn --exclude autoagents-mistral-rs --exclude wasm_agent

# Coverage (requires cargo-tarpaulin)
cargo install cargo-tarpaulin
cargo tarpaulin --all-features --out html
```

### Running Benchmarks

```bash
cargo bench -p autoagents-core --bench agent_runtime
```

### Git Hooks

This project uses LeftHook for Git hooks management. The hooks will automatically:

- Format code with `cargo fmt --check`
- Run linting with `cargo clippy -- -D warnings`
- Execute tests with `cargo test --all-features --workspace --exclude autoagents-burn`

### Contributing

We welcome contributions. Please see our [Contributing Guidelines](CONTRIBUTING.md)
and [Code of Conduct](CODE_OF_CONDUCT.md) for details.

---

## Documentation

- **[API Documentation](https://liquidos-ai.github.io/AutoAgents)**: Complete framework docs
- **[Examples](examples/)**: Practical implementation examples

---

## Community

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Community Q&A and ideas
- **Discord**: Join our Discord Community using https://discord.gg/zfAF9MkEtK

---

## Performance

AutoAgents is designed for high performance:

- **Memory Efficient**: Optimized memory usage with configurable backends
- **Concurrent**: Full async/await support with tokio
- **Scalable**: Horizontal scaling with multi-agent coordination
- **Type Safe**: Compile-time guarantees with Rust's type system

---

## License

AutoAgents is dual-licensed under:

- **MIT License** ([MIT_LICENSE](MIT_LICENSE))
- **Apache License 2.0** ([APACHE_LICENSE](APACHE_LICENSE))

You may choose either license for your use case.

---

## Acknowledgments

Built by the [Liquidos AI](https://liquidos.ai) team and wonderful community of researchers and engineers.

<a href="https://github.com/liquidos-ai/AutoAgents/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=liquidos-ai/AutoAgents" />
</a>

Special thanks to:

- The Rust community for the excellent ecosystem
- LLM providers for enabling high-quality model APIs
- All contributors who help improve AutoAgents

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=liquidos-ai/AutoAgents&type=Date)](https://www.star-history.com/#liquidos-ai/AutoAgents&Date)
