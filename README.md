<div align="center">
  <img src="assets/logo.png" alt="AutoAgents Logo" width="200" height="200">

  # AutoAgents

  **A Modern Multi-Agent Framework for Rust**

  [![Crates.io](https://img.shields.io/crates/v/autoagents.svg)](https://crates.io/crates/autoagents)
  [![Documentation](https://docs.rs/autoagents/badge.svg)](https://liquidos-ai.github.io/AutoAgents)
  [![License](https://img.shields.io/crates/l/autoagents.svg)](https://github.com/liquidos-ai/AutoAgents#license)
  [![Build Status](https://github.com/liquidos-ai/AutoAgents/workflows/coverage/badge.svg)](https://github.com/liquidos-ai/AutoAgents/actions)
  [![codecov](https://codecov.io/gh/liquidos-ai/AutoAgents/graph/badge.svg)](https://codecov.io/gh/liquidos-ai/AutoAgents)

  [Documentation](https://liquidos-ai.github.io/AutoAgents/) | [Examples](examples/) | [Contributing](CONTRIBUTING.md)
</div>

---

## 🚀 Overview

AutoAgents is a cutting-edge multi-agent framework built in Rust that enables the creation of intelligent, autonomous agents powered by Large Language Models (LLMs). Designed for performance, safety, and scalability, AutoAgents provides a robust foundation for building complex AI systems that can reason, act, and collaborate.

---

## ✨ Key Features

### 🔧 **Extensive Tool Integration**
- **Built-in Tools**: File operations, web scraping, API calls, and more coming soon!
- **Custom Tools**: Easy integration of external tools and services
- **Tool Chaining**: Complex workflows through tool composition

### 🏗️ **Flexible Architecture**
- **Modular Design**: Plugin-based architecture for easy extensibility
- **Provider Agnostic**: Support for multiple LLM providers
- **Memory Systems**: Configurable memory backends (sliding window, persistent, etc.)

### 📊 **Structured Outputs**
- **JSON Schema Support**: Type-safe agent responses with automatic validation
- **Custom Output Types**: Define complex structured outputs for your agents
- **Serialization**: Built-in support for various data formats

### 🎯 **ReAct Framework**
- **Reasoning**: Advanced reasoning capabilities with step-by-step logic
- **Acting**: Tool execution with intelligent decision making
- **Observation**: Environmental feedback and adaptation

### 🤖 **Multi-Agent Orchestration**
- **Agent Coordination**: Seamless communication and collaboration between multiple agents
- **Task Distribution**: Intelligent workload distribution across agent networks
- **Knowledge Sharing**: Shared memory and context between agents (In Roadmap

---

## 🌐 Supported LLM Providers

AutoAgents supports a wide range of LLM providers, allowing you to choose the best fit for your use case:

| Provider | Status |
|----------|---------|
| **OpenAI** | ✅ |
| **Anthropic** | ✅ |
| **Ollama** | ✅ |
| **DeepSeek** | ✅ |
| **xAI** | ✅ |
| **Phind** | ✅ |
| **Groq** | ✅ |
| **Google** | ✅ |
| **Azure OpenAI** | ✅ |

*Provider support is actively expanding based on community needs.*

---

## 📦 Installation

### Adding AutoAgents to Your Project

Add AutoAgents to your `Cargo.toml`:

```toml
[dependencies]
autoagents = "0.2.0-alpha.0"
autoagents-derive = "0.2.0-alpha.0"
tokio = { version = "1.43.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

### Development Setup

For contributing to AutoAgents or building from source:

#### Prerequisites

- **Rust** (latest stable recommended)
- **Cargo** package manager
- **LeftHook** for Git hooks management

#### Install LeftHook

**macOS (using Homebrew):**
```bash
brew install lefthook
```

**Linux/Windows:**
```bash
# Using npm
npm install -g lefthook

# Or download binary from releases
curl -1sLf 'https://dl.cloudsmith.io/public/evilmartians/lefthook/setup.deb.sh' | sudo -E bash
sudo apt install lefthook
```

#### Clone and Setup

```bash
# Clone the repository
git clone https://github.com/liquidos-ai/AutoAgents.git
cd AutoAgents

# Install Git hooks using lefthook
lefthook install

# Build the project
cargo build --release

# Run tests to verify setup
cargo test --all-features
```

The lefthook configuration will automatically:
- Format code with `cargo fmt`
- Run linting with `cargo clippy`
- Execute tests before commits

---

## 🚀 Quick Start

### Basic Usage

```rust
use autoagents::core::agent::prebuilt::react::{ReActAgentOutput, ReActExecutor};
use autoagents::core::agent::{AgentBuilder, AgentDeriveT, AgentOutputT};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::memory::SlidingWindowMemory;
use autoagents::core::protocol::{Event, TaskResult};
use autoagents::core::runtime::{Runtime, SingleThreadedRuntime};
use autoagents::init_logging;
use autoagents::llm::{ToolCallError, ToolInputT, ToolT};
use autoagents::{llm::backends::openai::OpenAI, llm::builder::LLMBuilder};
use autoagents_derive::{agent, tool, AgentOutput, ToolInput};
use colored::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

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
fn add(args: AdditionArgs) -> Result<i64, ToolCallError> {
    Ok(args.left + args.right)
}

/// Math agent output with Value and Explanation
#[derive(Debug, Serialize, Deserialize, AgentOutput)]
pub struct MathAgentOutput {
    #[output(description = "The addition result")]
    value: i64,
    #[output(description = "Explanation of the logic")]
    explanation: String,
}

#[agent(
    name = "math_agent",
    description = "You are a Math agent",
    tools = [Addition],
    executor = ReActExecutor,
    output = MathAgentOutput
)]
pub struct MathAgent {}

#[tokio::main]
async fn main() -> Result<(), Error> {
    init_logging();
    // Check if API key is set
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or("".into());

    // Initialize and configure the LLM client
    let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(api_key) // Set the API key
        .model("gpt-4o") // Use GPT-4o-mini model
        .max_tokens(512) // Limit response length
        .temperature(0.2) // Control response randomness (0.0-1.0)
        .stream(false) // Disable streaming responses
        .build()
        .expect("Failed to build LLM");

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent = MathAgent {};

    let runtime = SingleThreadedRuntime::new(None);

    let _ = AgentBuilder::new(agent)
        .with_llm(llm)
        .runtime(runtime.clone())
        .subscribe_topic("test")
        .with_memory(sliding_window_memory)
        .build()
        .await?;

    // Create environment and set up event handling
    let mut environment = Environment::new(None);
    let _ = environment.register_runtime(runtime.clone()).await;

    let receiver = environment.take_event_receiver(None).await;
    handle_events(receiver);

    environment.run();

    runtime
        .publish_message("What is 2 + 2?".into(), "test".into())
        .await
        .unwrap();
    runtime
        .publish_message("What did I ask before?".into(), "test".into())
        .await
        .unwrap();

    // Shutdown
    let _ = environment.shutdown().await;
    Ok(())
}

fn handle_events(event_stream: Option<ReceiverStream<Event>>) {
    if let Some(mut event_stream) = event_stream {
        tokio::spawn(async move {
            while let Some(event) = event_stream.next().await {
                match event {
                    Event::TaskComplete { result, .. } => {
                        match result {
                            TaskResult::Value(val) => {
                                let agent_out: ReActAgentOutput =
                                    serde_json::from_value(val).unwrap();
                                let math_out: MathAgentOutput =
                                    serde_json::from_str(&agent_out.response).unwrap();
                                println!(
                                    "{}",
                                    format!(
                                        "Math Value: {}, Explanation: {}",
                                        math_out.value, math_out.explanation
                                    )
                                    .green()
                                );
                            }
                            _ => {
                                //
                            }
                        }
                    }
                    _ => {
                        //
                    }
                }
            }
        });
    }
}
```

---

## 📚 Examples

Explore our comprehensive examples to get started quickly:

### [Basic Agent](examples/basic/)
A simple agent demonstrating core functionality and event-driven architecture.

```bash
export OPENAI_API_KEY="your-api-key"
cargo run --package basic -- --usecase simple
```

### [Coding Agent](examples/coding_agent/)
A sophisticated ReAct-based coding agent with file manipulation capabilities.

```bash
export OPENAI_API_KEY="your-api-key"
cargo run --package coding_agent -- --usecase interactive
```

---

## 🏗️ Architecture

AutoAgents is built with a modular architecture:

```
AutoAgents/
├── crates/
│   ├── autoagents/     # Main library entry point
│   ├── core/           # Core agent framework
│   ├── llm/            # LLM provider implementations
│   └── derive/         # Procedural macros
├── examples/           # Example implementations
```

### Core Components

- **Agent**: The fundamental unit of intelligence
- **Environment**: Manages agent lifecycle and communication
- **Memory**: Configurable memory systems
- **Tools**: External capability integration
- **Executors**: Different reasoning patterns (ReAct, Chain-of-Thought)

---

## 🛠️ Development

### Setup

For development setup instructions, see the [Installation](#-installation) section above.

### Running Tests

```bash
# Run all tests
cargo test --all-features

# Run tests with coverage (requires cargo-tarpaulin)
cargo install cargo-tarpaulin
cargo tarpaulin --all-features --out html
```

### Git Hooks

This project uses LeftHook for Git hooks management. The hooks will automatically:
- Format code with `cargo fmt --check`
- Run linting with `cargo clippy -- -D warnings`  
- Execute tests with `cargo test --features full`

### Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) for details.

---

## 📖 Documentation

- **[API Documentation](https://liquidos-ai.github.io/AutoAgents)**: Complete Framework Docs
- **[Examples](examples/)**: Practical implementation examples

---

## 🤝 Community

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Community Q&A and ideas
- **Discord**: Join our Discord Community using https://discord.gg/Ghau8xYn

---

## 📊 Performance

AutoAgents is designed for high performance:

- **Memory Efficient**: Optimized memory usage with configurable backends
- **Concurrent**: Full async/await support with tokio
- **Scalable**: Horizontal scaling with multi-agent coordination
- **Type Safe**: Compile-time guarantees with Rust's type system

---

## 🗺️ Roadmap

- [ ] Enhanced tool system and marketplace
- [ ] Advanced memory systems and RAG integration
- [ ] Distributed agent networks
- [ ] Production-ready stable release

---

## 📜 License

AutoAgents is dual-licensed under:

- **MIT License** ([MIT_LICENSE](MIT_LICENSE))
- **Apache License 2.0** ([APACHE_LICENSE](APACHE_LICENSE))

You may choose either license for your use case.

---

## 🙏 Acknowledgments

Built with ❤️ by the [Liquidos AI](https://liquidos.ai) team and our amazing community contributors.

Special thanks to:
- The Rust community for the excellent ecosystem
- OpenAI, Anthropic, and other LLM providers for their APIs
- All contributors who help make AutoAgents better

---

<div align="center">
  <strong>Ready to build intelligent agents? Get started with AutoAgents today!</strong>

  ⭐ **Star us on GitHub** | 🐛 **Report Issues** | 💬 **Join Discussions**
</div>
