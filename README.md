<div align="center">
  <img src="assets/logo.png" alt="AutoAgents Logo" width="200" height="200">

  # AutoAgents

  **A Modern Multi-Agent Framework for Rust**

  [![Crates.io](https://img.shields.io/crates/v/autoagents.svg)](https://crates.io/crates/autoagents)
  [![Documentation](https://docs.rs/autoagents/badge.svg)](https://docs.rs/autoagents)
  [![License](https://img.shields.io/crates/l/autoagents.svg)](https://github.com/liquidos-ai/AutoAgents#license)
  [![Build Status](https://github.com/liquidos-ai/AutoAgents/workflows/coverage/badge.svg)](https://github.com/liquidos-ai/AutoAgents/actions)
  [![codecov](https://codecov.io/gh/liquidos-ai/AutoAgents/graph/badge.svg)](https://codecov.io/gh/liquidos-ai/AutoAgents)

  [Documentation](https://docs.rs/autoagents) | [Examples](examples/) | [Contributing](CONTRIBUTING.md)
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
- **Agent Coordination**: Seamless communication and collaboration between multiple agents (In Roadmap)
- **Task Distribution**: Intelligent workload distribution across agent networks (In Roadmap)
- **Knowledge Sharing**: Shared memory and context between agents (In Roadmap)

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

## 🚀 Quick Start

### Basic Usage

```rust
use autoagents::core::agent::base::AgentBuilder;
use autoagents::core::agent::{AgentDeriveT, ReActExecutor};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::memory::SlidingWindowMemory;
use autoagents::llm::{LLMProvider, ToolCallError, ToolInputT, ToolT};
use autoagents_derive::{agent, tool, ToolInput};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct WeatherArgs {
    #[input(description = "City to get weather for")]
    city: String,
}

#[tool(
    name = "WeatherTool",
    description = "Use this tool to get current weather in celsius",
    input = WeatherArgs,
)]
fn get_weather(args: WeatherArgs) -> Result<String, ToolCallError> {
    println!("ToolCall: GetWeather {:?}", args);
    if args.city == "Hyderabad" {
        Ok(format!(
            "The current temperature in {} is 28 degrees celsius",
            args.city
        ))
    } else if args.city == "New York" {
        Ok(format!(
            "The current temperature in {} is 15 degrees celsius",
            args.city
        ))
    } else {
        Err(ToolCallError::RuntimeError(
            format!("Weather for {} is not supported", args.city).into(),
        ))
    }
}

#[agent(
    name = "weather_agent",
    description = "You are a weather assistant that helps users get weather information.",
    tools = [WeatherTool],
    executor = ReActExecutor,
)]
pub struct WeatherAgent {}

pub async fn simple_agent(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    println!("Starting Weather Agent Example...\n");

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let weather_agent = WeatherAgent {};

    let agent = AgentBuilder::new(weather_agent)
        .with_llm(llm)
        .with_memory(sliding_window_memory)
        .build()?;

    // Create environment and set up event handling
    let mut environment = Environment::new(None).await;

    // Register the agent
    let agent_id = environment.register_agent(agent.clone(), None).await?;

    // Add tasks
    let _ = environment
        .add_task(agent_id, "What is the weather in Hyderabad and New York?")
        .await?;

    let _ = environment
        .add_task(
            agent_id,
            "Compare the weather between Hyderabad and New York. Which city is warmer?",
        )
        .await?;

    // Run all tasks
    let results = environment.run_all(agent_id, None).await?;
    let last_result = results.last().unwrap();
    println!("Results: {}", last_result.output.as_ref().unwrap());

    // Shutdown
    let _ = environment.shutdown().await;
    Ok(())
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

### Prerequisites

- Rust (latest stable recommended)
- Cargo package manager
- LeftHook (Git Hooks)
- Cargo Tarpaulin (Coverage)

### Building from Source

```bash
git clone https://github.com/liquidos-ai/AutoAgents.git
cd AutoAgents
cargo build --release
```

### Running Tests

```bash
cargo test --all-features
```

### Contributing

We welcome contributions! Please see our [Code of Conduct](CODE_OF_CONDUCT.md) for details.

---

## 📖 Documentation

- **[API Documentation](https://docs.rs/autoagents)**: Complete API reference
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
