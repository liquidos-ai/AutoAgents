# Introduction

AutoAgents is a cutting-edge multi-agent framework built in Rust that enables the creation of intelligent, autonomous
agents powered by Large Language Models (LLMs) and [Ractor](https://github.com/slawlor/ractor). Designed for
performance, safety, and scalability. AutoAgents provides a robust foundation for building complex AI systems that can
reason, act, and collaborate. With AutoAgents you can create Cloud Native Agents, Edge Native Agents and Hybrid Models
as well. It is So extensible
that other ML Models can be used to create complex pipelines using Actor Framework.

## What is AutoAgents?

AutoAgents is a comprehensive framework that allows developers to create AI agents that can:

- **Reason**: Use advanced reasoning patterns like ReAct (Reasoning and Acting) to break down complex problems
- **Act**: Execute tools and interact with external systems to accomplish tasks
- **Remember**: Maintain context and conversation history through flexible memory systems
- **Collaborate**: Work together in multi-agent environments (coming soon)

---

## ✨ Key Features

### 🤖 **Agent Execution**

- **Multiple Executors**: ReAct (Reasoning + Acting) and Basic executors with streaming support
- **Structured Outputs**: Type-safe JSON schema validation and custom output types
- **Memory Systems**: Configurable memory backends (sliding window, persistent storage)

### 🔧 **Tool Integration**

- **Built-in Tools**: File operations, web scraping, API calls
- **Custom Tools**: Easy integration with derive macros
- **WASM Runtime**: Sandboxed tool execution with cross-platform compatibility

### 🏗️ **Flexible Architecture**

- **Provider Agnostic**: Support for OpenAI, Anthropic, Ollama, and local models
- **Multi-Platform**: Native Rust, WASM for browsers, and server deployments
- **Multi-Agent**: Type-safe pub/sub communication and agent orchestration

### 🌐 **Deployment Options**

- **Native**: High-performance server and desktop applications
- **Browser**: Run agents directly in web browsers via WebAssembly
- **Edge**: Local inference with ONNX models

---

## Getting Started

Ready to build your first agent? Here's a simple example:

```rust
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::{ReActAgentOutput, ReActExecutor};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentDeriveT, AgentOutputT, DirectAgent};
use autoagents::core::error::Error;
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents::llm::LLMProvider;
use autoagents_derive::{agent, tool, AgentOutput, ToolInput};
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

impl ToolRuntime for Addition {
    fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        println!("execute tool: {:?}", args);
        let typed_args: AdditionArgs = serde_json::from_value(args)?;
        let result = typed_args.left + typed_args.right;
        Ok(result.into())
    }
}

/// Math agent output with Value and Explanation
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
#[derive(Default, Clone)]
pub struct MathAgent {}
impl ReActExecutor for MathAgent {}
impl From<ReActAgentOutput> for MathAgentOutput {
    fn from(output: ReActAgentOutput) -> Self {
        let resp = output.response;
        if output.done && !resp.trim().is_empty() {
            // Try to parse as structured JSON first
            if let Ok(value) = serde_json::from_str::<MathAgentOutput>(&resp) {
                return value;
            }
        }
        // For streaming chunks or unparseable content, create a default response
        MathAgentOutput {
            value: 0,
            explanation: resp,
            generic: None,
        }
    }
}

pub async fn simple_agent(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent = AgentBuilder::<_, DirectAgent>::new(MathAgent {})
        .llm(llm)
        .memory(sliding_window_memory)
        .build()?;

    println!("Running simple_agent with direct run method");

    let result = agent.run(Task::new("What is 1 + 1?")).await?;
    println!("Result: {:?}", result);
    Ok(())
}
```

## Community and Support

AutoAgents is developed by the [Liquidos AI](https://liquidos.ai) team and maintained by a growing community of
contributors.

- 📖 **Documentation**: Comprehensive guides and API reference
- 💬 **Discord**: Join our community at [discord.gg/Ghau8xYn](https://discord.gg/Ghau8xYn)
- 🐛 **Issues**: Report bugs and request features on [GitHub](https://github.com/liquidos-ai/AutoAgents)
- 🤝 **Contributing**: We welcome contributions of all kinds

## What's Next?

This documentation will guide you through:

1. **Installation and Setup**: Get AutoAgents running in your environment
2. **Core Concepts**: Understand the fundamental building blocks
3. **Building Agents**: Create your first intelligent agents
4. **Advanced Features**: Explore powerful capabilities
5. **Real-world Examples**: Learn from practical implementations

Let's start building intelligent agents together! 🚀
