# Introduction

Welcome to **AutoAgents**, a cutting-edge multi-agent framework built in Rust that enables the creation of intelligent, autonomous agents powered by Large Language Models (LLMs). Designed with performance, safety, and scalability at its core, AutoAgents provides a robust foundation for building complex AI systems that can reason, act, and collaborate.

## What is AutoAgents?

AutoAgents is a comprehensive framework that allows developers to create AI agents that can:

- **Reason**: Use advanced reasoning patterns like ReAct (Reasoning and Acting) to break down complex problems
- **Act**: Execute tools and interact with external systems to accomplish tasks
- **Remember**: Maintain context and conversation history through flexible memory systems
- **Collaborate**: Work together in multi-agent environments (coming soon)

## Why Choose AutoAgents?

### 🚀 **Performance First**
Built in Rust, AutoAgents delivers exceptional performance with:
- Zero-cost abstractions
- Memory safety without garbage collection
- Async/await support for high concurrency
- Efficient tool execution and memory management

### 🔧 **Extensible Architecture**
- **Modular Design**: Plugin-based architecture for easy customization
- **Provider Agnostic**: Support for multiple LLM providers (OpenAI, Anthropic, Ollama, and more)
- **Custom Tools**: Easy integration of external tools and services
- **Flexible Memory**: Configurable memory backends for different use cases

### 🎯 **Developer Experience**
- **Declarative Macros**: Define agents with simple, readable syntax
- **Type Safety**: Compile-time guarantees with structured outputs
- **Rich Tooling**: Comprehensive error handling and debugging support
- **Extensive Documentation**: Clear guides and examples for every feature

### 🌐 **Multi-Provider Support**
AutoAgents supports a wide range of LLM providers out of the box:

| Provider | Status | Features |
|----------|--------|----------|
| OpenAI | ✅ | GPT-4, GPT-3.5, Function Calling |
| Anthropic | ✅ | Claude 3, Tool Use |
| Ollama | ✅ | Local Models, Custom Models |
| Google | ✅ | Gemini Pro, Gemini Flash |
| Groq | ✅ | Fast Inference |
| DeepSeek | ✅ | Code-focused Models |
| xAI | ✅ | Grok Models |
| Phind | ✅ | Developer-focused |
| Azure OpenAI | ✅ | Enterprise Integration |

## Key Features

### ReAct Framework
AutoAgents implements the ReAct (Reasoning and Acting) pattern, allowing agents to:
1. **Reason** about problems step-by-step
2. **Act** by calling tools and functions
3. **Observe** the results and adjust their approach

### Structured Outputs
Define type-safe outputs for your agents using JSON Schema:
```rust
#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct WeatherOutput {
    #[output(description = "Temperature in Celsius")]
    temperature: f64,
    #[output(description = "Weather description")]
    description: String,
}
```

### Tool Integration
Build powerful agents by connecting them to external tools:
- File system operations
- Web scraping and API calls
- Database interactions
- Custom business logic

### Memory Systems
Choose from various memory backends:
- **Sliding Window**: Keep recent conversation history
- **Persistent**: Long-term memory storage
- **Custom**: Implement your own memory strategy

## Use Cases

AutoAgents is perfect for building:

### 🤖 **AI Assistants**
- Customer support chatbots
- Personal productivity assistants
- Domain-specific expert systems

### 🛠️ **Development Tools**
- Code generation and review agents
- Automated testing assistants
- Documentation generators

### 📊 **Data Processing**
- Document analysis and summarization
- Data extraction and transformation
- Report generation

### 🔗 **Integration Agents**
- API orchestration
- Workflow automation
- System monitoring and alerting

## Getting Started

Ready to build your first agent? Here's a simple example:

```rust
use autoagents::prelude::*;

#[agent(
    name = "hello_agent",
    description = "A friendly greeting agent",
    executor = ReActExecutor,
)]
pub struct HelloAgent {}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = OpenAIProvider::new("your-api-key")?;
    
    let agent = AgentBuilder::new(HelloAgent {})
        .with_llm(Arc::new(llm))
        .build()?;
    
    let mut environment = Environment::new(None).await;
    let agent_id = environment.register_agent(agent, None).await?;
    
    environment.add_task(agent_id, "Say hello to the world!").await?;
    let results = environment.run_all(agent_id, None).await?;
    
    println!("Agent response: {}", results[0].output.as_ref().unwrap());
    Ok(())
}
```

## Community and Support

AutoAgents is developed by the [Liquidos AI](https://liquidos.ai) team and maintained by a growing community of contributors.

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