# Quick Start

This guide will get you up and running with AutoAgents in just a few minutes. We'll create a simple weather agent that demonstrates the core concepts of the framework.

## Prerequisites

Before starting, make sure you have:
- Rust 1.70+ installed
- AutoAgents added to your `Cargo.toml`
- An OpenAI API key (or another supported LLM provider)

## Step 1: Set Up Your Project

Create a new Rust project:

```bash
cargo new my-autoagent
cd my-autoagent
```

Add AutoAgents to your `Cargo.toml`:

```toml
[dependencies]
autoagents = { version = "0.2.0-alpha.0", features = ["openai"] }
autoagents-derive = "0.2.0-alpha.0"
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

## Step 2: Set Your API Key

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

## Step 3: Create Your First Agent

Replace the contents of `src/main.rs` with:

```rust
use autoagents::core::agent::base::AgentBuilder;
use autoagents::core::agent::{AgentDeriveT, ReActExecutor};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::memory::SlidingWindowMemory;
use autoagents::llm::{LLMProvider, OpenAIProvider, ToolCallError, ToolInputT, ToolT};
use autoagents_derive::{agent, tool, ToolInput};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// Define the input structure for our weather tool
#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct WeatherArgs {
    #[input(description = "City to get weather for")]
    city: String,
}

// Create a simple weather tool
#[tool(
    name = "WeatherTool",
    description = "Use this tool to get current weather in celsius",
    input = WeatherArgs,
)]
fn get_weather(args: WeatherArgs) -> Result<String, ToolCallError> {
    println!("Getting weather for: {}", args.city);
    
    // Simulate weather data (in a real app, you'd call a weather API)
    let weather = match args.city.to_lowercase().as_str() {
        "london" => "London: 15°C, cloudy with light rain",
        "tokyo" => "Tokyo: 22°C, sunny with clear skies",
        "new york" => "New York: 18°C, partly cloudy",
        "sydney" => "Sydney: 25°C, sunny and warm",
        _ => return Err(ToolCallError::RuntimeError(
            format!("Weather data not available for {}", args.city).into(),
        )),
    };
    
    Ok(weather)
}

// Define our weather agent
#[agent(
    name = "weather_agent",
    description = "A helpful weather assistant that provides current weather information for cities worldwide.",
    tools = [WeatherTool],
    executor = ReActExecutor,
)]
pub struct WeatherAgent {}

#[tokio::main]
async fn main() -> Result<(), Error> {
    println!("🌤️  Starting AutoAgents Weather Demo\n");

    // Initialize the LLM provider
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable must be set");
    
    let llm = Arc::new(OpenAIProvider::new(&api_key)?);

    // Create memory for the agent
    let memory = Box::new(SlidingWindowMemory::new(10));

    // Create the agent instance
    let weather_agent = WeatherAgent {};

    // Build the agent with LLM and memory
    let agent = AgentBuilder::new(weather_agent)
        .with_llm(llm)
        .with_memory(memory)
        .build()?;

    // Create the environment
    let mut environment = Environment::new(None).await;

    // Register the agent
    let agent_id = environment.register_agent(agent, None).await?;

    // Add some tasks
    println!("Adding tasks...");
    environment.add_task(
        agent_id,
        "What's the weather like in London?"
    ).await?;

    environment.add_task(
        agent_id,
        "Compare the weather between Tokyo and Sydney"
    ).await?;

    // Run all tasks
    println!("Running tasks...\n");
    let results = environment.run_all(agent_id, None).await?;

    // Display results
    for (i, result) in results.iter().enumerate() {
        println!("Task {} Result:", i + 1);
        if let Some(output) = &result.output {
            println!("{}\n", output);
        }
    }

    // Shutdown
    environment.shutdown().await;
    println!("✅ Demo completed successfully!");
    
    Ok(())
}
```

## Step 4: Run Your Agent

Execute your agent:

```bash
cargo run
```

You should see output similar to:

```
🌤️  Starting AutoAgents Weather Demo

Adding tasks...
Running tasks...

Task 1 Result:
The weather in London is currently 15°C with cloudy skies and light rain.

Task 2 Result:
Comparing the weather between Tokyo and Sydney:
- Tokyo: 22°C, sunny with clear skies
- Sydney: 25°C, sunny and warm

Sydney is warmer than Tokyo by 3°C, and both cities are experiencing sunny weather compared to London's rainy conditions.

✅ Demo completed successfully!
```

## What Just Happened?

Let's break down what we built:

### 1. Tool Definition
```rust
#[tool(
    name = "WeatherTool",
    description = "Use this tool to get current weather in celsius",
    input = WeatherArgs,
)]
```
We created a tool that the agent can call to get weather information. The `#[tool]` macro automatically generates the necessary code for tool integration.

### 2. Agent Definition
```rust
#[agent(
    name = "weather_agent",
    description = "A helpful weather assistant...",
    tools = [WeatherTool],
    executor = ReActExecutor,
)]
```
We defined an agent with:
- A descriptive name and purpose
- Access to our weather tool
- ReAct executor for reasoning and acting

### 3. Environment Setup
```rust
let mut environment = Environment::new(None).await;
let agent_id = environment.register_agent(agent, None).await?;
```
We created an environment to manage our agent and added tasks for it to execute.

## Next Steps

Now that you've created your first agent, you can:

1. **Add More Tools**: Create additional tools for different functionalities
2. **Structured Outputs**: Define specific output formats for your agents
3. **Memory Systems**: Experiment with different memory backends
4. **Multiple Agents**: Create agents that work together
5. **Custom Executors**: Implement your own reasoning patterns

## Common Patterns

### Adding Error Handling
```rust
let weather = match get_weather_from_api(&args.city).await {
    Ok(data) => data,
    Err(e) => return Err(ToolCallError::RuntimeError(e.into())),
};
```

### Using Environment Variables
```rust
let api_key = std::env::var("OPENAI_API_KEY")
    .map_err(|_| Error::Configuration("Missing OPENAI_API_KEY".into()))?;
```

### Multiple Tools
```rust
#[agent(
    name = "multi_tool_agent",
    tools = [WeatherTool, CalculatorTool, WebSearchTool],
    executor = ReActExecutor,
)]
pub struct MultiToolAgent {}
```

## Troubleshooting

### Common Issues
1. **API Key Not Set**: Make sure your API key environment variable is set
2. **Network Issues**: Check your internet connection and API endpoint availability
3. **Rate Limits**: Some providers have rate limits; add delays between requests if needed

### Debug Tips
- Enable logging: `RUST_LOG=debug cargo run`
- Check the agent's reasoning process in the output
- Verify tool inputs and outputs

## What's Next?

Continue with:
- [Your First Agent](./first-agent.md) - A deeper dive into agent creation
- [Core Concepts](../core-concepts/architecture.md) - Understanding the framework architecture
- [Examples](../examples/basic-weather.md) - More complex examples and use cases

Ready to build more sophisticated agents? Let's dive deeper! 🚀