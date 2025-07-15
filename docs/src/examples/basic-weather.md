# Basic Weather Agent Example

This example demonstrates how to create a simple weather agent using AutoAgents. The agent can retrieve weather information for cities and compare weather conditions between different locations.

## Overview

The weather agent example showcases:
- Basic agent creation with the `#[agent]` macro
- Tool implementation with the `#[tool]` macro
- ReAct executor for reasoning and acting
- Memory management with sliding window
- Environment setup and task execution

## Complete Example

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

// Define the input structure for the weather tool
#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct WeatherArgs {
    #[input(description = "City to get weather for")]
    city: String,
}

// Implement the weather tool
#[tool(
    name = "WeatherTool",
    description = "Use this tool to get current weather in celsius",
    input = WeatherArgs,
)]
fn get_weather(args: WeatherArgs) -> Result<String, ToolCallError> {
    println!("ToolCall: GetWeather {:?}", args);

    // Simulate weather data (in real implementation, call weather API)
    match args.city.as_str() {
        "Hyderabad" => Ok(format!(
            "The current temperature in {} is 28 degrees celsius",
            args.city
        )),
        "New York" => Ok(format!(
            "The current temperature in {} is 15 degrees celsius",
            args.city
        )),
        "London" => Ok(format!(
            "The current temperature in {} is 12 degrees celsius",
            args.city
        )),
        "Tokyo" => Ok(format!(
            "The current temperature in {} is 22 degrees celsius",
            args.city
        )),
        _ => Err(ToolCallError::RuntimeError(
            format!("Weather for {} is not supported", args.city).into(),
        )),
    }
}

// Define the weather agent
#[agent(
    name = "weather_agent",
    description = "You are a weather assistant that helps users get weather information.",
    tools = [WeatherTool],
    executor = ReActExecutor,
)]
pub struct WeatherAgent {}

// Main function demonstrating the weather agent
pub async fn simple_agent(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    println!("Starting Weather Agent Example...\n");

    // Create memory with sliding window of 10 messages
    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    // Create the weather agent instance
    let weather_agent = WeatherAgent {};

    // Build the agent with LLM and memory
    let agent = AgentBuilder::new(weather_agent)
        .with_llm(llm)
        .with_memory(sliding_window_memory)
        .build()?;

    // Create environment for agent execution
    let mut environment = Environment::new(None).await;

    // Register the agent with the environment
    let agent_id = environment.register_agent(agent.clone(), None).await?;

    // Add tasks for the agent to execute
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

    // Shutdown the environment
    let _ = environment.shutdown().await;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the LLM provider (requires API key)
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable must be set");

    let llm = Arc::new(autoagents::llm::OpenAIProvider::new(&api_key)?);

    // Run the weather agent example
    simple_agent(llm).await?;

    Ok(())
}
```

## Step-by-Step Breakdown

### 1. Tool Input Definition

```rust
#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct WeatherArgs {
    #[input(description = "City to get weather for")]
    city: String,
}
```

The `WeatherArgs` structure defines the input parameters for our weather tool. The `#[input]` attribute provides a description that helps the LLM understand how to use the parameter.

### 2. Tool Implementation

```rust
#[tool(
    name = "WeatherTool",
    description = "Use this tool to get current weather in celsius",
    input = WeatherArgs,
)]
fn get_weather(args: WeatherArgs) -> Result<String, ToolCallError> {
    // Tool implementation
}
```

The `#[tool]` macro automatically generates the necessary boilerplate code to integrate the function as a tool that agents can call. The function:
- Takes structured input defined by `WeatherArgs`
- Returns either a successful result or a `ToolCallError`
- Simulates weather data (in a real app, you'd call an actual weather API)

### 3. Agent Definition

```rust
#[agent(
    name = "weather_agent",
    description = "You are a weather assistant that helps users get weather information.",
    tools = [WeatherTool],
    executor = ReActExecutor,
)]
pub struct WeatherAgent {}
```

The `#[agent]` macro creates an agent with:
- A descriptive name and purpose
- Access to the `WeatherTool`
- ReAct executor for reasoning and acting

### 4. Agent Construction

```rust
let agent = AgentBuilder::new(weather_agent)
    .with_llm(llm)
    .with_memory(sliding_window_memory)
    .build()?;
```

The `AgentBuilder` provides a fluent API for constructing agents with:
- LLM provider for intelligence
- Memory system for context retention
- Additional configuration options

### 5. Environment Setup

```rust
let mut environment = Environment::new(None).await;
let agent_id = environment.register_agent(agent.clone(), None).await?;
```

The Environment manages agent lifecycle and task execution:
- Registers agents and assigns unique IDs
- Manages task queues
- Handles agent communication

### 6. Task Execution

```rust
environment.add_task(agent_id, "What is the weather in Hyderabad and New York?").await?;
let results = environment.run_all(agent_id, None).await?;
```

Tasks are added to the environment and executed by the agent using the ReAct pattern:
1. **Thought**: The agent analyzes the task
2. **Action**: Calls the weather tool for each city
3. **Observation**: Reviews the tool results
4. **Response**: Provides the final answer

## Expected Output

When you run this example, you should see output similar to:

```
Starting Weather Agent Example...

ToolCall: GetWeather WeatherArgs { city: "Hyderabad" }
ToolCall: GetWeather WeatherArgs { city: "New York" }
ToolCall: GetWeather WeatherArgs { city: "Hyderabad" }
ToolCall: GetWeather WeatherArgs { city: "New York" }

Results: Based on the weather information I retrieved:

**Current Weather:**
- Hyderabad: 28°C
- New York: 15°C

**Comparison:**
Hyderabad is warmer than New York by 13 degrees celsius. Hyderabad has a temperature of 28°C while New York has 15°C, making Hyderabad the warmer city of the two.
```

## Key Concepts Demonstrated

### ReAct Pattern
The agent uses the ReAct (Reasoning and Acting) pattern:
- **Reasoning**: Understands the task and plans actions
- **Acting**: Executes tools to gather information
- **Observing**: Processes tool results and continues reasoning

### Memory Management
The sliding window memory:
- Keeps the last 10 conversation messages
- Provides context for multi-turn conversations
- Automatically manages memory size

### Tool Integration
The weather tool demonstrates:
- Structured input validation
- Error handling for unsupported cities
- Returning formatted results

## Running the Example

1. **Set up your environment:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

2. **Add dependencies to Cargo.toml:**
   ```toml
   [dependencies]
   autoagents = { git = "https://github.com/liquidos-ai/AutoAgents", features = ["openai"] }
   autoagents-derive = {git = "https://github.com/liquidos-ai/AutoAgents"}
   tokio = { version = "1.0", features = ["full"] }
   serde = { version = "1.0", features = ["derive"] }
   serde_json = "1.0"
   ```

3. **Run the example:**
   ```bash
   cargo run
   ```

## Extending the Example

### Adding More Cities
```rust
fn get_weather(args: WeatherArgs) -> Result<String, ToolCallError> {
    match args.city.as_str() {
        "Hyderabad" => Ok("28°C, sunny".to_string()),
        "New York" => Ok("15°C, cloudy".to_string()),
        "London" => Ok("12°C, rainy".to_string()),
        "Paris" => Ok("18°C, partly cloudy".to_string()),
        "Sydney" => Ok("25°C, sunny".to_string()),
        _ => Err(ToolCallError::RuntimeError(
            format!("Weather for {} is not supported", args.city).into(),
        )),
    }
}
```

### Adding Weather Details
```rust
#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct DetailedWeatherArgs {
    #[input(description = "City to get weather for")]
    city: String,
    #[input(description = "Include forecast (true/false)")]
    include_forecast: bool,
}

#[tool(
    name = "DetailedWeatherTool",
    description = "Get detailed weather information including forecast",
    input = DetailedWeatherArgs,
)]
fn get_detailed_weather(args: DetailedWeatherArgs) -> Result<String, ToolCallError> {
    // Implementation with forecast data
}
```

### Real Weather API Integration
```rust
#[tool(
    name = "RealWeatherTool",
    description = "Get real weather data from OpenWeatherMap API",
    input = WeatherArgs,
)]
async fn get_real_weather(args: WeatherArgs) -> Result<String, ToolCallError> {
    let api_key = std::env::var("OPENWEATHER_API_KEY")
        .map_err(|_| ToolCallError::Configuration("Missing OpenWeather API key".into()))?;

    let url = format!(
        "https://api.openweathermap.org/data/2.5/weather?q={}&appid={}&units=metric",
        args.city, api_key
    );

    let response = reqwest::get(&url).await
        .map_err(|e| ToolCallError::RuntimeError(e.into()))?;

    let weather_data: serde_json::Value = response.json().await
        .map_err(|e| ToolCallError::RuntimeError(e.into()))?;

    let temp = weather_data["main"]["temp"].as_f64().unwrap_or(0.0);
    let description = weather_data["weather"][0]["description"].as_str().unwrap_or("Unknown");

    Ok(format!("The current temperature in {} is {:.1}°C with {}", args.city, temp, description))
}
```

This example provides a solid foundation for understanding how to create agents with tools in AutoAgents. You can extend it with additional tools, more sophisticated reasoning patterns, or real API integrations.
