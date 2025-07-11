# AutoAgents 🚀
A multi-agent framework powered by Rust and modern LLMs.


🎥 **Watch the full demo on YouTube** to see AutoAgents in action!

[![Coding Agent Demo](https://img.youtube.com/vi/MZLd4aRuftM/maxresdefault.jpg)](https://youtu.be/MZLd4aRuftM?si=XbHitYjgyffyOf5D)

---

## Features

- **Multi-Agent Coordination:** Allow agents to interact, share knowledge, and collaborate on complex tasks.
- **Tool Integration:** Seamlessly integrate with external tools and services to extend agent capabilities.
- **Extensible Framework:** Easily add new providers and tools using our intuitive plugin system.

---

## Supported Providers

**multiple LLM backends** in a single project: [OpenAI](https://openai.com), [Anthropic (Claude)](https://www.anthropic.com), [Ollama](https://github.com/ollama/ollama), [DeepSeek](https://www.deepseek.com), [xAI](https://x.ai), [Phind](https://www.phind.com), [Groq](https://www.groq.com), [Google](https://cloud.google.com/gemini).

*Note: Provider support is actively evolving.*


## Instructions to build
```sh
git clone https://github.com/liquidos-ai/autoagents.git
cd autoagents
cargo build --release
```

## Usage

### LLM Tool Calling
```rs
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
    output = String,
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
    println!("Results: {:?}", results.last());

    // Shutdown
    let _ = environment.shutdown().await;
    Ok(())
}
```


## Contributing
We welcome contributions!

## License
MIT OR Apache-2.0. See [MIT_LICENSE](MIT_LICENSE) or [APACHE_LICENSE](APACHE_LICENSE) for more.
