use autoagents::core::agent::prebuilt::react::ReActExecutor;
use autoagents::core::agent::{AgentBuilder, AgentDeriveT};
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

    // let weather_agent = WeatherAgent {};

    // let agent = AgentBuilder::new(weather_agent)
    //     .with_llm(llm)
    //     .with_memory(sliding_window_memory)
    //     .build()?;

    // // Create environment and set up event handling
    // let mut environment = Environment::new(None).await;

    // // Register the agent
    // let agent_id = environment.register_agent(agent.clone(), None).await?;

    // // Add tasks
    // let _ = environment
    //     .add_task(agent_id, "What is the weather in Hyderabad and New York?")
    //     .await?;

    // let _ = environment
    //     .add_task(
    //         agent_id,
    //         "Compare the weather between Hyderabad and New York. Which city is warmer?",
    //     )
    //     .await?;

    // // Run all tasks
    // let results = environment.run_all(agent_id, None).await?;
    // let last_result = results.last().unwrap();
    // println!("Results: {}", last_result.output.as_ref().unwrap());

    // // Shutdown
    // let _ = environment.shutdown().await;
    Ok(())
}
