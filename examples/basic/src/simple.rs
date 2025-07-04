use autoagents::core::agent::base::AgentBuilder;
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::protocol::Event;
use autoagents::llm::{LLMProvider, ToolCallError, ToolInputT, ToolT};
use autoagents_core::agent::{AgentDeriveT, ReActExecutor};
use autoagents_core::memory::SlidingWindowMemory;
use autoagents_derive::{agent, tool, ToolInput};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

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
    if args.city == "Chennai" || args.city == "Hyderabad" {
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
    output = String,
)]
pub struct WeatherAgent {}

fn handle_events(event_stream: Option<ReceiverStream<Event>>) {
    if let Some(mut event_stream) = event_stream {
        tokio::spawn(async move {
            while let Some(event) = event_stream.next().await {
                match event {
                    Event::TaskStarted {
                        agent_id,
                        task_description,
                        ..
                    } => {
                        println!(
                            "📋 Task Started - Agent: {:?}, Task: {}",
                            agent_id, task_description
                        );
                    }
                    Event::ToolCallRequested {
                        tool_name,
                        arguments,
                        ..
                    } => {
                        println!(
                            "🔧 Tool Call Started: {} with args: {}",
                            tool_name, arguments
                        );
                    }
                    Event::ToolCallCompleted {
                        tool_name, result, ..
                    } => {
                        println!(
                            "✅ Tool Call Completed: {} - Result: {:?}",
                            tool_name, result
                        );
                    }
                    Event::TaskComplete { result, .. } => {
                        println!("✨ Task Complete: {:?}", result);
                    }
                    Event::Warning { message } => {
                        println!("⚠️  Warning: {}", message);
                    }
                    Event::TurnStarted {
                        turn_number,
                        max_turns,
                    } => {
                        println!("🔄 Turn {}/{} started", turn_number + 1, max_turns);
                    }
                    Event::TurnCompleted {
                        turn_number,
                        final_turn,
                    } => {
                        println!(
                            "✓ Turn {} completed{}",
                            turn_number + 1,
                            if final_turn { " (final)" } else { "" }
                        );
                    }
                    _ => {
                        println!("📡 Event: {:?}", event);
                    }
                }
            }
        });
    }
}

pub async fn simple_agent(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    println!("Starting Weather Agent Example...\n");

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let weather_agent = WeatherAgent {};
    let executor = ReActExecutor::new();

    let agent = AgentBuilder::new(weather_agent, executor)
        .with_llm(llm)
        .with_memory(sliding_window_memory)
        .build()
        .unwrap();

    // Create environment and set up event handling
    let mut environment = Environment::new(None).await;
    let receiver = environment.take_event_receiver(None).await;
    handle_events(receiver);

    // Register the agent
    let agent_id = environment.register_agent(agent, None).await?;

    // Add tasks
    let _ = environment
        .add_task(agent_id, "Hello! My name is John")
        .await?;

    let _ = environment.add_task(agent_id, "What is My name?").await?;

    // let _ = environment
    //     .add_task(
    //         agent_id,
    //         "Compare the weather between Chennai and New York. Which city is warmer?",
    //     )
    //     .await?;

    // Run all tasks
    let _ = environment.run_all(agent_id, None).await?;

    // Shutdown
    let _ = environment.shutdown().await;
    Ok(())
}
