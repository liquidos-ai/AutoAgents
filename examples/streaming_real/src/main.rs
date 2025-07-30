use autoagents::core::agent::prebuilt::react::{ReActAgentOutput, ReActExecutor};
use autoagents::core::agent::{AgentBuilder, AgentDeriveT, AgentOutputT};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::memory::SlidingWindowMemory;
use autoagents::core::protocol::{Event, TaskResult};
use autoagents::core::runtime::{Runtime, SingleThreadedRuntime};
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents::llm::{LLMProvider, builder::LLMBuilder, backends::openai::OpenAI};
use autoagents_derive::{agent, tool, AgentOutput, ToolInput};
use colored::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct CalculatorArgs {
    #[input(description = "The mathematical operation to perform")]
    operation: String,
    #[input(description = "First number")]
    a: f64,
    #[input(description = "Second number")]
    b: f64,
}

#[tool(
    name = "calculator",
    description = "Perform basic mathematical operations (add, subtract, multiply, divide)",
    input = CalculatorArgs,
)]
struct Calculator {}

impl ToolRuntime for Calculator {
    fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let typed_args: CalculatorArgs = serde_json::from_value(args)?;
        
        let result = match typed_args.operation.as_str() {
            "add" => typed_args.a + typed_args.b,
            "subtract" => typed_args.a - typed_args.b,
            "multiply" => typed_args.a * typed_args.b,
            "divide" => {
                if typed_args.b == 0.0 {
                    return Err(ToolCallError::RuntimeError(
                        Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "Division by zero"
                        ))
                    ));
                }
                typed_args.a / typed_args.b
            }
            _ => {
                return Err(ToolCallError::RuntimeError(
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("Unknown operation: {}", typed_args.operation)
                    ))
                ));
            }
        };
        
        Ok(serde_json::json!({
            "result": result,
            "operation": typed_args.operation,
            "a": typed_args.a,
            "b": typed_args.b
        }))
    }
}

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct WeatherArgs {
    #[input(description = "City name to get weather for")]
    city: String,
}

#[tool(
    name = "get_weather",
    description = "Get current weather information for a city",
    input = WeatherArgs,
)]
struct WeatherTool {}

impl ToolRuntime for WeatherTool {
    fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let typed_args: WeatherArgs = serde_json::from_value(args)?;
        
        // Simulate weather data (in a real app, this would call a weather API)
        let weather_data = match typed_args.city.to_lowercase().as_str() {
            "new york" => serde_json::json!({
                "city": "New York",
                "temperature": 22,
                "condition": "Partly Cloudy",
                "humidity": 65,
                "wind_speed": 12
            }),
            "london" => serde_json::json!({
                "city": "London",
                "temperature": 15,
                "condition": "Rainy",
                "humidity": 80,
                "wind_speed": 18
            }),
            "tokyo" => serde_json::json!({
                "city": "Tokyo",
                "temperature": 28,
                "condition": "Sunny",
                "humidity": 55,
                "wind_speed": 8
            }),
            _ => serde_json::json!({
                "city": typed_args.city,
                "temperature": 20,
                "condition": "Unknown",
                "humidity": 60,
                "wind_speed": 10
            })
        };
        
        Ok(weather_data)
    }
}

/// Math agent output with Value and Explanation
#[derive(Debug, Serialize, Deserialize, AgentOutput)]
pub struct MathAgentOutput {
    #[output(description = "The calculated result")]
    value: f64,
    #[output(description = "Explanation of the calculation")]
    explanation: String,
}

#[agent(
    name = "math_agent",
    description = "You are a Math agent that can perform calculations and explain the results. Use the calculator tool to perform mathematical operations.",
    tools = [Calculator, WeatherTool],
    output = MathAgentOutput
)]
pub struct MathAgent {}

impl ReActExecutor for MathAgent {}

pub async fn streaming_agent_with_real_llm() -> Result<(), Error> {
    println!("🚀 Real Streaming Agent Demo");
    println!("This example uses real LLM providers with streaming support\n");

    // Get API key from environment
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| Error::LLMError(autoagents_llm::error::LLMError::InvalidRequest("OPENAI_API_KEY environment variable is required".to_string())))?;
    
    println!("✅ Using real OpenAI API with streaming");
    println!("🔐 API Key: {}...", &api_key[..8.min(api_key.len())]);
    
    // Create real OpenAI provider with streaming enabled
    let llm: Arc<dyn LLMProvider> = LLMBuilder::<OpenAI>::new()
        .api_key(api_key.clone())
        .model("gpt-4o-mini") // Use a model that supports streaming
        .stream(true)
        .temperature(0.7)
        .max_tokens(1000)
        .system("You are a helpful math assistant. When performing calculations, use the calculator tool and explain your reasoning step by step.")
        .build()?;



    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));
    let agent = MathAgent {};
    let runtime = SingleThreadedRuntime::new(None);

    let _ = AgentBuilder::new(agent)
        .with_llm(llm)
        .runtime(runtime.clone())
        .subscribe_topic("streaming_demo")
        .with_memory(sliding_window_memory)
        .build()
        .await?;

    // Create environment and set up event handling
    let mut environment = Environment::new(None);
    let _ = environment.register_runtime(runtime.clone()).await;

    let receiver = environment.take_event_receiver(None).await?;
    handle_streaming_events(receiver);

    // Send test messages
    println!("\n📝 Sending test messages...");
    
    runtime
        .publish_message("What is 15 + 27? Please use the calculator tool.".into(), "streaming_demo".into())
        .await?;
    
    runtime
        .publish_message("What's the weather like in New York?".into(), "streaming_demo".into())
        .await?;
    
    runtime
        .publish_message("Calculate 100 divided by 5 and explain the result.".into(), "streaming_demo".into())
        .await?;

    let _ = environment.run().await;
    Ok(())
}

fn handle_streaming_events(mut event_stream: ReceiverStream<Event>) {
    tokio::spawn(async move {
        while let Some(event) = event_stream.next().await {
            match event {
                Event::TaskStarted {
                    agent_id,
                    task_description,
                    ..
                } => {
                    println!(
                        "{}",
                        format!(
                            "📋 Task Started - Agent: {:?}, Task: {}",
                            agent_id, task_description
                        )
                        .green()
                    );
                }
                
                // Streaming events
                Event::StreamTextChunk { chunk, is_final, .. } => {
                    print!("{}", chunk);
                    if is_final {
                        println!();
                    }
                }
                
                Event::StreamToolCallStart { tool_call, .. } => {
                    println!(
                        "\n{}",
                        format!("🔧 Tool Call Started: {} with args: {}", 
                            tool_call.function.name, tool_call.function.arguments)
                            .yellow()
                    );
                }
                
                Event::StreamToolCallChunk { chunk, is_final, .. } => {
                    print!("{}", chunk);
                    if is_final {
                        println!();
                    }
                }
                
                Event::StreamToolCallEnd { tool_call_id, .. } => {
                    println!(
                        "{}",
                        format!("✅ Tool Call Completed: {}", tool_call_id).yellow()
                    );
                }
                
                Event::StreamThinkingChunk { chunk, is_final, .. } => {
                    print!("{}", format!("🧠 {}", chunk).blue());
                    if is_final {
                        println!();
                    }
                }
                
                // Regular tool call events
                Event::ToolCallRequested {
                    tool_name,
                    arguments,
                    ..
                } => {
                    println!(
                        "{}",
                        format!("🔧 Tool Call Requested: {} with args: {}", tool_name, arguments)
                            .yellow()
                    );
                }
                
                Event::ToolCallCompleted {
                    tool_name, result, ..
                } => {
                    println!(
                        "{}",
                        format!("✅ Tool Call Completed: {} - Result: {:?}", tool_name, result)
                            .green()
                    );
                }
                
                Event::TaskComplete { result, .. } => match result {
                    TaskResult::Value(val) => {
                        // Clone the value so we can use it in error cases
                        let val_clone = val.clone();
                        match serde_json::from_value::<ReActAgentOutput>(val) {
                            Ok(agent_out) => {
                                // Try to parse as structured output first
                                match serde_json::from_str::<MathAgentOutput>(&agent_out.response) {
                                    Ok(math_out) => {
                                        println!(
                                            "{}",
                                            format!(
                                                "🎉 Task Complete - Value: {}, Explanation: {}",
                                                math_out.value, math_out.explanation
                                            )
                                            .green()
                                        );
                                    }
                                    Err(_) => {
                                        // If structured parsing fails, just show the raw response
                                        println!(
                                            "{}",
                                            format!("🎉 Task Complete - Response: {}", agent_out.response)
                                                .green()
                                        );
                                    }
                                }
                            }
                            Err(_) => {
                                // If we can't parse the agent output, just show the raw value
                                println!(
                                    "{}",
                                    format!("🎉 Task Complete - Raw Result: {:?}", val_clone).green()
                                );
                            }
                        }
                    }
                    TaskResult::Failure(error) => {
                        println!("{}", format!("❌ Task Failed: {}", error).red());
                    }
                    TaskResult::Aborted => {
                        println!("{}", format!("⏹️ Task Aborted").yellow());
                    }
                },
                
                Event::TurnStarted {
                    turn_number,
                    max_turns,
                } => {
                    println!(
                        "{}",
                        format!("🔄 Turn {}/{} started", turn_number + 1, max_turns).cyan()
                    );
                }
                
                Event::TurnCompleted {
                    turn_number,
                    final_turn,
                } => {
                    println!(
                        "{}",
                        format!(
                            "✅ Turn {} completed{}",
                            turn_number + 1,
                            if final_turn { " (final)" } else { "" }
                        )
                        .cyan()
                    );
                }
                
                _ => {
                    // Ignore other events for cleaner output
                }
            }
        }
    });
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    streaming_agent_with_real_llm().await
} 