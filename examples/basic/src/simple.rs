use autoagents::core::agent::base::AgentDeriveT;
use autoagents::core::agent::types::SimpleAgentBuilder;
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::protocol::Event;
use autoagents::llm::{LLMProvider, ToolInputT, ToolT};
use autoagents_derive::{agent, tool, ToolInput};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct WeatherArgs {}

#[tool(
    name = "WeatherTool",
    description = "Use this tool to get current weather in celcius",
    input = WeatherArgs,
    output = String
)]
fn get_weather(args: WeatherArgs) -> String {
    println!("ToolCall: GetWeather {:?}", args);
    "23 degree".into()
}

#[agent(
    name = "general_agent",
    prompt = "You are general assistant and will answer user quesries in crips manner.",
    tools = [WeatherTool]
)]
pub struct WeatherAgent {}

fn handle_events(event_stream: Option<ReceiverStream<Event>>) {
    if let Some(mut event_stream) = event_stream {
        tokio::spawn(async move {
            while let Some(event) = event_stream.next().await {
                println!("Event: {:?}", event);
            }
        });
    }
}

pub async fn simple_agent(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    // Build a Simple agent
    let agent = SimpleAgentBuilder::from_agent(WeatherAgent {})
        .with_llm(llm)
        .build()
        .unwrap();
    let mut environment = Environment::new(None).await;
    let receiver = environment.take_event_receiver(None).await;
    handle_events(receiver);

    let agent_id = environment.register_agent(agent, None).await?;
    let _ = environment
        .add_task(agent_id, "What is the current weather??")
        .await?;
    let _ = environment
        .add_task(
            agent_id,
            "What is the weather you just said? reply back in beautiful format.",
        )
        .await?;

    let result = environment.run_all(agent_id, None).await?;
    println!("Result {:?}", result);
    let _ = environment.shutdown().await;
    Ok(())
}
