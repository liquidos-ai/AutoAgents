/// This example demonstrates Agent Streaming using the new runtime architecture
use autoagents::core::actor::Topic;
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::{ReActAgentOutput, ReActExecutor};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentDeriveT, RunnableAgent};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::protocol::TaskResult;
use autoagents::core::runtime::SingleThreadedRuntime;
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents::llm::LLMProvider;
use autoagents_derive::{agent, tool, ToolInput};
use colored::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio_stream::StreamExt as TokioStreamExt;

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
        let typed_args: AdditionArgs = serde_json::from_value(args)?;
        let result = typed_args.left + typed_args.right;
        println!("Tool Call Executed: {}", result);
        Ok(result.into())
    }
}

#[agent(
    name = "streaming_agent",
    description = "You are a math expert and knowledgeable assistant that provides detailed explanations. Respond in a conversational manner.",
    tools = [Addition],
)]
#[derive(Clone)]
pub struct StreamingAgent {}

impl ReActExecutor for StreamingAgent {}

pub async fn run(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    println!("🌊 Agent Streaming Example");

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent = StreamingAgent {};
    let runtime = SingleThreadedRuntime::new(None);

    // Create topic for streaming agent
    let streaming_topic = Topic::<Task>::new("streaming_agent");

    let agent_handler = AgentBuilder::new(agent)
        .with_llm(llm)
        .runtime(runtime.clone())
        .subscribe_topic(streaming_topic.clone())
        .stream(true) // Enable streaming for this agent
        .with_memory(sliding_window_memory)
        .build()
        .await?;

    // Create environment and set up event handling
    let mut environment = Environment::new(None);
    environment.register_runtime(runtime.clone()).await?;

    // let _ = environment.take_event_receiver(None).await?;

    // Start the environment
    let _handle = environment.run();

    let agent = agent_handler.agent;
    let task = Task::new("What is 2 + 2?");

    // Process the stream directly
    let mut stream = agent.run_stream(task).await?;
    println!("🔄 Processing stream tokens...\n");

    // Print each stream token as it arrives
    while let Some(result) = stream.next().await {
        match result {
            Ok(output) => match output {
                TaskResult::Value(val) => match serde_json::from_value::<ReActAgentOutput>(val) {
                    Ok(agent_out) => {
                        if !agent_out.done {
                            println!(
                                "{}",
                                format!("🌊 Streaming Response: {}", agent_out.response).green()
                            );
                        }
                    }
                    Err(e) => {
                        println!("{}", format!("❌ Failed to parse response: {}", e).red());
                    }
                },
                TaskResult::Failure(error) => {
                    println!("{}", format!("❌ Task failed: {}", error).red());
                }
                TaskResult::Aborted => todo!(),
            },
            Err(e) => {
                println!("🔴 {}", format!("Stream error: {}", e).red());
            }
        }
    }

    println!("\n✅ Streaming example completed!");
    Ok(())
}
