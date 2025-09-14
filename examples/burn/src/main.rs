use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::BasicAgent;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, DirectAgent};
use autoagents::core::error::Error;
use autoagents_burn::model::llama::TinyLlamaBuilder;
use autoagents_derive::{agent, AgentHooks};
use serde_json::Value;
use tokio_stream::StreamExt;

#[agent(name = "math_agent", description = "You are a Math agent")]
#[derive(Default, Clone, AgentHooks)]
pub struct MathAgent {}

#[tokio::main]
async fn main() -> Result<(), Error> {
    // Build TinyLlama model
    let llm = TinyLlamaBuilder::new()
        .model_path("./examples/burn/model/TinyLlama-1.1B/model.mpk") // Path to your model file
        .tokenizer_path("./examples/burn/model/TinyLlama-1.1B/tokenizer.json") // Path to your tokenizer file
        .max_seq_len(512)
        .temperature(0.7)
        .max_tokens(256)
        .build()
        .expect("Failed to build LLM");

    println!("Finished Model Loading!");

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent = BasicAgent::new(MathAgent {});
    let agent_handle = AgentBuilder::<_, DirectAgent>::new(agent)
        .llm(llm)
        .memory(sliding_window_memory)
        .build()
        .await?;

    println!("Running basic agent with direct run method");
    let mut stream = agent_handle
        .agent
        .run_stream(Task::new("Tell me a poem?"))
        .await?;

    while let Some(result) = stream.next().await {
        match result {
            Ok(output) => {
                println!("{}", format!("🌊 Streaming Response: {}", output));
            }
            _ => {
                //
            }
        }
    }

    Ok(())
}
