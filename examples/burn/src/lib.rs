use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::BasicAgent;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, DirectAgent};
use autoagents::core::error::Error;
use autoagents_burn::backend::burn_backend_types::InferenceBackend;
use autoagents_burn::model::llama::{TinyLlama, TinyLlamaBuilder};
use autoagents_derive::{agent, AgentHooks};
use serde_json::Value;
use std::sync::Arc;

#[agent(
    name = "math_agent",
    description = "You are a Math agent",
    tools = [],
)]
#[derive(Default, Clone, AgentHooks)]
pub struct MathAgent {}

async fn run_agent() -> Result<(), Error> {
    let llm = TinyLlamaBuilder::<TinyLlama<InferenceBackend>>::new()
        .model_path("models/TinyLlama-1.1B/tinyllama.mpk")
        .tokenizer_path("models/TinyLlama-1.1B/tokenizer.model")
        .max_seq_len(512)
        .temperature(0.7)
        .max_tokens(256)
        .build()
        .expect("Failed to build LLM");

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent = BasicAgent::new(MathAgent {});
    let agent_handle = AgentBuilder::<_, DirectAgent>::new(agent)
        .llm(llm)
        .memory(sliding_window_memory)
        .build()
        .await?;

    println!("Running basic agent with direct run method");
    let result = agent_handle
        .agent
        .run(Task::new("What is 20 + 10?"))
        .await?;
    println!("Result: {:?}", result);
    Ok(())
}
