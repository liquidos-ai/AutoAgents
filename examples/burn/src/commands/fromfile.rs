use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::BasicAgent;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, DirectAgent};
use autoagents::core::error::Error;
use autoagents_burn::model::llama::{Llama3Builder, TinyLlamaBuilder};
use autoagents_derive::{agent, AgentHooks};
use tokio_stream::StreamExt;

#[agent(
    name = "agent",
    description = "You are an Helpful Assistant, Your name is Vikram."
)]
#[derive(Default, Clone, AgentHooks)]
struct SimpleAgent {}

use crate::Model;

pub async fn run_fromfile(
    model_type: Model,
    model_path: String,
    tokenizer_path: String,
    prompt: String,
) -> Result<(), Error> {
    println!(
        "Running FromFile command with {:?} model: {}, tokenizer: {}",
        model_type, model_path, tokenizer_path
    );

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(100));

    let agent_handle = match model_type {
        Model::Tiny => {
            let llm = TinyLlamaBuilder::new()
                .model_path(&model_path)
                .tokenizer_path(&tokenizer_path)
                .max_seq_len(512)
                .temperature(0.7)
                .max_tokens(256)
                .build()
                .expect("Failed to build TinyLlama LLM");
            let agent = BasicAgent::new(SimpleAgent {});
            AgentBuilder::<_, DirectAgent>::new(agent)
                .llm(llm)
                .memory(sliding_window_memory)
                .build()
                .await?
        }
        Model::Llama3 => {
            let llm = Llama3Builder::new()
                .model_path(&model_path)
                .tokenizer_path(&tokenizer_path)
                .max_seq_len(512)
                .temperature(0.7)
                .max_tokens(256)
                .build()
                .expect("Failed to build Llama3 LLM");
            let agent = BasicAgent::new(SimpleAgent {});
            AgentBuilder::<_, DirectAgent>::new(agent)
                .llm(llm)
                .memory(sliding_window_memory)
                .build()
                .await?
        }
    };

    println!("Finished Model Loading!");

    println!("Running agent with task: {}", prompt);

    let mut stream = agent_handle.agent.run_stream(Task::new(&prompt)).await?;
    println!("Response:\n");

    while let Some(result) = stream.next().await {
        match result {
            Ok(output) => {
                print!("{}", format!("{}", output));
            }
            _ => {}
        }
    }

    Ok(())
}
