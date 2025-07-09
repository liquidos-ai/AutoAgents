use autoagents_core::agent::executor::AgentExecutor;
use autoagents_core::agent::runnable::AgentState;
use autoagents_core::agent::types::simple::SimpleAgentBuilder;
use autoagents_core::protocol::Event;
use autoagents_core::session::Task;
use autoagents_llm::backends::openai::OpenAI;
use autoagents_llm::builder::LLMBuilder;
use futures::StreamExt;
use std::io::{stdout, Write};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {

    // Initialize the LLM
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let llm = LLMBuilder::<OpenAI>::new()
        .api_key(api_key)
        .model("gpt-4o")
        .build()?;

    // Create a simple agent
    let agent = SimpleAgentBuilder::new(
        "Streaming Agent",
        "An agent that demonstrates streaming capabilities.",
        "You are a helpful assistant that provides concise answers.",
    )
    .with_llm(llm)
    .build()?;

    // Create a task
    let task = Task::new(
        "Tell me a future of auto agents that wrote in rust",
        None,
    );

    println!("--- Starting Agent Stream ---");

    // Get the stream from the executor
    let executor = agent.inner();
    let agent_state = Arc::new(RwLock::new(AgentState::new()));
    let (tx_event, _rx_event) = mpsc::channel(100);

    let mut stream = executor.stream(
        agent.llm(),
        agent.memory(),
        agent.tools(),
        &agent.agent_config(),
        task,
        agent_state,
        tx_event,
    );

    // Process the stream
    while let Some(event_result) = stream.next().await {
        match event_result {
            Ok(event) => match event {
                Event::Token(text) => {
                    print!("{}", text);
                    stdout().flush()?;
                }
                _ => {}
            },
            Err(e) => {
                eprintln!("\nAn error occurred in the stream: {}", e);
                break;
            }
        }
    }

    // Print a final newline to move to the next line in the console
    println!();

    println!("--- Agent Stream Finished ---");

    Ok(())
}
