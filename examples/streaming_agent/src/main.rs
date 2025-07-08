use autoagents_core::agent::runnable::IntoRunnable;
use autoagents_core::agent::types::simple::SimpleAgentBuilder;
use autoagents_core::protocol::Event;
use autoagents_core::session::Task;
use autoagents_llm::backends::openai::OpenAI;
use autoagents_llm::builder::LLMBuilder;
use futures::StreamExt;
use std::io::{stdout, Write};
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

    // Convert to a runnable agent
    let runnable_agent = agent.into_runnable();

    // Create a task
    let task = Task::new(
        "Tell me a short story about a robot who learns to dream.",
        None,
    );

    println!("--- Starting Agent Stream ---");

    // Get the stream
    let mut stream = runnable_agent.stream(task);

    // Process the stream
    while let Some(event_result) = stream.next().await {
        match event_result {
            Ok(event) => {
                if let Event::Token(token) = event {
                    print!("{}", token);
                    stdout().flush().unwrap();
                }
            }
            Err(e) => {
                eprintln!("An error occurred in the stream: {}", e);
                break;
            }
        }
    }

    // Add a final newline for clean output
    println!();

    println!("--- Agent Stream Finished ---");

    Ok(())
}
