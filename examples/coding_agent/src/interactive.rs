use crate::agent::CodingAgent;
use autoagents::core::agent::prebuilt::react::ReActAgentOutput;
use autoagents::core::agent::AgentBuilder;
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::memory::SlidingWindowMemory;
use autoagents::core::protocol::{Event, TaskResult};
use autoagents::core::runtime::{Runtime, SingleThreadedRuntime};
use autoagents::llm::LLMProvider;
use colored::*;
use std::io::{self, Write};
use std::sync::Arc;
use termimad::MadSkin;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

const CODING_TASK_TOPIC: &str = "coding_task";

pub async fn run_interactive_session(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    println!("🚀 Starting Interactive Coding Agent Session");
    println!("💡 Type 'help' for available commands, 'quit' to exit\n");

    // Create memory with larger window for complex tasks
    let memory = Box::new(SlidingWindowMemory::new(30));

    let runtime = SingleThreadedRuntime::new(None);

    // Create the coding agent
    let coding_agent = CodingAgent {};

    // Build the agent
    let _ = AgentBuilder::new(coding_agent)
        .with_llm(llm)
        .runtime(runtime.clone())
        .subscribe_topic(CODING_TASK_TOPIC)
        .with_memory(memory)
        .build()
        .await?;

    // Create environment
    let mut environment = Environment::new(None);

    // Register the agent
    let _ = environment.register_runtime(runtime.clone()).await?;

    let receiver = environment.take_event_receiver(None).await?;
    handle_events(receiver);

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("\n🤖 > ");
        let _ = stdout.flush();

        let mut input = String::new();
        if stdin.read_line(&mut input).is_err() {
            println!("\n❌ Failed to read input");
            continue;
        }
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        environment.run();

        match input.to_lowercase().as_str() {
            "quit" | "exit" | "q" => {
                println!("👋 Goodbye!");
                break;
            }
            "help" | "h" => {
                print_help();
                continue;
            }
            "clear" => {
                // Clear the terminal
                print!("\x1B[2J\x1B[1;1H");
                continue;
            }
            _ => {
                // Process the task
                println!("\n🔄 Processing your request...\n");

                // Add the task
                let _ = runtime
                    .clone()
                    .publish_message(input.into(), CODING_TASK_TOPIC.into())
                    .await?;
            }
        }
    }

    // Shutdown
    environment.shutdown().await;

    Ok(())
}

fn handle_events(mut event_stream: ReceiverStream<Event>) {
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
                Event::ToolCallRequested {
                    tool_name,
                    arguments,
                    ..
                } => {
                    println!(
                        "{}",
                        format!("Tool Call Started: {} with args: {}", tool_name, arguments)
                            .green()
                    );
                }
                Event::ToolCallCompleted {
                    tool_name, result, ..
                } => {
                    println!(
                        "{}",
                        format!("Tool Call Completed: {} - Result: {:?}", tool_name, result)
                            .green()
                    );
                }
                Event::TaskComplete { result, .. } => {
                    match result {
                        TaskResult::Value(val) => {
                            let agent_out: ReActAgentOutput = serde_json::from_value(val).unwrap();
                            // Run the specific task
                            let skin = MadSkin::default();
                            println!("📝 Agent Response:\n");
                            skin.print_text(&agent_out.response);
                        }
                        _ => {
                            //
                        }
                    }
                }
                Event::TurnStarted {
                    turn_number,
                    max_turns,
                } => {
                    println!(
                        "{}",
                        format!("Turn {}/{} started", turn_number + 1, max_turns).green()
                    );
                }
                Event::TurnCompleted {
                    turn_number,
                    final_turn,
                } => {
                    println!(
                        "{}",
                        format!(
                            "Turn {} completed{}",
                            turn_number + 1,
                            if final_turn { " (final)" } else { "" }
                        )
                        .green()
                    );
                }
                _ => {
                    println!("📡 Event: {:?}", event);
                }
            }
        }
    });
}

fn print_help() {
    println!(
        r#"
📚 Interactive Coding Agent Help

This agent uses the ReAct (Reasoning + Acting) pattern to solve coding tasks.

Commands:
  help, h     - Show this help message
  clear       - Clear the terminal
  quit, q     - Exit the session
"#
    );
}
