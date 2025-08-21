use crate::agent::CodingAgent;
use autoagents::core::agent::prebuilt::executor::{ReActAgentOutput};
use autoagents::core::agent::AgentBuilder;
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::protocol::{Event, TaskResult};
use autoagents::core::runtime::{SingleThreadedRuntime, RuntimeError, TypedRuntime};
use autoagents::core::actor::{ActorMessage, CloneableMessage, Topic};
use autoagents::core::agent::task::Task;
use autoagents::llm::LLMProvider;
use colored::*;
use std::io::{self, Write};
use std::sync::Arc;
use termimad::MadSkin;
use termimad::RelativePosition::Top;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

const CODING_TASK_TOPIC: &str = "coding_task";

pub async fn run_interactive_session(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    println!("🚀 Starting Interactive Coding Agent Session");
    println!("💡 Type 'help' for available commands, 'quit' to exit\n");

    // Create memory with larger window for complex tasks
    let memory = Box::new(SlidingWindowMemory::new(30));

    let runtime = SingleThreadedRuntime::new(None);

    // Create topic for coding tasks
    let coding_topic = Topic::<Task>::new(CODING_TASK_TOPIC);

    // Create the coding agent
    let coding_agent = CodingAgent {};

    // Build the agent
    let agent_handle = AgentBuilder::new(coding_agent)
        .with_llm(llm)
        .runtime(runtime.clone())
        .subscribe_topic(coding_topic.clone())
        .with_memory(memory)
        .build()
        .await?;

    // Create environment
    let mut environment = Environment::new(None);

    // Register the runtime
    environment.register_runtime(runtime.clone()).await?;

    let receiver = environment.take_event_receiver(None).await?;
    handle_events(receiver);

    // Start the environment in the background
    let _handle = environment.run();

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

                // Create task and send using the new messaging system
                let task = Task::new(input);
                
                // Publish to topic for all subscribers
                let any_topic = Topic::<Task>::new("test");
                runtime.publish(&any_topic, task).await?;

                // Give some time for processing
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        }
    }

    Ok(())
}

fn handle_events(mut event_stream: ReceiverStream<Event>) {
    tokio::spawn(async move {
        while let Some(event) = event_stream.next().await {
            match event {
                Event::TaskStarted {
                    actor_id,
                    task_description,
                    ..
                } => {
                    println!(
                        "{}",
                        format!(
                            "🎯 Task Started - Agent: {:?}\n   📝 Task: {}",
                            actor_id, task_description
                        )
                        .cyan()
                    );
                }
                Event::ToolCallRequested {
                    tool_name,
                    arguments,
                    ..
                } => {
                    println!(
                        "{}",
                        format!("🔧 Tool Call: {} with args: {}", tool_name, arguments)
                            .yellow()
                    );
                }
                Event::ToolCallCompleted {
                    tool_name, result, ..
                } => {
                    println!(
                        "{}",
                        format!("✅ Tool Completed: {} - Result: {:?}", tool_name, result)
                            .yellow()
                    );
                }
                Event::TaskComplete { result, .. } => {
                    match result {
                        TaskResult::Value(val) => {
                            match serde_json::from_value::<ReActAgentOutput>(val) {
                                Ok(agent_out) => {
                                    let skin = MadSkin::default();
                                    println!("\n📝 Agent Response:");
                                    println!("{}", "─".repeat(50).blue());
                                    skin.print_text(&agent_out.response);
                                    println!("{}", "─".repeat(50).blue());
                                }
                                Err(e) => {
                                    println!("{}", format!("❌ Failed to parse response: {}", e).red());
                                }
                            }
                        }
                        TaskResult::Failure(error) => {
                            println!("{}", format!("❌ Task failed: {}", error).red());
                        }
                        TaskResult::Aborted => {
                            println!("{}", "🚫 Task aborted".yellow());
                        }
                    }
                }
                Event::TurnStarted {
                    turn_number,
                    max_turns,
                } => {
                    println!(
                        "{}",
                        format!("🔄 Turn {}/{} started", turn_number + 1, max_turns).blue()
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
                        .blue()
                    );
                }
                _ => {
                    // Handle other events silently or with debug output
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
It has access to various coding tools for file operations, code execution, and more.

Example tasks you can try:
  - "Write a Python function to calculate fibonacci numbers"
  - "Create a simple HTTP server in Python"
  - "Help me debug this code: [paste your code]"
  - "Explain how quicksort works and implement it in Rust"
  - "Create a React component for a todo list"

Commands:
  help, h     - Show this help message
  clear       - Clear the terminal
  quit, q     - Exit the session

💡 The agent can read files, write code, execute commands, and explain concepts!
"#
    );
}