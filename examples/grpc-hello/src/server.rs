use autoagents::{
    async_trait,
    core::{
        agent::{
            AgentBuilder, AgentConfig, AgentDeriveT, AgentExecutor, AgentState, ExecutorConfig,
        },
        environment::Environment,
        error::Error,
        memory::{MemoryProvider, SlidingWindowMemory},
        protocol::{Event, TaskResult},
        runtime::{GrpcRuntime, GrpcRuntimeConfig, Runtime, Task},
    },
    init_logging,
    llm::{
        backends::openai::OpenAI,
        builder::LLMBuilder,
        chat::{ChatMessage, ChatRole, MessageType},
        LLMProvider, ToolT,
    },
};
use autoagents_derive::agent;
use colored::*;
use serde_json::Value;
use std::{net::SocketAddr, sync::Arc};
use tokio::sync::{mpsc, RwLock};
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

/// Simple greeting agent
#[agent(
    name = "greeter",
    description = "I greet people and answer simple questions",
    tools = []
)]
pub struct GreeterAgent {}

#[async_trait]
impl AgentExecutor for GreeterAgent {
    type Output = String;
    type Error = Error;

    fn config(&self) -> ExecutorConfig {
        ExecutorConfig { max_turns: 1 }
    }

    async fn execute(
        &self,
        llm: Arc<dyn LLMProvider>,
        _memory: Option<Arc<RwLock<Box<dyn MemoryProvider>>>>,
        _tools: Vec<Box<dyn ToolT>>,
        agent_config: &AgentConfig,
        task: Task,
        _state: Arc<RwLock<AgentState>>,
        _tx_event: mpsc::Sender<Event>,
    ) -> Result<Self::Output, Self::Error> {
        // Prepare messages for the LLM
        let mut messages = vec![
            ChatMessage {
                role: ChatRole::System,
                message_type: MessageType::Text,
                content: agent_config.description.clone(),
            },
            ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: task.prompt.clone(),
            },
        ];

        // Get response from LLM
        let response = llm
            .chat(&messages, agent_config.output_schema.clone())
            .await?;

        let response_text = response.text().unwrap_or_default();
        Ok(response_text)
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    init_logging();

    println!("{}", "Starting gRPC Hello World Example".green().bold());
    println!("{}", "=================================".green());

    // Configure gRPC runtime
    let addr: SocketAddr = "127.0.0.1:50051".parse().unwrap();
    let config = GrpcRuntimeConfig {
        bind_addr: addr,
        max_message_size: 4 * 1024 * 1024,
        max_connections: 100,
        channel_buffer_size: 1000,
    };

    println!(
        "{}",
        format!("Creating gRPC runtime on {}...", addr).yellow()
    );

    // Create gRPC runtime
    let runtime = GrpcRuntime::new(config).await?;

    // Initialize LLM
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    if api_key.is_empty() {
        println!("{}", "Warning: OPENAI_API_KEY not set!".red());
    }

    let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(api_key)
        .model("gpt-4o-mini")
        .max_tokens(256)
        .temperature(0.7)
        .build()
        .expect("Failed to build LLM");

    // Create and register greeter agent
    let greeter = GreeterAgent {};
    let memory = Box::new(SlidingWindowMemory::new(10));

    let _ = AgentBuilder::new(greeter)
        .with_llm(llm)
        .runtime(runtime.clone())
        .subscribe_topic("greetings")
        .subscribe_topic("questions")
        .with_memory(memory)
        .build()
        .await?;

    println!("{}", "✅ Greeter agent registered!".green());

    // Create environment
    let mut environment = Environment::new(None);
    let _ = environment.register_runtime(runtime.clone()).await;

    // Set up event handling
    let receiver = environment.take_event_receiver(None).await;
    handle_events(receiver);

    println!("{}", "\n🚀 gRPC server is ready!".green().bold());
    println!("{}", "Listening for connections...".cyan());
    println!("\n{}", "Topics:".blue());
    println!("  - greetings: Send greeting requests");
    println!("  - questions: Ask simple questions");
    println!("\n{}", "Press Ctrl+C to stop the server".yellow());

    // Run the environment
    tokio::select! {
        _ = environment.run() => {
            println!("Environment finished running.");
        }
        _ = tokio::signal::ctrl_c() => {
            println!("\n{}", "Shutting down server...".yellow());
            environment.shutdown().await;
        }
    }

    println!("{}", "Server stopped.".red());
    Ok(())
}

fn handle_events(event_stream: Option<ReceiverStream<Event>>) {
    if let Some(mut event_stream) = event_stream {
        tokio::spawn(async move {
            while let Some(event) = event_stream.next().await {
                match event {
                    Event::NewTask { agent_id, task } => {
                        println!("{}", format!("📋 New Task - Agent: {:?}", agent_id).blue());
                        println!("   Request: {}", task.prompt);
                    }
                    Event::TaskStarted {
                        agent_id,
                        task_description,
                        ..
                    } => {
                        println!(
                            "{}",
                            format!("▶️  Processing - Agent: {:?}", agent_id).green()
                        );
                    }
                    Event::TaskComplete { result, .. } => match result {
                        TaskResult::Value(val) => {
                            println!("{}", "✅ Task Complete!".green().bold());
                            if let Some(response) = val.as_str() {
                                println!("   Response: {}", response);
                            } else {
                                println!(
                                    "   Response: {}",
                                    serde_json::to_string_pretty(&val).unwrap_or_default()
                                );
                            }
                        }
                        TaskResult::Failure(err) => {
                            println!("{}", format!("❌ Task Failed: {}", err).red());
                        }
                        TaskResult::Aborted => {
                            println!("{}", "⚠️ Task Aborted".yellow());
                        }
                    },
                    Event::PublishMessage { topic, message } => {
                        println!(
                            "{}",
                            format!("📢 Message on '{}': {}", topic, message).cyan()
                        );
                    }
                    _ => {
                        // Log other events at debug level
                        log::debug!("Event: {:?}", event);
                    }
                }
            }
        });
    }
}
