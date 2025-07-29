use autoagents::async_trait;
/// This Exmaple demonstrages Agent Chaining
use autoagents::core::agent::{
    AgentBuilder, AgentConfig, AgentDeriveT, AgentExecutor, AgentState, ExecutorConfig,
};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::memory::{MemoryProvider, SlidingWindowMemory};
use autoagents::core::protocol::{Event, TaskResult};
use autoagents::core::runtime::{Runtime, SingleThreadedRuntime, Task};
use autoagents::core::tool::ToolT;
use autoagents::llm::chat::{ChatMessage, ChatRole, MessageType};
use autoagents::llm::LLMProvider;
use autoagents_derive::agent;
use colored::*;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

#[agent(
    name = "agent_1",
    description = "You are a math geek and expert in linear algebra",
    tools = [],
)]
pub struct Agent1 {}

#[agent(
    name = "agent_2",
    description = "You are a math professor in linear algebera, Your goal is to review the given content if correct",
    tools = [],
)]
pub struct Agent2 {}

#[async_trait]
impl AgentExecutor for Agent1 {
    type Output = String;
    type Error = Error;

    fn config(&self) -> ExecutorConfig {
        ExecutorConfig { max_turns: 10 }
    }

    async fn execute(
        &self,
        llm: Arc<dyn LLMProvider>,
        _memory: Option<Arc<RwLock<Box<dyn MemoryProvider>>>>,
        _tools: Vec<Box<dyn ToolT>>,
        agent_config: &AgentConfig,
        task: Task,
        _state: Arc<RwLock<AgentState>>,
        tx_event: mpsc::Sender<Event>,
    ) -> Result<Self::Output, Self::Error> {
        let mut messages = vec![ChatMessage {
            role: ChatRole::System,
            message_type: MessageType::Text,
            content: agent_config.description.clone(),
        }];
        let chat_msg = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: task.prompt.clone(),
        };
        messages.push(chat_msg);
        let response = llm
            .chat(&messages, agent_config.output_schema.clone())
            .await
            .unwrap();
        let response_text = response.text().unwrap_or_default();
        tx_event
            .send(Event::PublishMessage {
                topic: "agent_2".into(),
                message: response_text,
            })
            .await
            .unwrap();
        Ok("Agent 1 Responed".into())
    }
}

#[async_trait]
impl AgentExecutor for Agent2 {
    type Output = String;
    type Error = Error;

    fn config(&self) -> ExecutorConfig {
        ExecutorConfig { max_turns: 10 }
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
        let mut messages = vec![ChatMessage {
            role: ChatRole::System,
            message_type: MessageType::Text,
            content: agent_config.description.clone(),
        }];
        let chat_msg = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: task.prompt.clone(),
        };
        messages.push(chat_msg);
        let response = llm
            .chat(&messages, agent_config.output_schema.clone())
            .await
            .unwrap();
        let response_text = response.text().unwrap_or_default();
        Ok(response_text)
    }
}

pub async fn run(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent = Agent1 {};
    let agent2 = Agent2 {};

    let runtime = SingleThreadedRuntime::new(None);

    let _ = AgentBuilder::new(agent)
        .with_llm(llm.clone())
        .runtime(runtime.clone())
        .subscribe_topic("agent_1")
        .with_memory(sliding_window_memory.clone())
        .build()
        .await?;

    let _ = AgentBuilder::new(agent2)
        .with_llm(llm)
        .runtime(runtime.clone())
        .subscribe_topic("agent_2")
        .with_memory(sliding_window_memory)
        .build()
        .await?;

    // Create environment and set up event handling
    let mut environment = Environment::new(None);
    let _ = environment.register_runtime(runtime.clone()).await;

    let receiver = environment.take_event_receiver(None).await?;
    handle_events(receiver);

    runtime
        .publish_message("What is Vector Calculus?".into(), "agent_1".into())
        .await
        .unwrap();

    tokio::select! {
        _ = environment.run() => {
            println!("Environment finished running.");
        }
        _ = tokio::signal::ctrl_c() => {
            println!("Ctrl+C detected. Shutting down...");
            environment.shutdown().await;
        }
    }
    Ok(())
}

fn handle_events(mut event_stream: ReceiverStream<Event>) {
    tokio::spawn(async move {
        while let Some(event) = event_stream.next().await {
            match event {
                Event::NewTask { agent_id: _, task } => {
                    println!("{}", format!("New TASK: {:?}", task).green());
                }
                Event::TaskComplete { result, .. } => {
                    match result {
                        TaskResult::Value(val) => {
                            let agent_out: String = serde_json::from_value(val).unwrap();
                            println!("{}", format!("Thought: {}", agent_out).green());
                        }
                        _ => {
                            //
                        }
                    }
                }
                _ => {
                    //
                }
            }
        }
    });
}
