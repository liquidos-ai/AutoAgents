use crate::utils::handle_events;
/// This example demonstrates Agent Chaining using the runtime architecture
use autoagents::async_trait;
use autoagents::core::actor::Topic;
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{
    ActorAgent, AgentBuilder, AgentExecutor, Context, EventHelper, ExecutorConfig,
};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::runtime::{SingleThreadedRuntime, TypedRuntime};
use autoagents::core::tool::ToolT;
use autoagents::llm::chat::{ChatMessage, ChatRole, MessageType};
use autoagents::llm::LLMProvider;
use autoagents_derive::{agent, AgentHooks};
use serde_json::Value;
use std::sync::Arc;

#[agent(
    name = "agent_1",
    description = "You are a math geek and expert in linear algebra",
    tools = [],
)]
#[derive(AgentHooks)]
pub struct Agent1 {}

#[agent(
    name = "agent_2",
    description = "You are a math professor in linear algebra, Your goal is to review the given content if correct",
    tools = [],
)]
#[derive(AgentHooks)]
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
        task: &Task,
        context: Arc<Context>,
    ) -> Result<Self::Output, Self::Error> {
        println!("Agent 1 Executing");

        let tx_event = context.tx().ok();
        EventHelper::send_task_started(
            &tx_event,
            task.submission_id,
            context.config().id,
            task.prompt.clone(),
        )
        .await;

        let mut messages = vec![ChatMessage {
            role: ChatRole::System,
            message_type: MessageType::Text,
            content: context.config().description.clone(),
        }];

        let chat_msg = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: task.prompt.clone(),
        };
        messages.push(chat_msg);
        let response = context
            .llm()
            .chat(&messages, None, context.config().output_schema.clone())
            .await?;
        let response_text = response.text().unwrap_or_default();
        context
            .publish(Topic::<Task>::new("agent_2"), Task::new(response_text))
            .await?;
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
        task: &Task,
        context: Arc<Context>,
    ) -> Result<Self::Output, Self::Error> {
        println!("Agent 2 Executing");

        let tx_event = context.tx().ok();
        EventHelper::send_task_started(
            &tx_event,
            task.submission_id,
            context.config().id,
            task.prompt.clone(),
        )
        .await;

        let mut messages = vec![ChatMessage {
            role: ChatRole::System,
            message_type: MessageType::Text,
            content: context.config().description.clone(),
        }];

        let chat_msg = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: task.prompt.clone(),
        };
        messages.push(chat_msg);
        let response = context
            .llm()
            .chat(&messages, None, context.config().output_schema.clone())
            .await?;
        let response_text = response.text().unwrap_or_default();
        println!("Agent 2 respond: {}", response_text);
        Ok(response_text)
    }
}

pub async fn run(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let topic1 = Topic::<Task>::new("agent_1");
    let topic2 = Topic::<Task>::new("agent_2");

    let agent1 = Agent1 {};
    let agent2 = Agent2 {};

    let runtime = SingleThreadedRuntime::new(None);

    let _ = AgentBuilder::<_, ActorAgent>::new(agent1)
        .llm(llm.clone())
        .runtime(runtime.clone())
        .subscribe(topic1.clone())
        .memory(sliding_window_memory.clone())
        .build()
        .await?;

    let _ = AgentBuilder::<_, ActorAgent>::new(agent2)
        .llm(llm)
        .runtime(runtime.clone())
        .subscribe(topic2.clone())
        .memory(sliding_window_memory)
        .build()
        .await?;

    // Create environment and set up event handling
    let mut environment = Environment::new(None);
    let _ = environment.register_runtime(runtime.clone()).await;

    let receiver = environment.take_event_receiver(None).await?;
    handle_events(receiver);

    runtime
        .publish(&topic1, Task::new("What is Vector Calculus?"))
        .await?;

    runtime
        .publish(&topic1, Task::new("What is Linear Algebra?"))
        .await?;

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
