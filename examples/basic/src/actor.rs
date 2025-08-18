use autoagents::async_trait;
/// This Exmaple demonstrages Agent Chaining
use autoagents::core::agent::{AgentBuilder, AgentConfig, AgentDeriveT, AgentExecutor, AgentState, Context, ExecutorConfig};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::agent::memory::{MemoryProvider, SlidingWindowMemory};
use autoagents::core::protocol::{Event, TaskResult};
use autoagents::core::runtime::{Runtime, SingleThreadedRuntime};
use autoagents::core::tool::ToolT;
use autoagents::llm::chat::{ChatMessage, ChatRole, MessageType};
use autoagents::llm::LLMProvider;
use autoagents_derive::agent;
use colored::*;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use autoagents::core::actor::{ActorMessage, ActorTask};
use autoagents::core::agent::task::Task;
use autoagents::core::ractor::{Actor, ActorProcessingErr, ActorRef};
use uuid::Uuid;

#[agent(
    name = "agent_2",
    description = "You are a math professor in linear algebra, Your goal is to review the given content if correct",
    tools = [],
)]
pub struct Agent2 {}

pub struct Actor1{}
#[async_trait]
impl Actor for Actor1 {
    type Msg = ActorMessage;
    type State = ();
    type Arguments = ();

    async fn pre_start(&self, myself: ActorRef<Self::Msg>, args: Self::Arguments) -> Result<Self::State, ActorProcessingErr> {
        println!("Actor Prestart");
        Ok(())
    }

    async fn handle(&self, myself: ActorRef<Self::Msg>, message: Self::Msg, state: &mut Self::State) -> Result<(), ActorProcessingErr> {
        println!("Actor Handle");
        let tx = message.tx.clone();
        let task = message.task;
        let task = task
            .as_any()
            .downcast_ref::<Task>()
            .expect("Expected Task type")
            .clone();
        tx.send(Event::PublishMessage {
            topic: "agent_2".into(),
            message: task.prompt,
        })
            .await
            .unwrap();
        Ok(())
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
        context: Context,
    ) -> Result<Self::Output, Self::Error> {
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
        let response = context.llm()
            .chat(&messages, context.config().output_schema.clone())
            .await
            .unwrap();
        let response_text = response.text().unwrap_or_default();
        Ok(response_text)
    }
}

pub async fn run(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));
    let actor = Actor::spawn(None, Actor1{}, ()).await.unwrap();
    let agent2 = Agent2 {};

    let runtime = SingleThreadedRuntime::new(None);

    let actor_id = Uuid::new_v4();
    let _ = runtime.register_agent(actor_id, actor.0).await?;
    let _ = runtime.subscribe(actor_id, "actor".into()).await?;

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
        .publish_message("What is Vector Calculus?".into(), "actor".into())
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

fn handle_events(mut event_stream: ReceiverStream<Event>) {
    tokio::spawn(async move {
        while let Some(event) = event_stream.next().await {
            match event {
                Event::NewTask { actor_id: _, task } => {
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
