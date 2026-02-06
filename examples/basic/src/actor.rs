use crate::utils::handle_events;
use autoagents::async_trait;
use autoagents::core::actor::{ActorMessage, CloneableMessage, Topic};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::ractor::concurrency::sleep;
use autoagents::core::ractor::{Actor, ActorProcessingErr, ActorRef};
use autoagents::core::runtime::{SingleThreadedRuntime, TypedRuntime};
use autoagents::llm::LLMProvider;
use autoagents::protocol::ActorID;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

/// Simple message type for basic actor communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleMessage {
    pub content: usize,
}

impl CloneableMessage for SimpleMessage {}

impl ActorMessage for SimpleMessage {}

/// A simple actor that can receive and process SimpleMessage
#[derive(Debug)]
pub struct SimpleActor {
    pub id: ActorID,
    pub name: String,
}

impl SimpleActor {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
        }
    }
}

#[async_trait]
impl Actor for SimpleActor {
    type Msg = SimpleMessage;
    type State = ();
    type Arguments = ();

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        _args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        println!("🎭 Actor '{}' started with ID: {}", self.name, self.id);
        Ok(())
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        _state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        println!(
            "📨 Actor '{}' received message: '{}'",
            self.name, message.content
        );
        //Simulate some processing
        sleep(Duration::from_secs(1)).await;
        Ok(())
    }
}

/// Basic example showing new messaging system
pub async fn run(_llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    println!("🚀 Basic Actor Messaging Example");

    // Create runtime
    let runtime = SingleThreadedRuntime::new(Some(10));

    // Create environment
    let mut environment = Environment::new(None);
    environment.register_runtime(runtime.clone()).await?;

    // Set up event handling
    let receiver = environment.take_event_receiver(None).await?;
    handle_events(receiver);

    // Create actors
    let actor1 = SimpleActor::new("Actor1");
    let actor2 = SimpleActor::new("Actor2");

    let actor1_ref = Actor::spawn(None, actor1, ())
        .await
        .map_err(|e| Error::CustomError(e.to_string()))?
        .0;
    let actor2_ref = Actor::spawn(None, actor2, ())
        .await
        .map_err(|e| Error::CustomError(e.to_string()))?
        .0;

    // Create topics for pub/sub messaging
    let general_topic = Topic::<SimpleMessage>::new("general");
    let announcements_topic = Topic::<SimpleMessage>::new("announcements");

    // Subscribe actors to topics
    runtime
        .subscribe(&general_topic, actor1_ref.clone())
        .await?;
    runtime
        .subscribe(&general_topic, actor2_ref.clone())
        .await?;
    runtime
        .subscribe(&announcements_topic, actor1_ref.clone())
        .await?;

    // Topic publishing example
    println!("\n📡 Publishing to general topic...");
    let general_message = SimpleMessage { content: 0 };

    runtime
        .publish(&general_topic, general_message.clone())
        .await?;
    runtime
        .publish(&general_topic, SimpleMessage { content: 1 })
        .await?;

    // Announcement example
    println!("\n📢 Publishing announcement...");
    let announcement = SimpleMessage { content: 2 };

    runtime.publish(&announcements_topic, announcement).await?;

    //Send Direct Message
    runtime.send_message(general_message, actor1_ref).await?;

    let _ = environment.run().await;

    println!("\n✅ All messages sent successfully!");

    Ok(())
}
