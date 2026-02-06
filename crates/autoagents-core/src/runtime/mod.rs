use crate::actor::{AnyActor, CloneableMessage, Transport};
use async_trait::async_trait;
use autoagents_protocol::{Event, RuntimeID};
use ractor::ActorRef;
use std::any::{Any, TypeId};
use std::fmt::Debug;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::SendError;
use tokio::task::JoinError;

pub(crate) mod manager;
mod single_threaded;
use crate::actor::Topic;
use crate::utils::BoxEventStream;
pub use single_threaded::SingleThreadedRuntime;

/// Configuration for runtime instances.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub queue_size: Option<usize>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            queue_size: Some(100),
        }
    }
}

/// Error types for runtime operations and message routing.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("Send Message Error: {0}")]
    SendMessage(String),

    #[error("TopicTypeMismatch")]
    TopicTypeMismatch(String, TypeId),

    #[error("Join Error: {0}")]
    JoinError(JoinError),

    #[error("Event error: {0}")]
    EventError(#[from] SendError<Event>),
}

/// Abstract runtime that manages actor subscriptions, pub/sub delivery, and
/// emission of protocol events. Implementations can provide different threading
/// or transport strategies.
#[async_trait]
pub trait Runtime: Send + Sync {
    fn id(&self) -> RuntimeID;

    async fn subscribe_any(
        &self,
        topic_name: &str,
        topic_type: TypeId,
        actor: Arc<dyn AnyActor>,
    ) -> Result<(), RuntimeError>;

    async fn publish_any(
        &self,
        topic_name: &str,
        topic_type: TypeId,
        message: Arc<dyn Any + Send + Sync>,
    ) -> Result<(), RuntimeError>;

    /// Local event processing sender. Agents receive this and emit protocol
    /// `Event`s through it. The runtime is responsible for forwarding them to
    /// the owning `Environment`.
    fn tx(&self) -> mpsc::Sender<Event>;
    async fn transport(&self) -> Arc<dyn Transport>;
    async fn take_event_receiver(&self) -> Option<BoxEventStream<Event>>;
    /// Subscribe to runtime protocol events without consuming the receiver.
    async fn subscribe_events(&self) -> BoxEventStream<Event>;
    /// Run the runtime event loop and process internal messages until stopped.
    async fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    /// Request shutdown of the runtime.
    async fn stop(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

/// Type-safe convenience layer over `Runtime` for strongly-typed topics and
/// direct messaging to actors.
#[async_trait]
pub trait TypedRuntime: Runtime {
    async fn subscribe<M>(&self, topic: &Topic<M>, actor: ActorRef<M>) -> Result<(), RuntimeError>
    where
        M: CloneableMessage + 'static,
    {
        let arc_actor = Arc::new(actor) as Arc<dyn AnyActor>;
        self.subscribe_any(topic.name(), TypeId::of::<M>(), arc_actor)
            .await
    }

    async fn publish<M>(&self, topic: &Topic<M>, message: M) -> Result<(), RuntimeError>
    where
        M: CloneableMessage + 'static,
    {
        let arc_msg = Arc::new(message) as Arc<dyn Any + Send + Sync>;
        self.publish_any(topic.name(), TypeId::of::<M>(), arc_msg)
            .await
    }

    async fn send_message<M: CloneableMessage + 'static>(
        &self,
        message: M,
        addr: ActorRef<M>,
    ) -> Result<(), RuntimeError> {
        addr.cast(message)
            .map_err(|e| RuntimeError::SendMessage(e.to_string()))
    }
}

// Auto-implement TypedRuntime for all Runtime implementations
impl<T: Runtime + ?Sized> TypedRuntime for T {}
