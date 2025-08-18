use crate::agent::{RunnableAgentError};
use crate::error::Error;
use crate::protocol::{ActorID, Event, RuntimeID};
use async_trait::async_trait;
use ractor::ActorRef;
use serde_json::Value;
use std::fmt::Debug;
use tokio::sync::mpsc::error::SendError;
use tokio::task::JoinError;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;
use crate::actor::{ActorMessage};

pub(crate) mod manager;

#[cfg(feature = "single_threaded")]
mod single_threaded;
#[cfg(feature = "single_threaded")]
pub use single_threaded::SingleThreadedRuntime;
#[cfg(feature = "single_threaded")]
use single_threaded::InternalEvent;


/// Error types for Session operations
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("Agent not found: {0}")]
    AgentNotFound(Uuid),

    #[error("No task set for agent: {0}")]
    NoTaskSet(Uuid),

    #[error("Task is None")]
    EmptyTask,

    #[error("Task join error: {0}")]
    TaskJoinError(#[from] JoinError),

    #[error("Event error: {0}")]
    EventError(#[from] SendError<Event>),

    #[cfg(feature = "single_threaded")]
    #[error("Internal Event error: {0}")]
    InternalEventError(#[from] SendError<InternalEvent>),

    #[error("RunnableAgent error: {0}")]
    RunnableAgentError(#[from] RunnableAgentError),
}

#[async_trait]
pub trait Runtime: Send + Sync + 'static + Debug {
    fn id(&self) -> RuntimeID;
    async fn send_message(&self, message: String, actor_id: ActorID) -> Result<(), Error>;
    async fn publish_message(&self, message: String, topic: String) -> Result<(), Error>;
    async fn subscribe(&self, actor_id: ActorID, topic: String) -> Result<(), Error>;
    async fn register_agent(&self, id: Uuid, agent: ActorRef<ActorMessage>) -> Result<(), Error>;
    async fn take_event_receiver(&self) -> Option<ReceiverStream<Event>>;
    async fn run(&self) -> Result<(), Error>;
    async fn stop(&self) -> Result<(), Error>;
}
