use crate::agent::{RunnableAgent, RunnableAgentError};
use crate::error::Error;
use crate::protocol::{AgentID, Event, RuntimeID, SubmissionId};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use single_threaded::InternalEvent;
use std::fmt::Debug;
use std::sync::Arc;
use tokio::sync::mpsc::error::SendError;
use tokio::task::JoinError;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

pub(crate) mod manager;
mod single_threaded;
pub use single_threaded::SingleThreadedRuntime;

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

    #[error("Internal Event error: {0}")]
    InternalEventError(#[from] SendError<InternalEvent>),

    #[error("RunnableAgent error: {0}")]
    RunnableAgentError(#[from] RunnableAgentError),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub prompt: String,
    pub submission_id: SubmissionId,
    pub completed: bool,
    pub result: Option<Value>,
    agent_id: Option<AgentID>,
}

impl Task {
    pub fn new<T: Into<String>>(task: T, agent_id: Option<AgentID>) -> Self {
        Self {
            prompt: task.into(),
            submission_id: Uuid::new_v4(),
            completed: false,
            result: None,
            agent_id,
        }
    }
}

#[async_trait]
pub trait Runtime: Send + Sync + 'static + Debug {
    fn id(&self) -> RuntimeID;
    async fn send_message(&self, message: String, agent_id: AgentID) -> Result<(), Error>;
    async fn publish_message(&self, message: String, topic: String) -> Result<(), Error>;
    async fn subscribe(&self, agent_id: AgentID, topic: String) -> Result<(), Error>;
    async fn register_agent(&self, agent: Arc<dyn RunnableAgent>) -> Result<(), Error>;
    async fn take_event_receiver(&self) -> Option<ReceiverStream<Event>>;
    async fn run(&self) -> Result<(), Error>;
    async fn stop(&self) -> Result<(), Error>;
}
