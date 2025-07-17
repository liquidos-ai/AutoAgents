use std::fmt::Debug;
use thiserror::Error;

/// Error type for RunnableAgent operations
#[derive(Debug, Error)]
pub enum RunnableAgentError {
    /// Error from the agent executor
    #[error("Agent execution failed: {0}")]
    ExecutorError(String),

    /// Error during task processing
    #[error("Task processing failed: {0}")]
    TaskError(String),

    /// Error when agent is not found
    #[error("Agent not found: {0}")]
    AgentNotFound(uuid::Uuid),

    /// Error during agent initialization
    #[error("Agent initialization failed: {0}")]
    InitializationError(String),

    /// Error when sending events
    #[error("Event send error: {0}")]
    EventSendError(String),

    /// Error from agent state operations
    #[error("Agent state error: {0}")]
    StateError(String),

    /// Error when a feature is not supported
    #[error("Not supported: {0}")]
    NotSupported(String),

    /// Generic error wrapper for any std::error::Error
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl RunnableAgentError {
    /// Create an executor error from any error type
    pub fn executor_error(error: impl std::error::Error) -> Self {
        Self::ExecutorError(error.to_string())
    }

    /// Create a task error
    pub fn task_error(msg: impl Into<String>) -> Self {
        Self::TaskError(msg.into())
    }

    /// Create an event send error
    pub fn event_send_error(error: impl std::error::Error) -> Self {
        Self::EventSendError(error.to_string())
    }
}

/// Specific conversion for tokio mpsc send errors
impl<T> From<tokio::sync::mpsc::error::SendError<T>> for RunnableAgentError
where
    T: Debug + Send + 'static,
{
    fn from(error: tokio::sync::mpsc::error::SendError<T>) -> Self {
        Self::EventSendError(error.to_string())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AgentBuildError {
    #[error("Build Failure")]
    BuildFailure(String),
}
