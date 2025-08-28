#[cfg(not(target_arch = "wasm32"))]
use ractor::SpawnErr;
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
    #[error("Failed to send event: {0}")]
    EventSendError(String),

    /// Error from agent state operations
    #[error("Agent state error: {0}")]
    StateError(String),

    /// Error from agent state operations
    #[error("Downcast task error")]
    DowncastTaskError,

    /// Error during serialization/deserialization
    #[error("Serialization error: {0}")]
    SerializationError(String),

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
#[cfg(not(target_arch = "wasm32"))]
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

    #[cfg(not(target_arch = "wasm32"))]
    #[error("SpawnError")]
    SpawnError(#[from] SpawnErr),
}

impl AgentBuildError {
    pub fn build_failure(msg: impl Into<String>) -> Self {
        Self::BuildFailure(msg.into())
    }
}

#[derive(Error, Debug)]
pub enum AgentResultError {
    #[error("No output available in result")]
    NoOutput,

    #[error("Failed to deserialize executor output: {0}")]
    DeserializationError(#[from] serde_json::Error),

    #[error("Agent output extraction error: {0}")]
    AgentOutputError(String),
}

impl AgentResultError {
    pub fn agent_output_error(msg: impl Into<String>) -> Self {
        Self::AgentOutputError(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[test]
    fn test_runnable_agent_error_display() {
        let error = RunnableAgentError::ExecutorError("Test error".to_string());
        assert_eq!(error.to_string(), "Agent execution failed: Test error");

        let error = RunnableAgentError::TaskError("Task failed".to_string());
        assert_eq!(error.to_string(), "Task processing failed: Task failed");

        let error = RunnableAgentError::AgentNotFound(uuid::Uuid::new_v4());
        assert!(error.to_string().contains("Agent not found:"));
    }

    #[test]
    fn test_runnable_agent_error_constructors() {
        let error = RunnableAgentError::executor_error(std::io::Error::other("IO error"));
        assert!(matches!(error, RunnableAgentError::ExecutorError(_)));

        let error = RunnableAgentError::task_error("Custom task error");
        assert!(matches!(error, RunnableAgentError::TaskError(_)));

        let error = RunnableAgentError::event_send_error(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "Send failed",
        ));
        assert!(matches!(error, RunnableAgentError::EventSendError(_)));
    }

    #[tokio::test]
    async fn test_runnable_agent_error_from_mpsc_send_error() {
        let (_tx, rx) = mpsc::channel::<String>(1);
        drop(rx); // Close receiver to cause send error

        let (tx, _rx) = mpsc::channel::<String>(1);
        drop(tx); // This will cause an error when we try to send

        // Create a send error manually for testing
        let result: Result<(), mpsc::error::SendError<String>> =
            Err(mpsc::error::SendError("test message".to_string()));

        if let Err(send_error) = result {
            let agent_error: RunnableAgentError = send_error.into();
            assert!(matches!(agent_error, RunnableAgentError::EventSendError(_)));
        }
    }

    #[test]
    fn test_agent_build_error_display() {
        let error = AgentBuildError::BuildFailure("Failed to build agent".to_string());
        assert_eq!(error.to_string(), "Build Failure");

        let error = AgentBuildError::build_failure("Custom build failure");
        assert!(matches!(error, AgentBuildError::BuildFailure(_)));
    }

    #[test]
    fn test_agent_result_error_display() {
        let error = AgentResultError::NoOutput;
        assert_eq!(error.to_string(), "No output available in result");

        let error = AgentResultError::AgentOutputError("Custom output error".to_string());
        assert_eq!(
            error.to_string(),
            "Agent output extraction error: Custom output error"
        );

        let error = AgentResultError::agent_output_error("Helper constructor error");
        assert!(matches!(error, AgentResultError::AgentOutputError(_)));
    }

    #[test]
    fn test_agent_result_error_from_json_error() {
        let invalid_json = "{ invalid json }";
        let json_error: Result<serde_json::Value, serde_json::Error> =
            serde_json::from_str(invalid_json);

        if let Err(json_err) = json_error {
            let agent_error: AgentResultError = json_err.into();
            assert!(matches!(
                agent_error,
                AgentResultError::DeserializationError(_)
            ));
        }
    }

    #[test]
    fn test_error_debug_formatting() {
        let error = RunnableAgentError::InitializationError("Init failed".to_string());
        let debug_str = format!("{error:?}");
        assert!(debug_str.contains("InitializationError"));
        assert!(debug_str.contains("Init failed"));
    }
}
