use crate::{
    agent::error::{AgentError, RunnableAgentError},
    environment::EnvironmentError,
    session::SessionError,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    EnvironmentError(#[from] EnvironmentError),
    #[error(transparent)]
    SessionError(#[from] SessionError),
    #[error(transparent)]
    AgentError(#[from] AgentError),
    #[error(transparent)]
    RunnableAgentError(#[from] RunnableAgentError),
}
