use autoagents_llm::error::LLMError;

use crate::{

    agent::error::{AgentBuildError, RunnableAgentError},
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
    AgentBuildError(#[from] AgentBuildError),
    #[error(transparent)]
    RunnableAgentError(#[from] RunnableAgentError),
    #[error(transparent)]
    LLMError(#[from] LLMError),
}

