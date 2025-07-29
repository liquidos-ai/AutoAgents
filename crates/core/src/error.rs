use autoagents_llm::error::LLMError;

use crate::{
    agent::{AgentBuildError, AgentResultError, RunnableAgentError},
    environment::EnvironmentError,
    runtime::RuntimeError,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    EnvironmentError(#[from] EnvironmentError),
    #[error(transparent)]
    RuntimeError(#[from] RuntimeError),
    #[error(transparent)]
    AgentBuildError(#[from] AgentBuildError),
    #[error(transparent)]
    RunnableAgentError(#[from] RunnableAgentError),
    #[error(transparent)]
    LLMError(#[from] LLMError),
    #[error(transparent)]
    AgentResultError(#[from] AgentResultError),
}
