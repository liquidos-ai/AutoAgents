use crate::agent::AgentResultError;

use crate::agent::error::{AgentBuildError, RunnableAgentError};
#[cfg(not(target_arch = "wasm32"))]
use crate::{environment::EnvironmentError, runtime::RuntimeError};
use autoagents_llm::error::LLMError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[cfg(not(target_arch = "wasm32"))]
    #[error(transparent)]
    EnvironmentError(#[from] EnvironmentError),
    #[cfg(not(target_arch = "wasm32"))]
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
    #[error("Custom Error: {0}")]
    CustomError(String),
}
