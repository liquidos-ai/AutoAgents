use crate::{agent::error::AgentError, environment::EnvironmentError, session::SessionError};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    EnvironmentError(#[from] EnvironmentError),
    #[error(transparent)]
    SessionError(#[from] SessionError),
    #[error(transparent)]
    AgentError(#[from] AgentError),
}
