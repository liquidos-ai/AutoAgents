// Runtime-independent modules (available on all platforms)
mod config;
pub mod error;
pub mod memory;
mod output;
mod protocol;
pub mod task;

pub mod prebuilt;

// Exports for all platforms
pub use config::AgentConfig;
pub use error::AgentResultError;
pub use output::AgentOutputT;
pub use protocol::AgentProtocol;
mod base;
mod builder;
mod context;
mod executor;
mod runnable;
mod state;

#[cfg(not(target_arch = "wasm32"))]
pub use base::AgentHandle;
pub use base::{AgentDeriveT, BaseAgent};
pub use builder::AgentBuilder;
pub use context::Context;
pub use executor::{AgentExecutor, ExecutorConfig, TurnResult};

#[cfg(not(target_arch = "wasm32"))]
pub use runnable::AgentActor;
pub use runnable::{IntoRunnable, RunnableAgent, RunnableAgentImpl};
