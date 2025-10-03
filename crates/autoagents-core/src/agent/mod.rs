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
// mod runnable;
mod actor;
pub(crate) mod constants;
mod direct;
mod hooks;
mod state;

pub use actor::ActorAgent;
#[cfg(not(target_arch = "wasm32"))]
pub use actor::ActorAgentHandle;
pub use base::{AgentDeriveT, BaseAgent};
pub use builder::AgentBuilder;
pub use context::{Context, ContextError};
pub use direct::DirectAgent;
pub use executor::{
    event_helper::EventHelper, memory_helper::MemoryHelper, tool_processor::ToolProcessor,
    AgentExecutor, ExecutorConfig, TurnResult,
};
pub use hooks::{AgentHooks, HookOutcome};
