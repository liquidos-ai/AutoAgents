//! AutoAgents prelude: common traits, types, and macros for quick start.

// Macros and derives
pub use autoagents_derive::{AgentHooks, AgentOutput, ToolInput, agent, tool};

// Core agent types
pub use crate::core::agent::memory::SlidingWindowMemory;
pub use crate::core::agent::prebuilt::executor::{
    BasicAgent, BasicAgentOutput, ReActAgent, ReActAgentOutput,
};
pub use crate::core::agent::task::Task;
pub use crate::core::agent::{ActorAgent, AgentBuilder, DirectAgent};
pub use crate::core::agent::{AgentHooks as _, AgentOutputT, Context};

// Tools
pub use crate::core::tool::{ToolCallResult, ToolInputT, ToolRuntime, ToolT};

// Runtime / Environment / Messaging (non-WASM)
#[cfg(not(target_arch = "wasm32"))]
pub use crate::core::actor::Topic;
#[cfg(not(target_arch = "wasm32"))]
pub use crate::core::environment::Environment;
#[cfg(not(target_arch = "wasm32"))]
pub use crate::core::runtime::{SingleThreadedRuntime, TypedRuntime};
pub use crate::protocol::Event;

// Errors
pub use crate::core::error::Error;

// LLM abstractions
pub use crate::llm::LLMProvider;
pub use crate::llm::builder::LLMBuilder;

// Utils
pub use crate::init_logging;
