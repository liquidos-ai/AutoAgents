//! # AutoAgents Serve
//!
//! A library for running AutoAgents workflows from YAML configurations.
//!

pub mod builder;
pub mod config;
pub mod error;
pub mod tools;
mod utils;
pub mod workflow;

pub use builder::{BuiltWorkflow, WorkflowBuilder};
pub use config::{WorkflowConfig, WorkflowKind};
pub use error::{Result, WorkflowError};
pub use workflow::{Workflow, WorkflowOutput};

#[cfg(feature = "http-serve")]
mod server;

#[cfg(feature = "http-serve")]
pub use server::{HTTPServer, ServerConfig};
