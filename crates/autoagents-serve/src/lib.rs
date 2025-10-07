//! # AutoAgents Serve
//!
//! A library for running AutoAgents workflows from YAML configurations.
//!
//! ## Features
//!
//! - **Workflow Types**: Support for Direct, Sequential, Parallel, and Routing workflows
//! - **Builder Pattern**: Fluent API for constructing workflows
//! - **Multiple LLM Providers**: OpenAI, Anthropic, Ollama, Groq, and more
//! - **Tool Integration**: Built-in support for various tools
//! - **HTTP Server**: Optional HTTP REST API for serving workflows (feature: `http-serve`)
//!
//! ## Quick Start
//!
//! ```no_run
//! use autoagents_serve::WorkflowBuilder;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let workflow = WorkflowBuilder::from_yaml_file("workflow.yaml")?
//!         .build()?;
//!     
//!     let result = workflow.run("Your input here".to_string()).await?;
//!     println!("Result: {:?}", result);
//!     Ok(())
//! }
//! ```

pub mod builder;
pub mod config;
pub mod error;
pub mod tools;
pub mod workflow;

#[cfg(feature = "http-serve")]
pub mod server;

pub use builder::{BuiltWorkflow, WorkflowBuilder};
pub use config::{WorkflowConfig, WorkflowKind};
pub use error::{Result, WorkflowError};
pub use workflow::{Workflow, WorkflowOutput};

#[cfg(feature = "http-serve")]
pub use server::{generate_model_key, serve, ServerConfig};
