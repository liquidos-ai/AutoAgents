//! AutoAgents LLM is a unified interface for interacting with Large Language Model providers.
//!
//! # Overview
//! This crate provides a consistent API for working with different LLM backends by abstracting away
//! provider-specific implementation details. It supports:
//!
//! - Chat-based interactions
//! - Text completion
//! - Embeddings generation
//! - Multiple providers (OpenAI, Anthropic, etc.)
//! - Request validation and retry logic
//!
//! # Architecture
//! The crate is organized into modules that handle different aspects of LLM interactions:

// Re-export for convenience
pub use async_trait::async_trait;

use chat::Tool;
use serde::{Deserialize, Serialize};

/// Backend implementations for supported LLM providers like OpenAI, Anthropic, etc.
pub mod backends;

/// Builder pattern for configuring and instantiating LLM providers
pub mod builder;

/// Chat-based interactions with language models (e.g. ChatGPT style)
pub mod chat;

/// Text completion capabilities (e.g. GPT-3 style completion)
pub mod completion;

/// Vector embeddings generation for text
pub mod embedding;

/// Error types and handling
pub mod error;

/// Evaluator for LLM providers
pub mod evaluator;

/// Secret store for storing API keys and other sensitive information
pub mod secret_store;

/// Listing models support
pub mod models;

mod tool;
pub use tool::{ToolInputT, ToolT};

#[inline]
/// Initialize logging using env_logger if the "logging" feature is enabled.
/// This is a no-op if the feature is not enabled.
pub fn init_logging() {
    #[cfg(feature = "logging")]
    {
        let _ = env_logger::try_init();
    }
}

/// Core trait that all LLM providers must implement, combining chat, completion
/// and embedding capabilities into a unified interface
pub trait LLMProvider:
    chat::ChatProvider
    + completion::CompletionProvider
    + embedding::EmbeddingProvider
    + models::ModelsProvider
    + Send
    + Sync
    + 'static
{
    fn tools(&self) -> Option<&[Tool]> {
        None
    }
}

/// Tool call represents a function call that an LLM wants to make.
/// This is a standardized structure used across all providers.
#[derive(Debug, Deserialize, Serialize, Clone, Eq, PartialEq)]
pub struct ToolCall {
    /// The ID of the tool call.
    pub id: String,
    /// The type of the tool call (usually "function").
    #[serde(rename = "type")]
    pub call_type: String,
    /// The function to call.
    pub function: FunctionCall,
}

/// FunctionCall contains details about which function to call and with what arguments.
#[derive(Debug, Deserialize, Serialize, Clone, Eq, PartialEq)]
pub struct FunctionCall {
    /// The name of the function to call.
    pub name: String,
    /// The arguments to pass to the function, typically serialized as a JSON string.
    pub arguments: String,
}
