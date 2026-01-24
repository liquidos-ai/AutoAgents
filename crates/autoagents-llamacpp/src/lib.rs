//! # AutoAgents llama.cpp Backend
//!
//! Local LLM inference backend for AutoAgents using llama-cpp-2 bindings.
//!
//! ## Features
//!
//! - **GGUF Model Support**: Load local GGUF models via llama.cpp
//! - **Sampling Controls**: Temperature, top-k, top-p, penalties
//! - **Structured Output**: JSON schema hints with optional grammar enforcement
//! - **Streaming**: Token streaming for chat responses
//! - **Production Ready**: Robust error handling and configuration
//!

pub mod builder;
pub mod config;
pub mod conversion;
pub mod error;
pub mod huggingface;
pub mod models;
pub mod provider;

// Re-exports for convenience
pub use builder::LlamaCppProviderBuilder;
pub use config::{LlamaCppConfig, LlamaCppConfigBuilder, LlamaCppSplitMode};
pub use error::LlamaCppProviderError;
pub use models::ModelSource;
pub use provider::LlamaCppProvider;

// Re-export llama-cpp types that users might need
pub use llama_cpp_2::model::params::LlamaSplitMode;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_imports() {
        let _source = ModelSource::Gguf {
            model_path: "model.gguf".to_string(),
        };
    }

    #[test]
    fn test_builder_accessible() {
        let _builder = LlamaCppProvider::builder();
        let _config_builder = LlamaCppConfigBuilder::new();
    }
}
