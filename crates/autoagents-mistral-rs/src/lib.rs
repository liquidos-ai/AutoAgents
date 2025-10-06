//! # AutoAgents Mistral.rs Backend
//!
//! Local LLM inference backend for AutoAgents using mistral.rs.
//!
//! ## Features
//!
//! - **Dual Model Support**: Load models from HuggingFace or local GGUF files
//! - **Hardware Acceleration**: CUDA, Metal, cuDNN support via feature flags
//! - **Quantization**: In-situ quantization for HF models, native GGUF quantization
//! - **Memory Efficient**: Paged attention support
//! - **Production Ready**: Comprehensive error handling and logging
//!

pub mod config;
pub mod conversion;
pub mod error;
pub mod models;
pub mod provider;

// Re-exports for convenience
pub use config::{MistralRsConfig, MistralRsConfigBuilder};
pub use error::MistralRsError;
pub use models::{GgufQuant, HFModels, ModelSource};
pub use provider::{MistralRsProvider, MistralRsProviderBuilder};

// Re-export mistralrs types that users might need
pub use mistralrs::IsqType;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_imports() {
        // Test that all main types are accessible
        let _quant = GgufQuant::Q4_K_M;
        let _model = HFModels::Phi35MiniInstruct;
        let _source = ModelSource::HuggingFace {
            repo_id: "test/model".to_string(),
            revision: None,
            model_type: models::ModelType::Auto,
        };
    }

    #[test]
    fn test_builder_accessible() {
        let _builder = MistralRsProvider::builder();
        let _config_builder = MistralRsConfigBuilder::new();
    }
}
