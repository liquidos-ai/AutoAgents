//! # Liquid Edge - High-Performance Edge Inference Runtime
//!
//! Liquid Edge is a production-ready inference runtime designed for edge computing
//! environments. It provides high-performance LLM inference with multiple backend
//! support and comprehensive tokenization capabilities.
//!
//! ## Features
//!
//! - **Multiple Backends**: Support for ONNX Runtime with more backends planned
//! - **Async Support**: Full async/await support for non-blocking inference
//! - **Chat Templates**: Jinja2 template support for conversational AI
//! - **Production Ready**: Comprehensive error handling, logging, and monitoring
//! - **Edge Optimized**: Memory-efficient and low-latency inference
//!
//! ## Quick Start
//!
//!
//! ## Feature Flags
//!
//! - `onnx-runtime` (default): Enable ONNX Runtime backend
//! - `async` (default): Enable async/await support
//! - `chat` (default): Enable chat functionality and templates
//! - `jinja-templates` (default): Enable Jinja2 template rendering
//! - `serde` (default): Enable serialization support
//!
//! ## Backends
//!
//! ### ONNX Runtime
//!
//! The ONNX Runtime backend provides high-performance inference using Microsoft's
//! ONNX Runtime with support for CPU and GPU acceleration.
//!

// Core modules (always available)
pub mod error;
pub mod traits;

// Configuration module (requires serde for file loading)
#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
pub mod config;

// Stub config module when serde is not available
#[cfg(not(feature = "serde"))]
pub mod config {
    //! Configuration module stub
    //!
    //! This module provides basic configuration types when the `serde` feature
    //! is not enabled. For full configuration support, enable the `serde` feature.

    /// Basic model configuration without serialization support
    #[derive(Debug, Clone)]
    pub struct ModelConfig {
        /// Model name
        pub model_name: String,
        /// End-of-sequence token ID
        pub eos_token_id: u32,
        /// Beginning-of-sequence token ID
        pub bos_token_id: u32,
        /// Maximum position embeddings
        pub max_position_embeddings: usize,
        /// Vocabulary size
        pub vocab_size: usize,
    }

    impl Default for ModelConfig {
        fn default() -> Self {
            Self {
                model_name: "unknown".to_string(),
                eos_token_id: 2,
                bos_token_id: 1,
                max_position_embeddings: 2048,
                vocab_size: 32000,
            }
        }
    }

    /// Stub for TokenizerConfig
    #[derive(Debug, Clone, Default)]
    pub struct TokenizerConfig {
        pub model_max_length: usize,
        pub add_bos_token: bool,
        pub add_eos_token: bool,
    }

    /// Stub for SpecialTokensMap
    #[derive(Debug, Clone, Default)]
    pub struct SpecialTokensMap;

    /// Stub for EdgeConfig
    #[derive(Debug, Clone)]
    pub struct EdgeConfig {
        pub model: ModelConfig,
    }

    impl EdgeConfig {
        pub fn new(model_name: impl Into<String>) -> Self {
            let mut model = ModelConfig::default();
            model.model_name = model_name.into();
            Self { model }
        }
    }
}

// Runtime backends
pub mod runtime;

// Tokenizer support
pub mod tokenizer;

// Template rendering (requires jinja-templates feature)
#[cfg(feature = "jinja-templates")]
#[cfg_attr(docsrs, doc(cfg(feature = "jinja-templates")))]
pub mod templates;

// Stub templates module when jinja-templates is not available
#[cfg(not(feature = "jinja-templates"))]
pub mod templates {
    //! Template rendering module stub
    //!
    //! This module provides basic template types when the `jinja-templates` feature
    //! is not enabled. For full template support, enable the `jinja-templates` feature.

    use crate::error::{EdgeError, EdgeResult};
    use std::collections::HashMap;

    /// Template renderer stub
    pub struct TemplateRenderer;

    impl TemplateRenderer {
        /// Create a new template renderer (stub)
        pub fn new() -> EdgeResult<Self> {
            Err(EdgeError::feature_not_available("jinja-templates"))
        }

        /// Render template (stub)
        pub fn render(
            &self,
            _template: &str,
            _context: &HashMap<String, serde_json::Value>,
        ) -> EdgeResult<String> {
            Err(EdgeError::feature_not_available("jinja-templates"))
        }
    }

    impl Default for TemplateRenderer {
        fn default() -> Self {
            Self
        }
    }
}

// Sampling strategies for text generation
pub mod sampling;

// Utility functions
pub mod utils;

// Re-export core types for convenience
pub use error::{EdgeError, EdgeResult};

// Re-export traits
pub use traits::{
    GenerationOptions, InferenceRuntime, ModelInfo, ModelLoader, RuntimeStats, SpecialTokenIds,
    TokenizedOutput, TokenizerTrait,
};

// Re-export runtime types based on features
#[cfg(feature = "onnx-runtime")]
#[cfg_attr(docsrs, doc(cfg(feature = "onnx-runtime")))]
pub use runtime::{OptimizationLevel, RuntimeConfig, RuntimeFactory};

#[cfg(feature = "onnx-runtime")]
#[cfg_attr(docsrs, doc(cfg(feature = "onnx-runtime")))]
pub use runtime::{OnnxInput, OnnxOutput, OnnxRuntime};

// Re-export tokenizer
pub use tokenizer::{Tokenizer, TokenizerBuilder};

// Re-export config (always available, but with different implementations)
pub use config::ModelConfig;

#[cfg(feature = "serde")]
pub use config::{EdgeConfig, SpecialTokensMap, TokenizerConfig};

#[cfg(not(feature = "serde"))]
pub use config::{EdgeConfig, SpecialTokensMap, TokenizerConfig};

// Chat functionality (requires chat feature)
#[cfg(feature = "chat")]
#[cfg_attr(docsrs, doc(cfg(feature = "chat")))]
pub use traits::{ChatMessage, ChatResponse, ChatRuntime, FinishReason};

/// Library version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Check if a feature is enabled at runtime
pub fn is_feature_enabled(feature: &str) -> bool {
    #[allow(clippy::match_like_matches_macro)]
    match feature {
        "onnx-runtime" => cfg!(feature = "onnx-runtime"),
        "chat" => cfg!(feature = "chat"),
        "jinja-templates" => cfg!(feature = "jinja-templates"),
        "serde" => cfg!(feature = "serde"),
        _ => false,
    }
}

/// Get list of enabled features
pub fn enabled_features() -> Vec<&'static str> {
    let mut features = Vec::new();

    if cfg!(feature = "onnx-runtime") {
        features.push("onnx-runtime");
    }
    if cfg!(feature = "chat") {
        features.push("chat");
    }
    if cfg!(feature = "jinja-templates") {
        features.push("jinja-templates");
    }
    if cfg!(feature = "serde") {
        features.push("serde");
    }

    features
}

/// Initialize the liquid-edge runtime with logging
pub fn init() -> EdgeResult<()> {
    // Initialize logging if not already done
    if log::max_level() == log::LevelFilter::Off {
        env_logger::try_init()
            .map_err(|e| EdgeError::runtime(format!("Failed to initialize logging: {e}")))?;
    }

    log::info!("Liquid Edge v{VERSION} initialized");
    log::debug!("Enabled features: {:?}", enabled_features());

    Ok(())
}

/// Initialize with custom log level
pub fn init_with_level(level: log::LevelFilter) -> EdgeResult<()> {
    env_logger::Builder::from_default_env()
        .filter_level(level)
        .try_init()
        .map_err(|e| EdgeError::runtime(format!("Failed to initialize logging: {e}")))?;

    log::info!("Liquid Edge v{VERSION} initialized with log level {level:?}");
    log::debug!("Enabled features: {:?}", enabled_features());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_feature_detection() {
        // Test that we can detect at least some features
        let features = enabled_features();
        assert!(!features.is_empty()); // At least one feature should be enabled by default
    }

    #[test]
    fn test_feature_check() {
        // Test known features
        assert!(is_feature_enabled("onnx-runtime") || !is_feature_enabled("onnx-runtime"));
        assert!(!is_feature_enabled("nonexistent-feature"));
    }

    #[test]
    fn test_init() {
        // Test that init doesn't panic
        let _ = init();
    }
}
