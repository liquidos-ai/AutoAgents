//! Error types for liquid-edge runtime

use thiserror::Error;

/// Result type for liquid-edge operations
pub type EdgeResult<T> = Result<T, EdgeError>;

/// Main error type for liquid-edge runtime
#[derive(Error, Debug)]
pub enum EdgeError {
    #[error("Model error: {0}")]
    Model(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("ONNX Runtime error: {0}")]
    #[cfg(all(feature = "onnx", not(target_arch = "wasm32")))]
    OnnxRuntime(#[from] ort::Error),

    #[error("Runtime error: {0}")]
    Runtime(String),
}

impl EdgeError {
    pub fn model(msg: impl Into<String>) -> Self {
        Self::Model(msg.into())
    }

    pub fn inference(msg: impl Into<String>) -> Self {
        Self::Inference(msg.into())
    }

    pub fn tokenizer(msg: impl Into<String>) -> Self {
        Self::Tokenizer(msg.into())
    }

    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    pub fn runtime(msg: impl Into<String>) -> Self {
        Self::Runtime(msg.into())
    }
}
