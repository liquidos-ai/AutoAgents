//! Model abstraction layer for autoagents-onnx
//!
//! This module provides abstractions for different types of models that can be
//! loaded and used with the autoagents-onnx inference runtime.

use crate::error::EdgeResult;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

/// Trait representing a model that can be loaded and used for inference
pub trait Model: Send + Sync + std::fmt::Debug {
    /// Get the model type (e.g., "onnx", "tensorrt", "coreml")
    fn model_type(&self) -> &str;

    /// Get the model path or identifier
    fn model_path(&self) -> &Path;

    /// Get model metadata
    fn metadata(&self) -> &HashMap<String, Value>;

    /// Get model configuration as JSON
    fn config(&self) -> EdgeResult<Value>;

    /// Validate that the model files exist and are valid
    fn validate(&self) -> EdgeResult<()>;
}
