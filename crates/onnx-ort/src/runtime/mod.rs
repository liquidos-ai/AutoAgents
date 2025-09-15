//! Generic inference runtime for edge computing
//!
//! This module provides a generic interface for running inference on various
//! deep learning models using different backends.

use crate::error::{EdgeError, EdgeResult};
use serde_json::Value;
use std::collections::HashMap;

#[cfg(all(feature = "onnx", not(target_arch = "wasm32")))]
pub mod onnx;

/// Generic input for inference operations
#[derive(Debug, Clone, Default)]
pub struct InferenceInput {
    /// Named tensor inputs as key-value pairs
    /// Keys are input names, values are the tensor data as JSON
    pub inputs: HashMap<String, Value>,
    /// Input metadata
    pub metadata: HashMap<String, Value>,
}

/// Generic output from inference operations
#[derive(Debug, Clone, Default)]
pub struct InferenceOutput {
    /// Named tensor outputs as key-value pairs
    /// Keys are output names, values are the tensor data as JSON
    pub outputs: HashMap<String, Value>,
    /// Output metadata
    pub metadata: HashMap<String, Value>,
}

/// Generic inference runtime trait
pub trait RuntimeBackend: Send + Sync {
    /// Run inference with the given inputs
    fn infer(&mut self, input: InferenceInput) -> EdgeResult<InferenceOutput>;

    /// Get model information
    fn model_info(&self) -> HashMap<String, Value>;

    /// Check if the runtime is ready for inference
    fn is_ready(&self) -> bool;

    /// Get backend-specific metadata
    fn backend_info(&self) -> HashMap<String, Value>;
}

/// Main inference runtime that manages different backends
pub struct InferenceRuntime {
    backend: Box<dyn RuntimeBackend>,
    runtime_metadata: HashMap<String, Value>,
}

impl InferenceRuntime {
    /// Create a new inference runtime from a model with a specific device
    pub async fn from_model_with_device(
        model: Box<dyn crate::Model>,
        device: crate::Device,
    ) -> EdgeResult<Self> {
        let backend_type = model.model_type().to_string();

        let backend: Box<dyn RuntimeBackend> = match backend_type.as_str() {
            #[cfg(all(feature = "onnx", not(target_arch = "wasm32")))]
            "onnx" => {
                let backend = onnx::OnnxBackend::from_model_with_device(model, device)?;
                Box::new(backend)
            }
            #[cfg(target_arch = "wasm32")]
            "onnx" => {
                let backend = wasm::WasmBackend::from_model_with_device(model, device)?;
                Box::new(backend)
            }
            _ => {
                return Err(EdgeError::runtime(format!(
                    "Unsupported model type: {backend_type}"
                )));
            }
        };

        let mut runtime_metadata = HashMap::new();
        runtime_metadata.insert("backend_type".to_string(), Value::String(backend_type));
        runtime_metadata.insert("device_type".to_string(), Value::String(device.to_string()));
        runtime_metadata.insert(
            "created_at".to_string(),
            Value::String(chrono::Utc::now().to_rfc3339()),
        );

        Ok(Self {
            backend,
            runtime_metadata,
        })
    }

    /// Create a new inference runtime from a model (uses CPU device by default, WebGPU on WASM)
    pub async fn from_model(model: Box<dyn crate::Model>) -> EdgeResult<Self> {
        #[cfg(target_arch = "wasm32")]
        let device = crate::device::webgpu();
        #[cfg(not(target_arch = "wasm32"))]
        let device = crate::device::cpu();

        Self::from_model_with_device(model, device).await
    }

    /// Run inference on the loaded model
    pub fn infer(&mut self, input: InferenceInput) -> EdgeResult<InferenceOutput> {
        if !self.backend.is_ready() {
            return Err(EdgeError::runtime("Backend is not ready for inference"));
        }

        self.backend.infer(input)
    }

    /// Get comprehensive model information
    pub fn model_info(&self) -> HashMap<String, Value> {
        let mut info = self.backend.model_info();
        info.extend(self.runtime_metadata.clone());
        info
    }

    /// Check if the runtime is ready for inference
    pub fn is_ready(&self) -> bool {
        self.backend.is_ready()
    }

    /// Get backend information
    pub fn backend_info(&self) -> HashMap<String, Value> {
        self.backend.backend_info()
    }
}

impl InferenceInput {
    /// Create a new empty inference input
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tensor input
    pub fn add_input(mut self, name: String, data: Value) -> Self {
        self.inputs.insert(name, data);
        self
    }

    /// Add metadata
    pub fn add_metadata(mut self, key: String, value: Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

impl InferenceOutput {
    /// Create a new empty inference output
    pub fn new() -> Self {
        Self::default()
    }

    /// Get output by name
    pub fn get_output(&self, name: &str) -> Option<&Value> {
        self.outputs.get(name)
    }

    /// Get metadata by key
    pub fn get_metadata(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_input_creation() {
        let input = InferenceInput::new()
            .add_input(
                "input_ids".to_string(),
                Value::Array(vec![Value::Number(1.into())]),
            )
            .add_metadata("batch_size".to_string(), Value::Number(1.into()));

        assert!(input.inputs.contains_key("input_ids"));
        assert!(input.metadata.contains_key("batch_size"));
    }
}
