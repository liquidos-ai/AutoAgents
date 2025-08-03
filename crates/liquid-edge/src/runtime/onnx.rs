//! ONNX Runtime implementation for liquid-edge
//!
//! This module provides ONNX Runtime backend for inference operations.
//! It requires the "onnx-runtime" feature to be enabled.

#[cfg(feature = "onnx-runtime")]
use crate::{
    error::{EdgeError, EdgeResult},
    traits::{InferenceRuntime, ModelInfo, RuntimeStats},
};

#[cfg(feature = "onnx-runtime")]
use ndarray::{ArrayD, CowArray};
#[cfg(feature = "onnx-runtime")]
use ort::{Environment, Session, SessionBuilder, Value};
#[cfg(feature = "onnx-runtime")]
use std::{path::Path, sync::Arc};

#[cfg(feature = "onnx-runtime")]
/// ONNX Runtime inference backend
pub struct OnnxRuntime {
    session: Arc<Session>,
    #[allow(dead_code)]
    environment: Arc<Environment>,
    model_info: ModelInfo,
    stats: RuntimeStats,
}

#[cfg(feature = "onnx-runtime")]
/// Input tensors for ONNX inference
#[derive(Debug, Clone)]
pub struct OnnxInput {
    pub input_ids: Vec<i64>,
    pub attention_mask: Option<Vec<i64>>,
    pub position_ids: Option<Vec<i64>>,
}

#[cfg(feature = "onnx-runtime")]
/// Output tensors from ONNX inference
#[derive(Debug)]
pub struct OnnxOutput {
    pub logits: Vec<f32>,
    pub shape: Vec<usize>,
}

#[cfg(feature = "onnx-runtime")]
impl OnnxRuntime {
    /// Create a new ONNX runtime instance
    pub fn new<P: AsRef<Path>>(model_path: P, model_name: String) -> EdgeResult<Self> {
        let environment = Arc::new(
            Environment::builder()
                .with_name("liquid_edge_onnx")
                .build()
                .map_err(|e| {
                    EdgeError::runtime(format!("Failed to create ONNX environment: {}", e))
                })?,
        );

        let session = SessionBuilder::new(&environment)
            .map_err(|e| EdgeError::runtime(format!("Failed to create session builder: {}", e)))?
            .with_model_from_file(model_path.as_ref())
            .map_err(|e| EdgeError::model(format!("Failed to load ONNX model: {}", e)))?;

        // Extract model information from session metadata
        let model_info = Self::extract_model_info(&session, model_name)?;

        Ok(Self {
            session: Arc::new(session),
            environment,
            model_info,
            stats: RuntimeStats::default(),
        })
    }

    /// Create ONNX runtime with custom session options
    pub fn with_options<P: AsRef<Path>>(
        model_path: P,
        model_name: String,
        num_threads: Option<usize>,
    ) -> EdgeResult<Self> {
        let environment = Arc::new(
            Environment::builder()
                .with_name("liquid_edge_onnx")
                .build()
                .map_err(|e| {
                    EdgeError::runtime(format!("Failed to create ONNX environment: {}", e))
                })?,
        );

        let mut session_builder = SessionBuilder::new(&environment)
            .map_err(|e| EdgeError::runtime(format!("Failed to create session builder: {}", e)))?;

        if let Some(threads) = num_threads {
            session_builder = session_builder
                .with_intra_threads(threads as i16)
                .map_err(|e| EdgeError::runtime(format!("Failed to set thread count: {}", e)))?;
        }

        let session = session_builder
            .with_model_from_file(model_path.as_ref())
            .map_err(|e| EdgeError::model(format!("Failed to load ONNX model: {}", e)))?;

        let model_info = Self::extract_model_info(&session, model_name)?;

        Ok(Self {
            session: Arc::new(session),
            environment,
            model_info,
            stats: RuntimeStats::default(),
        })
    }

    /// Extract model information from the ONNX session
    fn extract_model_info(session: &Session, model_name: String) -> EdgeResult<ModelInfo> {
        // Get input information
        let _inputs = &session.inputs;

        // Default values - in a real implementation these would come from model metadata
        let vocab_size = 32000; // Default fallback
        let context_length = 2048; // Default fallback

        Ok(ModelInfo {
            name: model_name,
            version: None,
            architecture: "transformer".to_string(),
            parameter_count: None,
            vocab_size,
            context_length,
        })
    }

    /// Run inference directly to avoid borrowing issues
    fn run_inference(&self, input: &OnnxInput) -> EdgeResult<OnnxOutput> {
        let seq_len = input.input_ids.len();

        // Create all arrays first to ensure they live long enough
        let input_ids_array = ArrayD::from_shape_vec(vec![1, seq_len], input.input_ids.clone())
            .map_err(|e| {
                EdgeError::inference(format!("Failed to create input_ids tensor: {}", e))
            })?;
        let input_ids_cow = CowArray::from(input_ids_array);

        let attention_mask = if let Some(ref mask) = input.attention_mask {
            mask.clone()
        } else {
            vec![1i64; seq_len] // Default attention mask (all ones)
        };
        let attention_array =
            ArrayD::from_shape_vec(vec![1, seq_len], attention_mask).map_err(|e| {
                EdgeError::inference(format!("Failed to create attention_mask tensor: {}", e))
            })?;
        let attention_cow = CowArray::from(attention_array);

        // Create position arrays if needed
        let position_cow = if self.session.inputs.len() > 2 {
            let position_ids = if let Some(ref pos_ids) = input.position_ids {
                pos_ids.clone()
            } else {
                (0..seq_len as i64).collect()
            };
            let position_array =
                ArrayD::from_shape_vec(vec![1, seq_len], position_ids).map_err(|e| {
                    EdgeError::inference(format!("Failed to create position_ids tensor: {}", e))
                })?;
            Some(CowArray::from(position_array))
        } else {
            None
        };

        // Now create tensors - all arrays are guaranteed to live until the end of this function
        let input_ids_tensor = Value::from_array(self.session.allocator(), &input_ids_cow)
            .map_err(|e| {
                EdgeError::inference(format!("Failed to create input_ids value: {}", e))
            })?;

        let attention_tensor = Value::from_array(self.session.allocator(), &attention_cow)
            .map_err(|e| {
                EdgeError::inference(format!("Failed to create attention_mask value: {}", e))
            })?;

        let mut inputs = vec![input_ids_tensor, attention_tensor];

        // Add position tensor if created
        if let Some(ref pos_cow) = position_cow {
            let position_tensor =
                Value::from_array(self.session.allocator(), pos_cow).map_err(|e| {
                    EdgeError::inference(format!("Failed to create position_ids value: {}", e))
                })?;
            inputs.push(position_tensor);
        }

        // Run inference
        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| EdgeError::inference(format!("ONNX inference failed: {}", e)))?;

        // Process outputs
        if outputs.is_empty() {
            return Err(EdgeError::inference("No outputs received from ONNX model"));
        }

        let logits_tensor = &outputs[0];
        let logits_data = logits_tensor
            .try_extract::<f32>()
            .map_err(|e| EdgeError::inference(format!("Failed to extract logits: {}", e)))?;

        let shape = logits_data.view().shape().to_vec();
        let logits = logits_data.view().iter().cloned().collect();

        Ok(OnnxOutput { logits, shape })
    }
}

#[cfg(feature = "onnx-runtime")]
impl InferenceRuntime for OnnxRuntime {
    type Tensor = ArrayD<f32>;
    type Input = OnnxInput;
    type Output = OnnxOutput;

    fn infer(&self, inputs: Self::Input) -> EdgeResult<Self::Output> {
        use std::time::Instant;

        let start_time = Instant::now();

        // Run inference directly
        let result = self.run_inference(&inputs)?;

        let inference_time = start_time.elapsed().as_millis() as f64;
        log::debug!("ONNX inference completed in {:.2}ms", inference_time);

        Ok(result)
    }

    fn model_info(&self) -> ModelInfo {
        self.model_info.clone()
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn stats(&self) -> RuntimeStats {
        self.stats.clone()
    }
}

#[cfg(feature = "onnx-runtime")]
impl OnnxInput {
    /// Create a new ONNX input with just input IDs
    pub fn new(input_ids: Vec<i64>) -> Self {
        Self {
            input_ids,
            attention_mask: None,
            position_ids: None,
        }
    }

    /// Create ONNX input with attention mask
    pub fn with_attention_mask(mut self, attention_mask: Vec<i64>) -> Self {
        self.attention_mask = Some(attention_mask);
        self
    }

    /// Create ONNX input with position IDs
    pub fn with_position_ids(mut self, position_ids: Vec<i64>) -> Self {
        self.position_ids = Some(position_ids);
        self
    }

    /// Get sequence length
    pub fn sequence_length(&self) -> usize {
        self.input_ids.len()
    }
}

#[cfg(feature = "onnx-runtime")]
impl OnnxOutput {
    /// Get logits for the last token
    pub fn last_token_logits(&self) -> EdgeResult<&[f32]> {
        if self.shape.len() < 2 {
            return Err(EdgeError::inference("Invalid logits shape"));
        }

        let seq_len = self.shape[self.shape.len() - 2];
        let vocab_size = self.shape[self.shape.len() - 1];
        let last_pos = seq_len - 1;

        let start_idx = last_pos * vocab_size;
        let end_idx = start_idx + vocab_size;

        self.logits
            .get(start_idx..end_idx)
            .ok_or_else(|| EdgeError::inference("Invalid logits indices"))
    }

    /// Get all logits
    pub fn all_logits(&self) -> &[f32] {
        &self.logits
    }

    /// Get output shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

// Stub implementations when feature is not enabled
#[cfg(not(feature = "onnx-runtime"))]
/// Stub for OnnxRuntime when feature is disabled
pub struct OnnxRuntime;

#[cfg(not(feature = "onnx-runtime"))]
/// Stub for OnnxInput when feature is disabled
pub struct OnnxInput;

#[cfg(not(feature = "onnx-runtime"))]
/// Stub for OnnxOutput when feature is disabled
pub struct OnnxOutput;

#[cfg(not(feature = "onnx-runtime"))]
impl OnnxRuntime {
    /// Create method that returns error when feature is disabled
    pub fn new<P: AsRef<Path>>(_model_path: P, _model_name: String) -> EdgeResult<Self> {
        Err(EdgeError::feature_not_available("onnx-runtime"))
    }

    /// With options method that returns error when feature is disabled
    pub fn with_options<P: AsRef<Path>>(
        _model_path: P,
        _model_name: String,
        _num_threads: Option<usize>,
        _memory_pattern: bool,
    ) -> EdgeResult<Self> {
        Err(EdgeError::feature_not_available("onnx-runtime"))
    }
}

#[cfg(not(feature = "onnx-runtime"))]
impl OnnxInput {
    /// Stub method when feature is disabled
    pub fn new(_input_ids: Vec<i64>) -> Self {
        Self
    }
}

#[cfg(test)]
#[cfg(feature = "onnx-runtime")]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_input_creation() {
        let input_ids = vec![1, 2, 3, 4, 5];
        let input = OnnxInput::new(input_ids.clone());

        assert_eq!(input.input_ids, input_ids);
        assert!(input.attention_mask.is_none());
        assert_eq!(input.sequence_length(), 5);
    }

    #[test]
    fn test_onnx_input_with_attention_mask() {
        let input_ids = vec![1, 2, 3];
        let attention_mask = vec![1, 1, 0];

        let input = OnnxInput::new(input_ids.clone()).with_attention_mask(attention_mask.clone());

        assert_eq!(input.input_ids, input_ids);
        assert_eq!(input.attention_mask, Some(attention_mask));
    }
}
