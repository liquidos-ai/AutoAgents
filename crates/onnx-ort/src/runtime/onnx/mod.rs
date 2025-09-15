//! ONNX Runtime backend for onnx inference

use crate::error::{EdgeError, EdgeResult};
use crate::runtime::{InferenceInput, InferenceOutput, RuntimeBackend};
use crate::{Device, Model};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::{ArrayD, IxDyn};

#[cfg(feature = "onnx")]
use ort::{session::Session, value::Value as OrtValue};

#[cfg(all(feature = "onnx", not(target_arch = "wasm32")))]
use ort::execution_providers::ExecutionProvider;

/// ONNX Runtime backend for edge inference
pub struct OnnxBackend {
    session: Session,
    input_info: Vec<InputInfo>,
    output_info: Vec<OutputInfo>,
}

#[derive(Debug, Clone)]
struct InputInfo {
    name: String,
    shape: Vec<i64>,
    data_type: String,
}

#[derive(Debug, Clone)]
struct OutputInfo {
    name: String,
    shape: Vec<i64>,
    data_type: String,
}

impl OnnxBackend {
    /// Create a new ONNX backend from a model with a specific device
    pub fn from_model_with_device(model: Box<dyn Model>, device: Device) -> EdgeResult<Self> {
        // Validate the model
        model.validate()?;

        // Check if device is available
        if !device.is_available() {
            return Err(EdgeError::runtime(format!(
                "Device {device} is not available"
            )));
        }

        // Get the actual ONNX model file path
        let model_path = model.model_path();
        let onnx_file = if model_path.is_file()
            && model_path.extension().and_then(|e| e.to_str()) == Some("onnx")
        {
            model_path.to_path_buf()
        } else {
            model_path.join("model.onnx")
        };

        if !onnx_file.exists() {
            return Err(EdgeError::model(format!(
                "ONNX model file not found: {}",
                onnx_file.display()
            )));
        }

        Self::new_with_device(onnx_file, device)
    }

    /// Create a new ONNX backend from a model (uses CPU device by default)
    pub fn from_model(model: Box<dyn Model>) -> EdgeResult<Self> {
        let device = crate::device::cpu();
        Self::from_model_with_device(model, device)
    }

    /// Create a new ONNX backend with a specific device
    pub fn new_with_device<P: AsRef<Path>>(model_path: P, device: Device) -> EdgeResult<Self> {
        // Check if device is available
        if !device.is_available() {
            return Err(EdgeError::runtime(format!(
                "Device {device} is not available"
            )));
        }

        // Create session with device-specific execution provider following USLS pattern
        let mut builder = Session::builder()
            .map_err(|e| EdgeError::runtime(format!("Failed to create session builder: {e}")))?;

        // Register execution provider based on device type (following USLS pattern)
        match device {
            #[allow(unused_variables)]
            Device::Cuda(id) => {
                #[cfg(feature = "cuda")]
                {
                    use ort::execution_providers::CUDAExecutionProvider;
                    let ep = CUDAExecutionProvider::default().with_device_id(id as i32);
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|e| {
                                EdgeError::runtime(format!("Failed to register CUDA: {e}"))
                            })?;
                        }
                        _ => {
                            return Err(EdgeError::runtime("CUDA execution provider not available"))
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(EdgeError::runtime("CUDA feature not enabled"));
                }
            }
            Device::Cpu(_) => {
                use ort::execution_providers::CPUExecutionProvider;
                let ep = CPUExecutionProvider::default();
                ep.register(&mut builder)
                    .map_err(|e| EdgeError::runtime(format!("Failed to register CPU: {e}")))?;
            }
        }

        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| EdgeError::model(format!("Failed to load ONNX model: {e}")))?;

        Self::create_backend(session)
    }

    /// Create a new ONNX backend (uses CPU device by default)
    pub fn new<P: AsRef<Path>>(model_path: P) -> EdgeResult<Self> {
        let device = crate::device::cpu();
        Self::new_with_device(model_path, device)
    }

    /// Common backend creation logic
    fn create_backend(session: Session) -> EdgeResult<Self> {
        // Extract input information
        let input_info: Vec<InputInfo> = session
            .inputs
            .iter()
            .map(|input| {
                let shape = vec![-1, -1]; // Dynamic shape for now

                InputInfo {
                    name: input.name.clone(),
                    shape,
                    data_type: format!("{:?}", input.input_type),
                }
            })
            .collect();

        // Extract output information
        let output_info: Vec<OutputInfo> = session
            .outputs
            .iter()
            .map(|output| {
                let shape = vec![-1, -1, -1]; // Dynamic shape for now

                OutputInfo {
                    name: output.name.clone(),
                    shape,
                    data_type: format!("{:?}", output.output_type),
                }
            })
            .collect();

        log::info!(
            "ONNX Backend initialized with {} inputs and {} outputs",
            input_info.len(),
            output_info.len()
        );

        for (i, input) in input_info.iter().enumerate() {
            log::info!(
                "  Input {}: name='{}', type={}, shape={:?}",
                i,
                input.name,
                input.data_type,
                input.shape
            );
        }

        for (i, output) in output_info.iter().enumerate() {
            log::info!(
                "  Output {}: name='{}', type={}, shape={:?}",
                i,
                output.name,
                output.data_type,
                output.shape
            );
        }

        Ok(Self {
            session,
            input_info,
            output_info,
        })
    }

    /// Convert JSON value to ONNX tensor
    fn json_to_tensor(
        &self,
        name: &str,
        data: &Value,
    ) -> EdgeResult<ort::value::Value<ort::value::DynValueTypeMarker>> {
        match data {
            Value::Array(arr) => {
                if let Ok(i64_values) = arr
                    .iter()
                    .map(|v| v.as_i64().ok_or("Invalid i64"))
                    .collect::<Result<Vec<_>, _>>()
                {
                    let len = i64_values.len();
                    let array = ArrayD::<i64>::from_shape_vec(IxDyn(&[1, len]), i64_values)
                        .map_err(|e| {
                            EdgeError::inference(format!(
                                "Failed to create i64 tensor for {name}: {e}"
                            ))
                        })?;

                    Ok(OrtValue::from_array(array)
                        .map_err(|e| {
                            EdgeError::inference(format!(
                                "Failed to create ONNX value for {name}: {e}"
                            ))
                        })?
                        .into_dyn())
                }
                // Try f32 array
                else if let Ok(f32_values) = arr
                    .iter()
                    .map(|v| v.as_f64().map(|f| f as f32).ok_or("Invalid f32"))
                    .collect::<Result<Vec<_>, _>>()
                {
                    let len = f32_values.len();
                    let array = ArrayD::<f32>::from_shape_vec(IxDyn(&[1, len]), f32_values)
                        .map_err(|e| {
                            EdgeError::inference(format!(
                                "Failed to create f32 tensor for {name}: {e}"
                            ))
                        })?;

                    Ok(OrtValue::from_array(array)
                        .map_err(|e| {
                            EdgeError::inference(format!(
                                "Failed to create ONNX value for {name}: {e}"
                            ))
                        })?
                        .into_dyn())
                } else {
                    Err(EdgeError::inference(format!(
                        "Unsupported data type in array for input: {name}"
                    )))
                }
            }
            _ => Err(EdgeError::inference(format!(
                "Unsupported JSON type for input: {name}"
            ))),
        }
    }

    /// Convert ONNX tensor to JSON value
    fn tensor_to_json_static(
        tensor: &ort::value::Value<ort::value::DynValueTypeMarker>,
    ) -> EdgeResult<Value> {
        // Try to extract as f32 first
        if let Ok((_, data)) = tensor.try_extract_tensor::<f32>() {
            let values: Vec<Value> = data
                .iter()
                .map(|&x| {
                    Value::Number(
                        serde_json::Number::from_f64(x as f64)
                            .unwrap_or(serde_json::Number::from(0)),
                    )
                })
                .collect();
            return Ok(Value::Array(values));
        }

        // Try to extract as i64
        if let Ok((_, data)) = tensor.try_extract_tensor::<i64>() {
            let values: Vec<Value> = data.iter().map(|&x| Value::Number(x.into())).collect();
            return Ok(Value::Array(values));
        }

        Err(EdgeError::inference(
            "Unsupported tensor type for output conversion",
        ))
    }
}

impl RuntimeBackend for OnnxBackend {
    fn infer(&mut self, input: InferenceInput) -> EdgeResult<InferenceOutput> {
        // Convert inputs to ONNX format
        let mut onnx_inputs = HashMap::new();

        for input_info in &self.input_info {
            if let Some(data) = input.inputs.get(&input_info.name) {
                let tensor = self.json_to_tensor(&input_info.name, data)?;
                onnx_inputs.insert(input_info.name.clone(), tensor);
            } else {
                return Err(EdgeError::inference(format!(
                    "Missing required input: {}",
                    input_info.name
                )));
            }
        }

        // Run inference
        let outputs = self
            .session
            .run(onnx_inputs)
            .map_err(|e| EdgeError::inference(format!("ONNX inference failed: {e}")))?;

        // Convert outputs back to JSON
        let mut result_outputs = HashMap::new();
        for output_info in &self.output_info {
            if let Some(tensor) = outputs.get(&output_info.name) {
                let json_data = Self::tensor_to_json_static(tensor)?;
                result_outputs.insert(output_info.name.clone(), json_data);
            }
        }

        let mut metadata = HashMap::new();
        metadata.insert("backend".to_string(), Value::String("onnx".to_string()));
        metadata.insert("inference_time_ms".to_string(), Value::Number(0.into())); // TODO: Add timing

        Ok(InferenceOutput {
            outputs: result_outputs,
            metadata,
        })
    }

    fn model_info(&self) -> HashMap<String, Value> {
        let mut info = HashMap::new();
        info.insert(
            "backend_type".to_string(),
            Value::String("onnx".to_string()),
        );
        info.insert(
            "num_inputs".to_string(),
            Value::Number(self.input_info.len().into()),
        );
        info.insert(
            "num_outputs".to_string(),
            Value::Number(self.output_info.len().into()),
        );

        let inputs: Vec<Value> = self
            .input_info
            .iter()
            .map(|input| {
                serde_json::json!({
                    "name": input.name,
                    "data_type": input.data_type,
                    "shape": input.shape
                })
            })
            .collect();
        info.insert("inputs".to_string(), Value::Array(inputs));

        let outputs: Vec<Value> = self
            .output_info
            .iter()
            .map(|output| {
                serde_json::json!({
                    "name": output.name,
                    "data_type": output.data_type,
                    "shape": output.shape
                })
            })
            .collect();
        info.insert("outputs".to_string(), Value::Array(outputs));

        info
    }

    fn is_ready(&self) -> bool {
        true // ONNX session is ready once created
    }

    fn backend_info(&self) -> HashMap<String, Value> {
        let mut info = HashMap::new();
        info.insert(
            "name".to_string(),
            Value::String("ONNX Runtime".to_string()),
        );
        info.insert("version".to_string(), Value::String("2.0".to_string()));
        info.insert("supports_gpu".to_string(), Value::Bool(false)); // TODO: Detect GPU support
        info
    }
}

/// ONNX model implementation
#[derive(Debug, Clone)]
pub struct OnnxModel {
    path: PathBuf,
    metadata: HashMap<String, Value>,
}

impl OnnxModel {
    /// Create a new ONNX model from a directory path
    pub fn from_directory<P: AsRef<Path>>(path: P) -> EdgeResult<Self> {
        let path = path.as_ref().to_path_buf();
        let mut metadata = HashMap::new();

        // Check if path exists
        if !path.exists() {
            return Err(EdgeError::model(format!(
                "Model directory does not exist: {}",
                path.display()
            )));
        }

        // Try to load config.json if it exists
        let config_path = path.join("config.json");
        if config_path.exists() {
            let config_content = std::fs::read_to_string(&config_path)?;
            let config: Value = serde_json::from_str(&config_content)?;

            // Extract metadata from config
            if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
                metadata.insert(
                    "model_type".to_string(),
                    Value::String(model_type.to_string()),
                );
            }

            // Add other relevant config fields to metadata
            if let Some(vocab_size) = config.get("vocab_size") {
                metadata.insert("vocab_size".to_string(), vocab_size.clone());
            }
            if let Some(hidden_size) = config.get("hidden_size") {
                metadata.insert("hidden_size".to_string(), hidden_size.clone());
            }
            if let Some(max_position_embeddings) = config.get("max_position_embeddings") {
                metadata.insert(
                    "max_position_embeddings".to_string(),
                    max_position_embeddings.clone(),
                );
            }
            if let Some(bos_token_id) = config.get("bos_token_id") {
                metadata.insert("bos_token_id".to_string(), bos_token_id.clone());
            }
            if let Some(eos_token_id) = config.get("eos_token_id") {
                metadata.insert("eos_token_id".to_string(), eos_token_id.clone());
            }
            if let Some(pad_token_id) = config.get("pad_token_id") {
                metadata.insert("pad_token_id".to_string(), pad_token_id.clone());
            }
        }

        // Add model format info
        metadata.insert("format".to_string(), Value::String("onnx".to_string()));
        metadata.insert(
            "path".to_string(),
            Value::String(path.display().to_string()),
        );

        Ok(Self { path, metadata })
    }

    /// Create a new ONNX model from a single .onnx file
    pub fn from_file<P: AsRef<Path>>(path: P) -> EdgeResult<Self> {
        let path = path.as_ref().to_path_buf();

        if !path.exists() {
            return Err(EdgeError::model(format!(
                "Model file does not exist: {}",
                path.display()
            )));
        }

        if path.extension().and_then(|e| e.to_str()) != Some("onnx") {
            return Err(EdgeError::model("File must have .onnx extension"));
        }

        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), Value::String("onnx".to_string()));
        metadata.insert(
            "path".to_string(),
            Value::String(path.display().to_string()),
        );

        Ok(Self { path, metadata })
    }

    /// Add metadata to the model
    pub fn with_metadata(mut self, key: String, value: Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

impl Model for OnnxModel {
    fn model_type(&self) -> &str {
        "onnx"
    }

    fn model_path(&self) -> &Path {
        &self.path
    }

    fn metadata(&self) -> &HashMap<String, Value> {
        &self.metadata
    }

    fn config(&self) -> EdgeResult<Value> {
        let config_path = self.path.join("config.json");
        if config_path.exists() {
            let config_content = std::fs::read_to_string(&config_path)?;
            let config: Value = serde_json::from_str(&config_content)?;
            Ok(config)
        } else {
            // Return basic config if no config.json exists
            Ok(serde_json::json!({
                "model_type": "onnx",
                "path": self.path.display().to_string()
            }))
        }
    }

    fn validate(&self) -> EdgeResult<()> {
        if !self.path.exists() {
            return Err(EdgeError::model(format!(
                "Model path does not exist: {}",
                self.path.display()
            )));
        }

        // Check for ONNX model file
        let onnx_file = if self.path.is_file() {
            // Direct .onnx file
            self.path.clone()
        } else {
            // Directory containing model.onnx
            self.path.join("model.onnx")
        };

        if !onnx_file.exists() {
            return Err(EdgeError::model(format!(
                "ONNX model file not found: {}",
                onnx_file.display()
            )));
        }

        Ok(())
    }
}

/// Builder for creating models
pub struct ModelBuilder;

impl ModelBuilder {
    /// Create an ONNX model from a directory
    pub fn onnx_from_directory<P: AsRef<Path>>(path: P) -> EdgeResult<OnnxModel> {
        OnnxModel::from_directory(path)
    }

    /// Create an ONNX model from a file
    pub fn onnx_from_file<P: AsRef<Path>>(path: P) -> EdgeResult<OnnxModel> {
        OnnxModel::from_file(path)
    }
}

/// Convenience function to create an ONNX model
pub fn onnx_model<P: AsRef<Path>>(path: P) -> EdgeResult<OnnxModel> {
    OnnxModel::from_directory(path)
}
