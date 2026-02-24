use super::VadEngine;
use super::config::VadConfig;
use super::error::{VadError, VadResult};
use super::result::VadOutput;
use super::session::create_session;
use crate::model_source::ModelSource;
use ndarray::{Array1, Array2, Array3, ArrayBase, Ix3, OwnedRepr};
use ort::session::Session;
use ort::value::Value;

/// Silero VAD engine backed by an ONNX model.
pub struct SileroVad {
    session: Session,
    h_tensor: ArrayBase<OwnedRepr<f32>, Ix3>,
    c_tensor: ArrayBase<OwnedRepr<f32>, Ix3>,
    sample_rate_i64: i64,
    sample_rate: u32,
}

impl SileroVad {
    pub fn new(model_source: ModelSource, config: VadConfig) -> VadResult<Self> {
        config.validate()?;
        let model_path = model_source.resolve()?;
        let session = create_session(&model_path)?;
        let h_tensor = Array3::<f32>::zeros((2, 1, 64));
        let c_tensor = Array3::<f32>::zeros((2, 1, 64));

        Ok(Self {
            session,
            h_tensor,
            c_tensor,
            sample_rate_i64: config.sample_rate as i64,
            sample_rate: config.sample_rate,
        })
    }

    pub fn from_file(path: impl Into<std::path::PathBuf>, config: VadConfig) -> VadResult<Self> {
        Self::new(ModelSource::from_file(path), config)
    }

    pub fn from_hf(
        repo_id: impl Into<String>,
        filename: impl Into<String>,
        config: VadConfig,
    ) -> VadResult<Self> {
        Self::new(ModelSource::from_hf(repo_id, filename), config)
    }

    pub fn reset(&mut self) {
        self.h_tensor.fill(0.0);
        self.c_tensor.fill(0.0);
    }
}

impl VadEngine for SileroVad {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn reset(&mut self) {
        self.reset();
    }

    fn compute(&mut self, samples: &[f32]) -> VadResult<VadOutput> {
        if samples.is_empty() {
            return Err(VadError::InvalidInput(
                "VAD input must contain at least one sample".to_string(),
            ));
        }

        let samples_tensor = Array2::from_shape_vec((1, samples.len()), samples.to_vec())
            .map_err(|err| VadError::InvalidInput(err.to_string()))?;
        let samples_value = Value::from_array(samples_tensor)
            .map_err(|err| VadError::Inference(err.to_string()))?;
        let sr_value = Value::from_array(Array1::from_elem(1, self.sample_rate_i64))
            .map_err(|err| VadError::Inference(err.to_string()))?;
        let h_value = Value::from_array(self.h_tensor.clone())
            .map_err(|err| VadError::Inference(err.to_string()))?;
        let c_value = Value::from_array(self.c_tensor.clone())
            .map_err(|err| VadError::Inference(err.to_string()))?;

        let result = self
            .session
            .run(ort::inputs![
                "input" => samples_value,
                "sr" => sr_value,
                "h" => h_value,
                "c" => c_value
            ])
            .map_err(|err| VadError::Inference(err.to_string()))?;

        let h_output = result
            .get("hn")
            .ok_or_else(|| VadError::Inference("missing output 'hn'".to_string()))?
            .try_extract_tensor::<f32>()
            .map_err(|err| VadError::Inference(err.to_string()))?;
        self.h_tensor = Array3::from_shape_vec((2, 1, 64), h_output.1.to_vec())
            .map_err(|err| VadError::Inference(err.to_string()))?;

        let c_output = result
            .get("cn")
            .ok_or_else(|| VadError::Inference("missing output 'cn'".to_string()))?
            .try_extract_tensor::<f32>()
            .map_err(|err| VadError::Inference(err.to_string()))?;
        self.c_tensor = Array3::from_shape_vec((2, 1, 64), c_output.1.to_vec())
            .map_err(|err| VadError::Inference(err.to_string()))?;

        let output = result
            .get("output")
            .ok_or_else(|| VadError::Inference("missing output 'output'".to_string()))?
            .try_extract_tensor::<f32>()
            .map_err(|err| VadError::Inference(err.to_string()))?;
        let prob = output.1.first().copied().unwrap_or(0.0);

        Ok(VadOutput { probability: prob })
    }
}
