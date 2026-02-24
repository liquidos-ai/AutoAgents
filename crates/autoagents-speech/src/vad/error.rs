use crate::model_source::ModelSourceError;

#[derive(Debug, thiserror::Error)]
pub enum VadError {
    #[error("Unsupported sample rate: {0}. Supported rates are 8000 or 16000 Hz.")]
    UnsupportedSampleRate(u32),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Model source error: {0}")]
    ModelSource(#[from] ModelSourceError),
    #[error("Model load failed: {0}")]
    ModelLoad(String),
    #[error("Model inference failed: {0}")]
    Inference(String),
}

pub type VadResult<T> = Result<T, VadError>;
