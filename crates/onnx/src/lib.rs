//! # Liquid Edge - Generic Edge Inference Runtime
//!
//! A lightweight, efficient inference runtime designed for edge computing environments.
//! Supports multiple backends for running deep learning models on edge devices.

pub mod device;
pub mod error;
pub mod model;
pub mod runtime;

pub mod chat;

// Re-exports
pub use device::{cpu, cpu_with_threads, Device};

#[cfg(feature = "cuda")]
pub use device::{cuda, cuda_default};

pub use error::{EdgeError, EdgeResult};
pub use model::Model;
pub use runtime::{
    inference::{OnnxBackend, OnnxModel},
    onnx_model, InferenceInput, InferenceOutput, InferenceRuntime,
};
