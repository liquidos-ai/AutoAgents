//! # Liquid Edge - Generic Edge Inference Runtime
//!
//! A lightweight, efficient inference runtime designed for edge computing environments.
//! Supports multiple backends for running deep learning models on edge devices.

pub mod device;
pub mod error;
pub mod model;
pub mod runtime;

// Re-exports
pub use device::{cpu, cpu_with_threads, cuda, cuda_default, Device};
pub use error::{EdgeError, EdgeResult};
pub use model::Model;
pub use runtime::{InferenceInput, InferenceOutput, InferenceRuntime, RuntimeBackend};
