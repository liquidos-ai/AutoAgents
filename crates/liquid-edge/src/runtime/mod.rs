//! Runtime backends for liquid-edge inference
//!
//! This module provides different inference runtime backends that can be
//! enabled through feature flags. Each backend implements the core
//! `InferenceRuntime` trait to provide a consistent interface.
//!
//! ## Available Backends
//!
//! - **ONNX Runtime** (`onnx-runtime` feature): High-performance inference
//!   using Microsoft's ONNX Runtime with CPU and GPU acceleration support.
//!
//! ## Example
//!
//! ```rust,no_run
//! # #[cfg(feature = "onnx-runtime")]
//! # async fn example() -> liquid_edge::EdgeResult<()> {
//! use liquid_edge::runtime::{OnnxRuntime, OnnxInput};
//! use liquid_edge::traits::InferenceRuntime;
//!
//! // Create ONNX runtime
//! let runtime = OnnxRuntime::new("path/to/model.onnx", "my-model".to_string())?;
//!
//! // Prepare input
//! let input = OnnxInput::new(vec![1, 2, 3, 4, 5]);
//!
//! // Run inference
//! let output = runtime.infer(input).await?;
//! # Ok(())
//! # }
//! ```

use crate::error::{EdgeError, EdgeResult};
use crate::traits::ModelLoader;
use std::path::Path;

// Import runtime backends based on features
#[cfg(feature = "onnx-runtime")]
pub mod onnx;

#[cfg(feature = "onnx-runtime")]
pub use onnx::{OnnxInput, OnnxOutput, OnnxRuntime};

// Runtime factory for creating backends
pub struct RuntimeFactory;

impl RuntimeFactory {
    /// Create an ONNX runtime instance
    #[cfg(feature = "onnx-runtime")]
    pub fn create_onnx_runtime<P: AsRef<Path>>(
        model_path: P,
        model_name: impl Into<String>,
    ) -> EdgeResult<OnnxRuntime> {
        OnnxRuntime::new(model_path, model_name.into())
    }

    /// Create an ONNX runtime with custom options
    #[cfg(feature = "onnx-runtime")]
    pub fn create_onnx_runtime_with_options<P: AsRef<Path>>(
        model_path: P,
        model_name: impl Into<String>,
        num_threads: Option<usize>,
    ) -> EdgeResult<OnnxRuntime> {
        OnnxRuntime::with_options(model_path, model_name.into(), num_threads)
    }

    /// Auto-detect runtime based on model file extension
    #[cfg(feature = "onnx-runtime")]
    pub fn auto_detect_runtime<P: AsRef<Path>>(
        model_path: P,
        model_name: impl Into<String>,
    ) -> EdgeResult<OnnxRuntime> {
        let path = model_path.as_ref();
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        match extension.as_str() {
            "onnx" => Self::create_onnx_runtime(path, model_name),
            _ => Err(EdgeError::model(format!(
                "Unsupported model format: {extension}"
            ))),
        }
    }

    /// Get list of supported runtime backends
    pub fn supported_backends() -> Vec<&'static str> {
        #[cfg(feature = "onnx-runtime")]
        let backends = vec!["onnx"];
        #[cfg(not(feature = "onnx-runtime"))]
        let backends = vec![];

        backends
    }

    /// Check if a specific backend is available
    pub fn is_backend_available(backend: &str) -> bool {
        match backend {
            #[cfg(feature = "onnx-runtime")]
            "onnx" => true,
            #[cfg(not(feature = "onnx-runtime"))]
            "onnx" => false,
            _ => false,
        }
    }
}

/// Generic model loader that can handle multiple formats
#[cfg(feature = "onnx-runtime")]
pub struct GenericModelLoader;

#[cfg(feature = "onnx-runtime")]
impl ModelLoader for GenericModelLoader {
    type Runtime = OnnxRuntime;

    fn load_model<P: AsRef<Path>>(&self, model_path: P) -> EdgeResult<Self::Runtime> {
        // Extract model name from path
        let model_name = model_path
            .as_ref()
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown")
            .to_string();

        RuntimeFactory::auto_detect_runtime(model_path, model_name)
    }

    fn supports_format<P: AsRef<Path>>(&self, model_path: P) -> bool {
        let path = model_path.as_ref();
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        self.supported_extensions().contains(&extension.as_str())
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        #[cfg(feature = "onnx-runtime")]
        let extensions = vec!["onnx"];
        #[cfg(not(feature = "onnx-runtime"))]
        let extensions = vec![];

        extensions
    }
}

/// Runtime configuration builder
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub num_threads: Option<usize>,
    pub memory_pattern: bool,
    pub enable_profiling: bool,
    pub optimization_level: OptimizationLevel,
}

/// Optimization levels for runtime performance tuning
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No optimization - fastest startup
    None,
    /// Basic optimizations - balanced startup and performance
    Basic,
    /// All optimizations - best performance, slower startup
    All,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            num_threads: None, // Auto-detect
            memory_pattern: true,
            enable_profiling: false,
            optimization_level: OptimizationLevel::Basic,
        }
    }
}

impl RuntimeConfig {
    /// Create a new runtime configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of threads for inference
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = Some(num_threads);
        self
    }

    /// Enable or disable memory pattern optimization
    pub fn with_memory_pattern(mut self, enable: bool) -> Self {
        self.memory_pattern = enable;
        self
    }

    /// Enable or disable profiling
    pub fn with_profiling(mut self, enable: bool) -> Self {
        self.enable_profiling = enable;
        self
    }

    /// Set optimization level
    pub fn with_optimization_level(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }

    /// Create a high-performance configuration
    pub fn high_performance() -> Self {
        Self {
            num_threads: None, // Use all available threads
            memory_pattern: true,
            enable_profiling: false,
            optimization_level: OptimizationLevel::All,
        }
    }

    /// Create a low-latency configuration
    pub fn low_latency() -> Self {
        Self {
            num_threads: Some(1), // Single thread for predictable latency
            memory_pattern: false,
            enable_profiling: false,
            optimization_level: OptimizationLevel::Basic,
        }
    }

    /// Create a memory-efficient configuration
    pub fn memory_efficient() -> Self {
        Self {
            num_threads: Some(2), // Limited threads to save memory
            memory_pattern: true,
            enable_profiling: false,
            optimization_level: OptimizationLevel::None,
        }
    }
}

/// Runtime builder for creating configured runtime instances
pub struct RuntimeBuilder {
    config: RuntimeConfig,
}

impl RuntimeBuilder {
    /// Create a new runtime builder
    pub fn new() -> Self {
        Self {
            config: RuntimeConfig::default(),
        }
    }

    /// Create a runtime builder with custom configuration
    pub fn with_config(config: RuntimeConfig) -> Self {
        Self { config }
    }

    /// Set runtime configuration
    pub fn config(mut self, config: RuntimeConfig) -> Self {
        self.config = config;
        self
    }

    /// Build an ONNX runtime
    #[cfg(feature = "onnx-runtime")]
    pub fn build_onnx<P: AsRef<Path>>(
        &self,
        model_path: P,
        model_name: impl Into<String>,
    ) -> EdgeResult<OnnxRuntime> {
        OnnxRuntime::with_options(model_path, model_name.into(), self.config.num_threads)
    }

    /// Build runtime using auto-detection
    #[cfg(feature = "onnx-runtime")]
    pub fn build_auto<P: AsRef<Path>>(
        &self,
        model_path: P,
        model_name: impl Into<String>,
    ) -> EdgeResult<OnnxRuntime> {
        // For now, this just delegates to RuntimeFactory
        // In the future, it could apply the configuration to the created runtime
        RuntimeFactory::auto_detect_runtime(model_path, model_name)
    }
}

impl Default for RuntimeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_factory_supported_backends() {
        let backends = RuntimeFactory::supported_backends();

        #[cfg(feature = "onnx-runtime")]
        assert!(backends.contains(&"onnx"));

        #[cfg(not(feature = "onnx-runtime"))]
        assert!(backends.is_empty());
    }

    #[test]
    fn test_runtime_config_builder() {
        let config = RuntimeConfig::new()
            .with_threads(4)
            .with_memory_pattern(true)
            .with_optimization_level(OptimizationLevel::All);

        assert_eq!(config.num_threads, Some(4));
        assert!(config.memory_pattern);
        assert_eq!(config.optimization_level, OptimizationLevel::All);
    }

    #[test]
    fn test_predefined_configs() {
        let hp_config = RuntimeConfig::high_performance();
        assert_eq!(hp_config.optimization_level, OptimizationLevel::All);

        let ll_config = RuntimeConfig::low_latency();
        assert_eq!(ll_config.num_threads, Some(1));

        let me_config = RuntimeConfig::memory_efficient();
        assert_eq!(me_config.optimization_level, OptimizationLevel::None);
    }

    #[cfg(feature = "onnx-runtime")]
    #[test]
    fn test_backend_availability() {
        assert!(RuntimeFactory::is_backend_available("onnx"));
    }

    #[cfg(not(feature = "onnx-runtime"))]
    #[test]
    fn test_backend_unavailability() {
        assert!(!RuntimeFactory::is_backend_available("onnx"));
    }
}
