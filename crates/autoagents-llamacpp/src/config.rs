//! Configuration structures for llama.cpp provider.

use crate::models::ModelSource;
use llama_cpp_2::model::params::LlamaSplitMode;
use serde::{Deserialize, Serialize};

/// Serializable split mode wrapper for llama.cpp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlamaCppSplitMode {
    /// Single device.
    None,
    /// Split layers and KV across GPUs.
    Layer,
    /// Split layers and KV across GPUs, use tensor parallelism if supported.
    Row,
}

impl From<LlamaCppSplitMode> for LlamaSplitMode {
    fn from(value: LlamaCppSplitMode) -> Self {
        match value {
            LlamaCppSplitMode::None => LlamaSplitMode::None,
            LlamaCppSplitMode::Layer => LlamaSplitMode::Layer,
            LlamaCppSplitMode::Row => LlamaSplitMode::Row,
        }
    }
}

/// Complete configuration for LlamaCppProvider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppConfig {
    /// Model source (GGUF path).
    pub model_source: ModelSource,

    /// Optional chat template name or inline template.
    pub chat_template: Option<String>,

    /// Optional system prompt to prepend if no system message exists.
    pub system_prompt: Option<String>,

    /// Force JSON grammar enforcement even without a structured output schema.
    pub force_json_grammar: bool,

    /// Optional HuggingFace cache directory (defaults to HF_HOME or ~/.cache/huggingface/hub).
    pub model_dir: Option<String>,

    /// Optional HuggingFace filename override (GGUF file).
    pub hf_filename: Option<String>,

    /// Optional HuggingFace revision (defaults to "main").
    pub hf_revision: Option<String>,

    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,

    /// Sampling temperature (0.0 - 2.0).
    pub temperature: Option<f32>,

    /// Top-p sampling parameter.
    pub top_p: Option<f32>,

    /// Top-k sampling parameter.
    pub top_k: Option<u32>,

    /// Repeat penalty (1.0 disables).
    pub repeat_penalty: Option<f32>,

    /// Penalize frequency of tokens (0.0 disables).
    pub frequency_penalty: Option<f32>,

    /// Penalize presence of tokens (0.0 disables).
    pub presence_penalty: Option<f32>,

    /// Number of tokens to consider for penalties (None = default 64).
    pub repeat_last_n: Option<i32>,

    /// RNG seed for sampling.
    pub seed: Option<u32>,

    /// Context size override.
    pub n_ctx: Option<u32>,

    /// Batch size override.
    pub n_batch: Option<u32>,

    /// Micro-batch size override.
    pub n_ubatch: Option<u32>,

    /// Number of threads for prompt evaluation.
    pub n_threads: Option<i32>,

    /// Number of threads for batch evaluation.
    pub n_threads_batch: Option<i32>,

    /// Number of GPU layers to offload.
    pub n_gpu_layers: Option<u32>,

    /// Main GPU index.
    pub main_gpu: Option<i32>,

    /// Split mode for multi-GPU.
    pub split_mode: Option<LlamaCppSplitMode>,

    /// Enable memory lock (mlock) if supported.
    pub use_mlock: Option<bool>,

    /// Explicit device indices for offload.
    pub devices: Option<Vec<usize>>,
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            model_source: ModelSource::Gguf {
                model_path: String::new(),
            },
            chat_template: None,
            system_prompt: None,
            force_json_grammar: false,
            model_dir: None,
            hf_filename: None,
            hf_revision: None,
            max_tokens: Some(512),
            temperature: Some(0.7),
            top_p: None,
            top_k: None,
            repeat_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            repeat_last_n: None,
            seed: None,
            n_ctx: None,
            n_batch: None,
            n_ubatch: None,
            n_threads: None,
            n_threads_batch: None,
            n_gpu_layers: None,
            main_gpu: None,
            split_mode: None,
            use_mlock: None,
            devices: None,
        }
    }
}

/// Builder for LlamaCppConfig.
#[derive(Debug)]
pub struct LlamaCppConfigBuilder {
    config: LlamaCppConfig,
}

impl LlamaCppConfigBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: LlamaCppConfig::default(),
        }
    }

    /// Set the model source.
    pub fn model_source(mut self, source: ModelSource) -> Self {
        self.config.model_source = source;
        self
    }

    /// Set the model path for a local GGUF model.
    pub fn model_path(mut self, path: impl Into<String>) -> Self {
        self.config.model_source = ModelSource::gguf(path);
        self
    }

    /// Set chat template.
    pub fn chat_template(mut self, template: impl Into<String>) -> Self {
        self.config.chat_template = Some(template.into());
        self
    }

    /// Set system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.system_prompt = Some(prompt.into());
        self
    }

    /// Force JSON grammar enforcement even without a structured output schema.
    pub fn force_json_grammar(mut self, force: bool) -> Self {
        self.config.force_json_grammar = force;
        self
    }

    /// Set the HuggingFace cache directory.
    pub fn model_dir(mut self, dir: impl Into<String>) -> Self {
        self.config.model_dir = Some(dir.into());
        self
    }

    /// Set the HuggingFace filename (GGUF file).
    pub fn hf_filename(mut self, filename: impl Into<String>) -> Self {
        self.config.hf_filename = Some(filename.into());
        self
    }

    /// Set the HuggingFace revision.
    pub fn hf_revision(mut self, revision: impl Into<String>) -> Self {
        self.config.hf_revision = Some(revision.into());
        self
    }

    /// Set maximum tokens to generate.
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.config.max_tokens = Some(tokens);
        self
    }

    /// Set sampling temperature.
    pub fn temperature(mut self, temp: f32) -> Self {
        self.config.temperature = Some(temp);
        self
    }

    /// Set top-p sampling parameter.
    pub fn top_p(mut self, p: f32) -> Self {
        self.config.top_p = Some(p);
        self
    }

    /// Set top-k sampling parameter.
    pub fn top_k(mut self, k: u32) -> Self {
        self.config.top_k = Some(k);
        self
    }

    /// Set repeat penalty.
    pub fn repeat_penalty(mut self, penalty: f32) -> Self {
        self.config.repeat_penalty = Some(penalty);
        self
    }

    /// Set frequency penalty.
    pub fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.config.frequency_penalty = Some(penalty);
        self
    }

    /// Set presence penalty.
    pub fn presence_penalty(mut self, penalty: f32) -> Self {
        self.config.presence_penalty = Some(penalty);
        self
    }

    /// Set repeat last N for penalties.
    pub fn repeat_last_n(mut self, last_n: i32) -> Self {
        self.config.repeat_last_n = Some(last_n);
        self
    }

    /// Set sampling seed.
    pub fn seed(mut self, seed: u32) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Set context size.
    pub fn n_ctx(mut self, n_ctx: u32) -> Self {
        self.config.n_ctx = Some(n_ctx);
        self
    }

    /// Set batch size.
    pub fn n_batch(mut self, n_batch: u32) -> Self {
        self.config.n_batch = Some(n_batch);
        self
    }

    /// Set micro-batch size.
    pub fn n_ubatch(mut self, n_ubatch: u32) -> Self {
        self.config.n_ubatch = Some(n_ubatch);
        self
    }

    /// Set number of threads for prompt evaluation.
    pub fn n_threads(mut self, n_threads: i32) -> Self {
        self.config.n_threads = Some(n_threads);
        self
    }

    /// Set number of threads for batch evaluation.
    pub fn n_threads_batch(mut self, n_threads: i32) -> Self {
        self.config.n_threads_batch = Some(n_threads);
        self
    }

    /// Set number of GPU layers to offload.
    pub fn n_gpu_layers(mut self, layers: u32) -> Self {
        self.config.n_gpu_layers = Some(layers);
        self
    }

    /// Set main GPU index.
    pub fn main_gpu(mut self, main_gpu: i32) -> Self {
        self.config.main_gpu = Some(main_gpu);
        self
    }

    /// Set split mode.
    pub fn split_mode(mut self, mode: LlamaCppSplitMode) -> Self {
        self.config.split_mode = Some(mode);
        self
    }

    /// Enable memory lock.
    pub fn use_mlock(mut self, use_mlock: bool) -> Self {
        self.config.use_mlock = Some(use_mlock);
        self
    }

    /// Set explicit device indices for offload.
    pub fn devices(mut self, devices: Vec<usize>) -> Self {
        self.config.devices = Some(devices);
        self
    }

    /// Build the configuration.
    pub fn build(self) -> LlamaCppConfig {
        self.config
    }
}

impl Default for LlamaCppConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LlamaCppConfig::default();
        assert_eq!(config.max_tokens, Some(512));
        assert_eq!(config.temperature, Some(0.7));
        assert!(!config.force_json_grammar);
        assert!(config.model_dir.is_none());
        assert!(config.hf_filename.is_none());
        assert!(config.hf_revision.is_none());
    }

    #[test]
    fn test_config_builder_basic() {
        let config = LlamaCppConfigBuilder::new()
            .model_path("model.gguf")
            .max_tokens(1024)
            .temperature(0.8)
            .build();

        assert_eq!(
            config.model_source,
            ModelSource::Gguf {
                model_path: "model.gguf".to_string(),
            }
        );
        assert_eq!(config.max_tokens, Some(1024));
        assert_eq!(config.temperature, Some(0.8));
    }

    #[test]
    fn test_config_builder_split_mode() {
        let config = LlamaCppConfigBuilder::new()
            .model_path("model.gguf")
            .split_mode(LlamaCppSplitMode::Row)
            .build();

        assert_eq!(config.split_mode, Some(LlamaCppSplitMode::Row));
    }

    #[test]
    fn test_config_builder_default() {
        let builder1 = LlamaCppConfigBuilder::default();
        let builder2 = LlamaCppConfigBuilder::new();

        let config1 = builder1.build();
        let config2 = builder2.build();

        assert_eq!(config1.max_tokens, config2.max_tokens);
        assert_eq!(config1.temperature, config2.temperature);
    }
}
