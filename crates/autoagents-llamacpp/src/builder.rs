use autoagents_llm::error::LLMError;

use crate::{LlamaCppConfigBuilder, LlamaCppProvider, ModelSource};

/// Builder for LlamaCppProvider.
pub struct LlamaCppProviderBuilder {
    config_builder: LlamaCppConfigBuilder,
}

impl Default for LlamaCppProviderBuilder {
    fn default() -> Self {
        Self {
            config_builder: LlamaCppConfigBuilder::new(),
        }
    }
}

impl LlamaCppProviderBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model source.
    pub fn model_source(mut self, source: ModelSource) -> Self {
        self.config_builder = self.config_builder.model_source(source);
        self
    }

    /// Set model path for GGUF.
    pub fn model_path(mut self, path: impl Into<String>) -> Self {
        self.config_builder = self.config_builder.model_path(path);
        self
    }

    /// Set chat template.
    pub fn chat_template(mut self, template: impl Into<String>) -> Self {
        self.config_builder = self.config_builder.chat_template(template);
        self
    }

    /// Set system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config_builder = self.config_builder.system_prompt(prompt);
        self
    }

    /// Force JSON grammar enforcement even without a structured output schema.
    pub fn force_json_grammar(mut self, force: bool) -> Self {
        self.config_builder = self.config_builder.force_json_grammar(force);
        self
    }

    /// Set the model cache directory.
    pub fn model_dir(mut self, dir: impl Into<String>) -> Self {
        self.config_builder = self.config_builder.model_dir(dir);
        self
    }

    /// Set the HuggingFace filename (GGUF file).
    pub fn hf_filename(mut self, filename: impl Into<String>) -> Self {
        self.config_builder = self.config_builder.hf_filename(filename);
        self
    }

    /// Set the HuggingFace revision.
    pub fn hf_revision(mut self, revision: impl Into<String>) -> Self {
        self.config_builder = self.config_builder.hf_revision(revision);
        self
    }

    /// Set the multimodal projection (mmproj) file path.
    pub fn mmproj_path(mut self, path: impl Into<String>) -> Self {
        self.config_builder = self.config_builder.mmproj_path(path);
        self
    }

    /// Set MTMD media marker.
    pub fn media_marker(mut self, marker: impl Into<String>) -> Self {
        self.config_builder = self.config_builder.media_marker(marker);
        self
    }

    /// Enable or disable GPU offload for MTMD projection.
    pub fn mmproj_use_gpu(mut self, use_gpu: bool) -> Self {
        self.config_builder = self.config_builder.mmproj_use_gpu(use_gpu);
        self
    }

    /// Set max tokens.
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.config_builder = self.config_builder.max_tokens(tokens);
        self
    }

    /// Set temperature.
    pub fn temperature(mut self, temp: f32) -> Self {
        self.config_builder = self.config_builder.temperature(temp);
        self
    }

    /// Set top-p.
    pub fn top_p(mut self, p: f32) -> Self {
        self.config_builder = self.config_builder.top_p(p);
        self
    }

    /// Set top-k.
    pub fn top_k(mut self, k: u32) -> Self {
        self.config_builder = self.config_builder.top_k(k);
        self
    }

    /// Set frequency penalty.
    pub fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.config_builder = self.config_builder.frequency_penalty(penalty);
        self
    }

    /// Set presence penalty.
    pub fn presence_penalty(mut self, penalty: f32) -> Self {
        self.config_builder = self.config_builder.presence_penalty(penalty);
        self
    }

    /// Set repeat last N for penalties.
    pub fn repeat_last_n(mut self, last_n: i32) -> Self {
        self.config_builder = self.config_builder.repeat_last_n(last_n);
        self
    }

    /// Set sampling seed.
    pub fn seed(mut self, seed: u32) -> Self {
        self.config_builder = self.config_builder.seed(seed);
        self
    }

    /// Set context size.
    pub fn n_ctx(mut self, n_ctx: u32) -> Self {
        self.config_builder = self.config_builder.n_ctx(n_ctx);
        self
    }

    /// Set batch size.
    pub fn n_batch(mut self, n_batch: u32) -> Self {
        self.config_builder = self.config_builder.n_batch(n_batch);
        self
    }

    /// Set micro-batch size.
    pub fn n_ubatch(mut self, n_ubatch: u32) -> Self {
        self.config_builder = self.config_builder.n_ubatch(n_ubatch);
        self
    }

    /// Set number of threads for prompt evaluation.
    pub fn n_threads(mut self, n_threads: i32) -> Self {
        self.config_builder = self.config_builder.n_threads(n_threads);
        self
    }

    /// Set number of threads for batch evaluation.
    pub fn n_threads_batch(mut self, n_threads: i32) -> Self {
        self.config_builder = self.config_builder.n_threads_batch(n_threads);
        self
    }

    /// Set number of GPU layers to offload.
    pub fn n_gpu_layers(mut self, layers: u32) -> Self {
        self.config_builder = self.config_builder.n_gpu_layers(layers);
        self
    }

    /// Set main GPU index.
    pub fn main_gpu(mut self, main_gpu: i32) -> Self {
        self.config_builder = self.config_builder.main_gpu(main_gpu);
        self
    }

    /// Set split mode.
    pub fn split_mode(mut self, mode: crate::config::LlamaCppSplitMode) -> Self {
        self.config_builder = self.config_builder.split_mode(mode);
        self
    }

    /// Enable memory lock.
    pub fn use_mlock(mut self, use_mlock: bool) -> Self {
        self.config_builder = self.config_builder.use_mlock(use_mlock);
        self
    }

    /// Set explicit device indices for offload.
    pub fn devices(mut self, devices: Vec<usize>) -> Self {
        self.config_builder = self.config_builder.devices(devices);
        self
    }

    /// Set repeat penalty.
    pub fn repeat_penalty(mut self, penalty: f32) -> Self {
        self.config_builder = self.config_builder.repeat_penalty(penalty);
        self
    }

    /// Build the provider.
    pub async fn build(self) -> Result<LlamaCppProvider, LLMError> {
        let config = self.config_builder.build();
        LlamaCppProvider::from_config(config).await
    }
}
