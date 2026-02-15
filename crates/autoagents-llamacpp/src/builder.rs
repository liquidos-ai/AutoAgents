use autoagents_llm::error::LLMError;

use crate::{LlamaCppConfigBuilder, LlamaCppProvider, ModelSource};

/// Builder for LlamaCppProvider.
#[derive(Default)]
pub struct LlamaCppProviderBuilder {
    config_builder: LlamaCppConfigBuilder,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LlamaCppSplitMode;
    use crate::models::ModelSource;

    #[test]
    fn builder_maps_config_fields() {
        let builder = LlamaCppProviderBuilder::default()
            .model_source(ModelSource::gguf("model.gguf"))
            .chat_template("chat")
            .system_prompt("system")
            .force_json_grammar(true)
            .model_dir("/models")
            .hf_filename("model-file.gguf")
            .hf_revision("rev1")
            .mmproj_path("mmproj.gguf")
            .media_marker("<image>")
            .mmproj_use_gpu(true)
            .max_tokens(321)
            .temperature(0.2)
            .top_p(0.9)
            .top_k(42)
            .frequency_penalty(0.1)
            .presence_penalty(0.2)
            .repeat_last_n(64)
            .seed(7)
            .n_ctx(2048)
            .n_batch(16)
            .n_ubatch(8)
            .n_threads(2)
            .n_threads_batch(4)
            .n_gpu_layers(5)
            .main_gpu(1)
            .split_mode(LlamaCppSplitMode::Row)
            .use_mlock(true)
            .devices(vec![0, 1]);

        let config = builder.config_builder.build();
        assert_eq!(config.model_source, ModelSource::gguf("model.gguf"));
        assert_eq!(config.chat_template.as_deref(), Some("chat"));
        assert_eq!(config.system_prompt.as_deref(), Some("system"));
        assert!(config.force_json_grammar);
        assert_eq!(config.model_dir.as_deref(), Some("/models"));
        assert_eq!(config.hf_filename.as_deref(), Some("model-file.gguf"));
        assert_eq!(config.hf_revision.as_deref(), Some("rev1"));
        assert_eq!(config.mmproj_path.as_deref(), Some("mmproj.gguf"));
        assert_eq!(config.media_marker.as_deref(), Some("<image>"));
        assert_eq!(config.mmproj_use_gpu, Some(true));
        assert_eq!(config.max_tokens, Some(321));
        assert_eq!(config.temperature, Some(0.2));
        assert_eq!(config.top_p, Some(0.9));
        assert_eq!(config.top_k, Some(42));
        assert_eq!(config.frequency_penalty, Some(0.1));
        assert_eq!(config.presence_penalty, Some(0.2));
        assert_eq!(config.repeat_last_n, Some(64));
        assert_eq!(config.seed, Some(7));
        assert_eq!(config.n_ctx, Some(2048));
        assert_eq!(config.n_batch, Some(16));
        assert_eq!(config.n_ubatch, Some(8));
        assert_eq!(config.n_threads, Some(2));
        assert_eq!(config.n_threads_batch, Some(4));
        assert_eq!(config.n_gpu_layers, Some(5));
        assert_eq!(config.main_gpu, Some(1));
        assert_eq!(config.split_mode, Some(LlamaCppSplitMode::Row));
        assert_eq!(config.use_mlock, Some(true));
        assert_eq!(config.devices, Some(vec![0, 1]));
    }
}
