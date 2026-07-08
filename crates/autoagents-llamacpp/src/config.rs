//! Configuration structures for llama.cpp provider.

use crate::models::ModelSource;
use llama_cpp_2::model::params::LlamaSplitMode;
use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self},
};
use serde_json::{Value, json};

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

/// Reasoning extraction format for llama.cpp OpenAI-compatible parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlamaCppReasoningFormat {
    /// Disable reasoning extraction into `reasoning_content`.
    None,
    /// Let llama.cpp auto-detect the model/template strategy.
    Auto,
    /// Parse DeepSeek/Qwen-style thinking into `reasoning_content`.
    Deepseek,
    /// Legacy DeepSeek behavior.
    DeepseekLegacy,
}

/// Tool-call selection behavior for llama.cpp chat completions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlamaCppToolChoice {
    /// Let the model decide whether to call a tool.
    Auto,
    /// Force the model to call a tool when tools are provided.
    Required,
    /// Disable tool calls even when tools are provided.
    None,
    /// Force a specific function tool by name.
    Function { name: String },
}

impl Serialize for LlamaCppToolChoice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Auto => serializer.serialize_str("auto"),
            Self::Required => serializer.serialize_str("required"),
            Self::None => serializer.serialize_str("none"),
            Self::Function { name } => json!({
                "type": "function",
                "function": {
                    "name": name
                }
            })
            .serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for LlamaCppToolChoice {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(choice) => match choice.as_str() {
                "auto" => Ok(Self::Auto),
                "required" => Ok(Self::Required),
                "none" => Ok(Self::None),
                other => Err(de::Error::custom(format!(
                    "unsupported llama.cpp tool_choice string `{other}`"
                ))),
            },
            Value::Object(object) => {
                let choice_type = object.get("type").and_then(Value::as_str);
                if choice_type.is_some_and(|value| value != "function") {
                    return Err(de::Error::custom(
                        "llama.cpp tool_choice object type must be `function`",
                    ));
                }
                let name = object
                    .get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(Value::as_str)
                    .filter(|name| !name.trim().is_empty())
                    .ok_or_else(|| {
                        de::Error::custom("llama.cpp function tool_choice requires `function.name`")
                    })?;
                Ok(Self::Function {
                    name: name.to_string(),
                })
            }
            _ => Err(de::Error::custom(
                "llama.cpp tool_choice must be `auto`, `required`, `none`, or a function object",
            )),
        }
    }
}

/// Continuation behavior for the final chat message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlamaCppChatContinuation {
    /// Do not continue the final message.
    None,
    /// Match llama.cpp server auto mode.
    Auto,
    /// Continue the final message as assistant content.
    Content,
    /// Continue the final message as assistant reasoning.
    Reasoning,
}

/// Embedded scheduler settings for llama.cpp inference.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LlamaCppSchedulerConfig {
    /// Maximum number of queued requests, excluding active requests.
    pub queue_capacity: usize,
    /// Number of concurrent inference slots.
    pub n_slots: usize,
}

impl Default for LlamaCppSchedulerConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 1024,
            n_slots: 1,
        }
    }
}

impl LlamaCppReasoningFormat {
    /// Convert to llama.cpp reasoning format string.
    pub fn as_str(self) -> Option<&'static str> {
        match self {
            Self::None => None,
            Self::Auto => Some("auto"),
            Self::Deepseek => Some("deepseek"),
            Self::DeepseekLegacy => Some("deepseek-legacy"),
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

    /// Render and parse chat completions as pure assistant content.
    ///
    /// Mirrors llama.cpp server's `force_pure_content`: the prompt is rendered
    /// without reasoning extraction, and generated text is not split into
    /// reasoning/tool-call structures.
    pub force_pure_content: bool,

    /// Override the model tokenizer metadata that tells llama.cpp to add BOS
    /// during chat tokenization.
    ///
    /// The upstream Rust binding does not expose `llama_vocab_get_add_bos`, so
    /// this allows production deployments to mirror llama.cpp server behavior
    /// for models that need exact duplicate-BOS stripping.
    pub tokenizer_add_bos: Option<bool>,

    /// Override the model tokenizer metadata that tells llama.cpp to add EOS
    /// during chat tokenization.
    ///
    /// The upstream Rust binding does not expose `llama_vocab_get_add_eos`, so
    /// this allows production deployments to mirror llama.cpp server behavior
    /// for models that need exact duplicate-EOS stripping.
    pub tokenizer_add_eos: Option<bool>,

    /// Reasoning extraction mode for structured `reasoning_content`.
    pub reasoning_format: Option<LlamaCppReasoningFormat>,

    /// Optional provider-specific request metadata.
    ///
    /// The llama.cpp backend consumes `chat_template_kwargs` from this object
    /// when rendering Jinja chat templates.
    pub extra_body: Option<serde_json::Value>,

    /// Optional HuggingFace cache directory (defaults to HF_HOME or ~/.cache/huggingface/hub).
    pub model_dir: Option<String>,

    /// Optional HuggingFace filename override (GGUF file).
    pub hf_filename: Option<String>,

    /// Optional HuggingFace revision (defaults to "main").
    pub hf_revision: Option<String>,

    /// Optional multimodal projection file for MTMD models.
    pub mmproj_path: Option<String>,

    /// Optional MTMD media marker override.
    pub media_marker: Option<String>,

    /// Enable GPU offload for MTMD projection.
    pub mmproj_use_gpu: Option<bool>,

    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,

    /// Sampling temperature (0.0 - 2.0).
    pub temperature: Option<f32>,

    /// Top-p sampling parameter.
    pub top_p: Option<f32>,

    /// Top-k sampling parameter.
    pub top_k: Option<u32>,

    /// Minimum-p sampling parameter.
    pub min_p: Option<f32>,

    /// Locally typical sampling parameter.
    pub typical_p: Option<f32>,

    /// Top-n-sigma sampling parameter.
    pub top_n_sigma: Option<f32>,

    /// XTC probability.
    pub xtc_probability: Option<f32>,

    /// XTC threshold.
    pub xtc_threshold: Option<f32>,

    /// Dynamic temperature range.
    pub dynatemp_range: Option<f32>,

    /// Dynamic temperature exponent.
    pub dynatemp_exponent: Option<f32>,

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

    /// Minimum candidates to keep for samplers that support it.
    pub min_keep: Option<usize>,

    /// Mirostat mode: 1 = v1, 2 = v2.
    pub mirostat: Option<u8>,

    /// Mirostat target entropy.
    pub mirostat_tau: Option<f32>,

    /// Mirostat learning rate.
    pub mirostat_eta: Option<f32>,

    /// DRY repetition penalty multiplier.
    pub dry_multiplier: Option<f32>,

    /// DRY repetition penalty base.
    pub dry_base: Option<f32>,

    /// DRY allowed repeated length.
    pub dry_allowed_length: Option<i32>,

    /// DRY penalty lookback.
    pub dry_penalty_last_n: Option<i32>,

    /// DRY sequence breakers.
    pub dry_sequence_breakers: Option<Vec<String>>,

    /// Per-token logit biases.
    pub logit_bias: Option<Vec<(i32, f32)>>,

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

    /// Enable thinking/reasoning tokens in chat template.
    ///
    /// This is passed as template context (`enable_thinking`) and is never
    /// rewritten into model-specific prompt text.
    pub enable_thinking: Option<bool>,

    /// Add the assistant generation prompt while rendering chat templates.
    pub add_generation_prompt: bool,

    /// Continue the final chat message instead of adding a new assistant
    /// generation prompt.
    pub continue_final_message: LlamaCppChatContinuation,

    /// Tool-call choice behavior.
    pub tool_choice: LlamaCppToolChoice,

    /// Allow native parallel tool calls when the template supports them.
    pub parallel_tool_calls: Option<bool>,

    /// Embedded queue/slot scheduler configuration.
    pub scheduler: LlamaCppSchedulerConfig,

    /// Enable KV-cache prefix reuse across inference calls.
    ///
    /// When `true`, the provider persists the `LlamaContext` between calls and
    /// reuses the KV-cache for any common token prefix. Subsequent calls that
    /// share a prefix (e.g. same system prompt) skip re-decoding the cached
    /// tokens, reducing time-to-first-token significantly.
    ///
    /// Defaults to `false`. Enable for workloads with repeated system prompts
    /// or multi-turn conversations on a single provider instance.
    ///
    /// **Note:** When enabled, inference calls on the same provider are
    /// serialized through a mutex for the duration of token generation.
    /// Concurrent callers will block until the active call completes.
    /// This is expected — `LlamaContext` is not thread-safe.
    pub context_reuse: bool,
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            model_source: ModelSource::Gguf {
                model_path: String::default(),
            },
            chat_template: None,
            system_prompt: None,
            force_json_grammar: false,
            force_pure_content: false,
            tokenizer_add_bos: None,
            tokenizer_add_eos: None,
            reasoning_format: None,
            extra_body: None,
            model_dir: None,
            hf_filename: None,
            hf_revision: None,
            mmproj_path: None,
            media_marker: None,
            mmproj_use_gpu: None,
            max_tokens: Some(512),
            temperature: Some(0.7),
            top_p: None,
            top_k: None,
            min_p: None,
            typical_p: None,
            top_n_sigma: None,
            xtc_probability: None,
            xtc_threshold: None,
            dynatemp_range: None,
            dynatemp_exponent: None,
            repeat_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            repeat_last_n: None,
            seed: None,
            min_keep: None,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            dry_multiplier: None,
            dry_base: None,
            dry_allowed_length: None,
            dry_penalty_last_n: None,
            dry_sequence_breakers: None,
            logit_bias: None,
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
            enable_thinking: None,
            add_generation_prompt: true,
            continue_final_message: LlamaCppChatContinuation::None,
            tool_choice: LlamaCppToolChoice::Auto,
            parallel_tool_calls: None,
            scheduler: LlamaCppSchedulerConfig::default(),
            context_reuse: false,
        }
    }
}

/// Builder for LlamaCppConfig.
#[derive(Debug, Default)]
pub struct LlamaCppConfigBuilder {
    config: LlamaCppConfig,
}

impl LlamaCppConfigBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self::default()
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

    /// Force pure-content chat rendering and response parsing.
    pub fn force_pure_content(mut self, force: bool) -> Self {
        self.config.force_pure_content = force;
        self
    }

    /// Override whether chat tokenization should add BOS.
    pub fn tokenizer_add_bos(mut self, add: bool) -> Self {
        self.config.tokenizer_add_bos = Some(add);
        self
    }

    /// Override whether chat tokenization should add EOS.
    pub fn tokenizer_add_eos(mut self, add: bool) -> Self {
        self.config.tokenizer_add_eos = Some(add);
        self
    }

    /// Set reasoning extraction format.
    pub fn reasoning_format(mut self, format: LlamaCppReasoningFormat) -> Self {
        self.config.reasoning_format = Some(format);
        self
    }

    /// Set optional provider-specific request metadata.
    ///
    /// `chat_template_kwargs` are merged into the Jinja chat-template context.
    pub fn extra_body(mut self, extra_body: impl Serialize) -> Self {
        self.config.extra_body = serde_json::to_value(extra_body).ok();
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

    /// Set the multimodal projection (mmproj) file path.
    pub fn mmproj_path(mut self, path: impl Into<String>) -> Self {
        self.config.mmproj_path = Some(path.into());
        self
    }

    /// Set MTMD media marker.
    pub fn media_marker(mut self, marker: impl Into<String>) -> Self {
        self.config.media_marker = Some(marker.into());
        self
    }

    /// Enable or disable GPU offload for MTMD projection.
    pub fn mmproj_use_gpu(mut self, use_gpu: bool) -> Self {
        self.config.mmproj_use_gpu = Some(use_gpu);
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

    /// Set minimum-p sampling parameter.
    pub fn min_p(mut self, p: f32) -> Self {
        self.config.min_p = Some(p);
        self
    }

    /// Set locally typical sampling parameter.
    pub fn typical_p(mut self, p: f32) -> Self {
        self.config.typical_p = Some(p);
        self
    }

    /// Set top-n-sigma sampling parameter.
    pub fn top_n_sigma(mut self, n: f32) -> Self {
        self.config.top_n_sigma = Some(n);
        self
    }

    /// Set XTC sampling parameters.
    pub fn xtc(mut self, probability: f32, threshold: f32) -> Self {
        self.config.xtc_probability = Some(probability);
        self.config.xtc_threshold = Some(threshold);
        self
    }

    /// Set dynamic temperature parameters.
    pub fn dynatemp(mut self, range: f32, exponent: f32) -> Self {
        self.config.dynatemp_range = Some(range);
        self.config.dynatemp_exponent = Some(exponent);
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

    /// Set the minimum number of candidates kept by supported samplers.
    pub fn min_keep(mut self, min_keep: usize) -> Self {
        self.config.min_keep = Some(min_keep);
        self
    }

    /// Set mirostat sampling mode and parameters.
    pub fn mirostat(mut self, mode: u8, tau: f32, eta: f32) -> Self {
        self.config.mirostat = Some(mode);
        self.config.mirostat_tau = Some(tau);
        self.config.mirostat_eta = Some(eta);
        self
    }

    /// Set DRY repetition sampling parameters.
    pub fn dry(
        mut self,
        multiplier: f32,
        base: f32,
        allowed_length: i32,
        penalty_last_n: i32,
        sequence_breakers: Vec<String>,
    ) -> Self {
        self.config.dry_multiplier = Some(multiplier);
        self.config.dry_base = Some(base);
        self.config.dry_allowed_length = Some(allowed_length);
        self.config.dry_penalty_last_n = Some(penalty_last_n);
        self.config.dry_sequence_breakers = Some(sequence_breakers);
        self
    }

    /// Set per-token logit bias values.
    pub fn logit_bias(mut self, biases: Vec<(i32, f32)>) -> Self {
        self.config.logit_bias = Some(biases);
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

    /// Enable or disable thinking/reasoning tokens in chat template.
    ///
    pub fn enable_thinking(mut self, enable: bool) -> Self {
        self.config.enable_thinking = Some(enable);
        self
    }

    /// Set whether chat templates should add an assistant generation prompt.
    pub fn add_generation_prompt(mut self, enable: bool) -> Self {
        self.config.add_generation_prompt = enable;
        self
    }

    /// Set final-message continuation behavior.
    pub fn continue_final_message(mut self, continuation: LlamaCppChatContinuation) -> Self {
        self.config.continue_final_message = continuation;
        self
    }

    /// Set tool-call choice behavior.
    pub fn tool_choice(mut self, choice: LlamaCppToolChoice) -> Self {
        self.config.tool_choice = choice;
        self
    }

    /// Force a specific function tool by name.
    pub fn tool_choice_function(mut self, name: impl Into<String>) -> Self {
        self.config.tool_choice = LlamaCppToolChoice::Function { name: name.into() };
        self
    }

    /// Enable or disable parallel tool calls when supported by the template.
    pub fn parallel_tool_calls(mut self, enable: bool) -> Self {
        self.config.parallel_tool_calls = Some(enable);
        self
    }

    /// Set embedded scheduler configuration.
    pub fn scheduler(mut self, scheduler: LlamaCppSchedulerConfig) -> Self {
        self.config.scheduler = scheduler;
        self
    }

    /// Set the number of embedded inference slots.
    pub fn n_slots(mut self, n_slots: usize) -> Self {
        self.config.scheduler.n_slots = n_slots.max(1);
        self
    }

    /// Set embedded request queue capacity.
    pub fn queue_capacity(mut self, capacity: usize) -> Self {
        self.config.scheduler.queue_capacity = capacity;
        self
    }

    /// Enable KV-cache prefix reuse across inference calls.
    pub fn context_reuse(mut self, enable: bool) -> Self {
        self.config.context_reuse = enable;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> LlamaCppConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder_basic() {
        let config = LlamaCppConfigBuilder::default()
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
    fn test_config_builder_optional_flags() {
        let config = LlamaCppConfigBuilder::default()
            .model_path("model.gguf")
            .force_json_grammar(true)
            .force_pure_content(true)
            .tokenizer_add_bos(true)
            .tokenizer_add_eos(true)
            .reasoning_format(LlamaCppReasoningFormat::Deepseek)
            .extra_body(serde_json::json!({
                "chat_template_kwargs": {
                    "enable_thinking": true
                }
            }))
            .mmproj_use_gpu(true)
            .continue_final_message(LlamaCppChatContinuation::Content)
            .split_mode(LlamaCppSplitMode::Layer)
            .use_mlock(true)
            .devices(vec![0, 1])
            .build();

        assert!(config.force_json_grammar);
        assert!(config.force_pure_content);
        assert_eq!(config.tokenizer_add_bos, Some(true));
        assert_eq!(config.tokenizer_add_eos, Some(true));
        assert_eq!(
            config.continue_final_message,
            LlamaCppChatContinuation::Content
        );
        assert_eq!(
            config.reasoning_format,
            Some(LlamaCppReasoningFormat::Deepseek)
        );
        assert_eq!(
            config
                .extra_body
                .as_ref()
                .and_then(|v| v.get("chat_template_kwargs"))
                .and_then(|v| v.get("enable_thinking"))
                .and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(config.mmproj_use_gpu, Some(true));
        assert_eq!(config.split_mode, Some(LlamaCppSplitMode::Layer));
        assert_eq!(config.use_mlock, Some(true));
        assert_eq!(config.devices, Some(vec![0, 1]));
    }

    #[test]
    fn test_config_default_reasoning_format_is_opt_in() {
        let config = LlamaCppConfig::default();
        assert_eq!(config.reasoning_format, None);
        assert!(!config.force_pure_content);
        assert_eq!(config.tokenizer_add_bos, None);
        assert_eq!(config.tokenizer_add_eos, None);
        assert_eq!(
            config.continue_final_message,
            LlamaCppChatContinuation::None
        );
    }

    #[test]
    fn test_tool_choice_serde_matches_openai_server_shape() {
        assert_eq!(
            serde_json::to_value(&LlamaCppToolChoice::Auto).expect("serialize auto"),
            serde_json::json!("auto")
        );
        assert_eq!(
            serde_json::from_value::<LlamaCppToolChoice>(serde_json::json!("required"))
                .expect("deserialize required"),
            LlamaCppToolChoice::Required
        );

        let named = LlamaCppToolChoice::Function {
            name: "lookup".to_string(),
        };
        let value = serde_json::to_value(&named).expect("serialize named tool choice");
        assert_eq!(
            value,
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "lookup"
                }
            })
        );
        assert_eq!(
            serde_json::from_value::<LlamaCppToolChoice>(value).expect("deserialize named choice"),
            named
        );
    }

    #[test]
    fn test_config_builder_named_tool_choice() {
        let config = LlamaCppConfigBuilder::default()
            .tool_choice_function("lookup")
            .build();

        assert_eq!(
            config.tool_choice,
            LlamaCppToolChoice::Function {
                name: "lookup".to_string()
            }
        );
    }

    #[test]
    fn test_config_builder_selected_options() {
        let config = LlamaCppConfigBuilder::default()
            .model_source(ModelSource::huggingface_with_filename(
                "org/model",
                "model.gguf",
            ))
            .chat_template("chat-template")
            .system_prompt("system")
            .model_dir("cache")
            .hf_filename("override.gguf")
            .hf_revision("rev1")
            .mmproj_path("mmproj.gguf")
            .media_marker("[IMG]")
            .max_tokens(123)
            .temperature(0.5)
            .top_p(0.9)
            .top_k(42)
            .repeat_penalty(1.1)
            .frequency_penalty(0.2)
            .presence_penalty(0.3)
            .repeat_last_n(32)
            .seed(7)
            .n_ctx(2048)
            .n_batch(64)
            .n_ubatch(8)
            .n_threads(4)
            .n_threads_batch(2)
            .n_gpu_layers(3)
            .main_gpu(1)
            .build();

        assert!(matches!(
            config.model_source,
            ModelSource::HuggingFace { .. }
        ));
        assert_eq!(config.chat_template.as_deref(), Some("chat-template"));
        assert_eq!(config.system_prompt.as_deref(), Some("system"));
        assert_eq!(config.model_dir.as_deref(), Some("cache"));
        assert_eq!(config.hf_filename.as_deref(), Some("override.gguf"));
        assert_eq!(config.hf_revision.as_deref(), Some("rev1"));
        assert_eq!(config.mmproj_path.as_deref(), Some("mmproj.gguf"));
        assert_eq!(config.media_marker.as_deref(), Some("[IMG]"));
        assert_eq!(config.max_tokens, Some(123));
        assert_eq!(config.temperature, Some(0.5));
        assert_eq!(config.n_ctx, Some(2048));
        assert_eq!(config.n_threads, Some(4));
        assert_eq!(config.n_gpu_layers, Some(3));
        assert_eq!(config.main_gpu, Some(1));
    }
}
