//! Core traits for liquid-edge inference runtime
//!
//! This module defines the fundamental traits that enable different
//! inference backends, tokenizers, and runtime configurations.

use crate::error::EdgeResult;
use std::collections::HashMap;
use std::path::Path;

/// Core inference runtime trait for executing model inference
pub trait InferenceRuntime: Send + Sync {
    /// The tensor type used by this runtime
    type Tensor;

    /// Input data type for inference
    type Input;

    /// Output data type from inference
    type Output;

    /// Run inference with the given inputs
    fn infer(&self, inputs: Self::Input) -> EdgeResult<Self::Output>;

    /// Get model information
    fn model_info(&self) -> ModelInfo;

    /// Check if the runtime is ready for inference
    fn is_ready(&self) -> bool;

    /// Get runtime statistics
    fn stats(&self) -> RuntimeStats;
}

/// Tokenizer trait for text processing
pub trait TokenizerTrait: Send + Sync {
    /// Encode text to token IDs
    fn encode(&self, text: &str, add_special_tokens: bool) -> EdgeResult<Vec<u32>>;

    /// Decode token IDs to text
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> EdgeResult<String>;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Get special token IDs
    fn special_tokens(&self) -> SpecialTokenIds;

    /// Tokenize text and return detailed encoding information
    fn tokenize_detailed(&self, text: &str) -> EdgeResult<TokenizedOutput>;
}

/// Model loading trait for different model formats
pub trait ModelLoader: Send + Sync {
    /// The runtime type this loader creates
    type Runtime: InferenceRuntime;

    /// Load a model from the given path
    fn load_model<P: AsRef<Path>>(&self, model_path: P) -> EdgeResult<Self::Runtime>;

    /// Check if a model file is supported by this loader
    fn supports_format<P: AsRef<Path>>(&self, model_path: P) -> bool;

    /// Get supported file extensions
    fn supported_extensions(&self) -> Vec<&'static str>;
}

/// Configuration loading trait
pub trait ConfigLoader: Send + Sync {
    /// The configuration type this loader produces
    type Config;

    /// Load configuration from a directory
    fn load_config<P: AsRef<Path>>(&self, config_dir: P) -> EdgeResult<Self::Config>;

    /// Validate configuration
    fn validate_config(&self, config: &Self::Config) -> EdgeResult<()>;
}

/// Chat runtime trait for conversational AI
#[cfg(feature = "chat")]
pub trait ChatRuntime: Send + Sync {
    /// Message type for conversations
    type Message;

    /// Chat response type
    type Response;

    /// Generate a response to a message
    fn generate_response(
        &mut self,
        message: Self::Message,
        options: Option<GenerationOptions>,
    ) -> EdgeResult<Self::Response>;

    /// Get conversation history
    fn get_history(&self) -> Vec<(Self::Message, Self::Response)>;

    /// Clear conversation history
    fn clear_history(&mut self);

    /// Apply chat template to messages
    fn apply_template(&self, messages: &[ChatMessage]) -> EdgeResult<String>;
}

/// Template rendering trait for chat templates
#[cfg(feature = "jinja-templates")]
pub trait TemplateRenderer: Send + Sync {
    /// Render a template with the given context
    fn render(
        &self,
        template: &str,
        context: &HashMap<String, serde_json::Value>,
    ) -> EdgeResult<String>;

    /// Load template from file
    fn load_template<P: AsRef<Path>>(&mut self, name: &str, path: P) -> EdgeResult<()>;
}

/// Sampling strategy trait for text generation
pub trait SamplingStrategy: Send + Sync {
    /// Sample the next token from logits
    fn sample(&self, logits: &[f32], options: &GenerationOptions) -> EdgeResult<u32>;

    /// Get strategy name
    fn name(&self) -> &'static str;
}

/// Model information structure
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub version: Option<String>,
    pub architecture: String,
    pub parameter_count: Option<u64>,
    pub context_length: usize,
    pub vocab_size: usize,
}

/// Runtime statistics
#[derive(Debug, Clone, Default)]
pub struct RuntimeStats {
    pub total_inferences: u64,
    pub total_tokens_generated: u64,
    pub average_latency_ms: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub memory_usage_bytes: usize,
}

/// Special token IDs
#[derive(Debug, Clone, Default)]
pub struct SpecialTokenIds {
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub unk_token_id: Option<u32>,
    pub mask_token_id: Option<u32>,
}

/// Detailed tokenization output
#[derive(Debug, Clone)]
pub struct TokenizedOutput {
    pub token_ids: Vec<u32>,
    pub tokens: Vec<String>,
    pub attention_mask: Vec<u32>,
    pub special_tokens_mask: Vec<bool>,
    pub offsets: Vec<(usize, usize)>,
}

/// Generation options for text generation
#[derive(Debug, Clone)]
pub struct GenerationOptions {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub do_sample: bool,
    pub seed: Option<u64>,
    pub stop_sequences: Vec<String>,
}

/// Chat message structure
#[cfg(feature = "chat")]
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Chat response structure
#[cfg(feature = "chat")]
#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub content: String,
    pub finish_reason: FinishReason,
    pub token_count: usize,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Reason why generation finished
#[cfg(feature = "chat")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// Hit maximum token limit
    MaxTokens,
    /// Generated end-of-sequence token
    EndOfSequence,
    /// Hit a stop sequence
    StopSequence(String),
    /// User requested stop
    UserStop,
    /// Error occurred
    Error,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 0,
            repetition_penalty: 1.0,
            do_sample: true,
            seed: None,
            stop_sequences: Vec::new(),
        }
    }
}

impl GenerationOptions {
    /// Create deterministic generation options
    pub fn deterministic() -> Self {
        Self {
            temperature: 0.0,
            do_sample: false,
            ..Default::default()
        }
    }

    /// Create creative generation options
    pub fn creative() -> Self {
        Self {
            temperature: 0.9,
            top_p: 0.95,
            do_sample: true,
            ..Default::default()
        }
    }

    /// Create balanced generation options
    pub fn balanced() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            do_sample: true,
            ..Default::default()
        }
    }
}

#[cfg(feature = "chat")]
impl ChatMessage {
    /// Create a new user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
            metadata: None,
        }
    }

    /// Create a new assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
            metadata: None,
        }
    }

    /// Create a new system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
            metadata: None,
        }
    }

    /// Add metadata to the message
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        if self.metadata.is_none() {
            self.metadata = Some(HashMap::new());
        }
        self.metadata.as_mut().unwrap().insert(key.into(), value);
        self
    }
}

#[cfg(feature = "chat")]
impl ChatResponse {
    /// Create a new chat response
    pub fn new(
        content: impl Into<String>,
        finish_reason: FinishReason,
        token_count: usize,
    ) -> Self {
        Self {
            content: content.into(),
            finish_reason,
            token_count,
            metadata: None,
        }
    }

    /// Add metadata to the response
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        if self.metadata.is_none() {
            self.metadata = Some(HashMap::new());
        }
        self.metadata.as_mut().unwrap().insert(key.into(), value);
        self
    }
}

/// Extension trait for adding functionality to existing types
pub trait InferenceRuntimeExt: InferenceRuntime {
    /// Run inference with automatic batching
    fn infer_batch(&self, inputs: Vec<Self::Input>) -> EdgeResult<Vec<Self::Output>> {
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            results.push(self.infer(input)?);
        }
        Ok(results)
    }

    /// Warm up the runtime with a dummy input
    fn warmup(&self, dummy_input: Self::Input) -> EdgeResult<()> {
        self.infer(dummy_input)?;
        Ok(())
    }
}

// Blanket implementation for all InferenceRuntime types
impl<T: InferenceRuntime> InferenceRuntimeExt for T {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_options_defaults() {
        let opts = GenerationOptions::default();
        assert_eq!(opts.max_new_tokens, 100);
        assert_eq!(opts.temperature, 0.7);
        assert!(opts.do_sample);
    }

    #[test]
    fn test_deterministic_options() {
        let opts = GenerationOptions::deterministic();
        assert_eq!(opts.temperature, 0.0);
        assert!(!opts.do_sample);
    }

    #[cfg(feature = "chat")]
    #[test]
    fn test_chat_message_creation() {
        let msg = ChatMessage::user("Hello, world!");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello, world!");
        assert!(msg.metadata.is_none());
    }

    #[cfg(feature = "chat")]
    #[test]
    fn test_chat_message_with_metadata() {
        let msg = ChatMessage::assistant("Hi there!")
            .with_metadata("confidence", serde_json::json!(0.95));
        assert_eq!(msg.role, "assistant");
        assert!(msg.metadata.is_some());
    }
}
