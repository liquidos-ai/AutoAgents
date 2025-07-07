use async_trait::async_trait;

use crate::{chat::ChatResponse, error::LLMError, ToolCall};

/// A request for text completion from an LLM provider.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    /// The input prompt text to complete
    pub prompt: String,
    /// Optional maximum number of tokens to generate
    pub max_tokens: Option<u32>,
    /// Optional temperature parameter to control randomness (0.0-1.0)
    pub temperature: Option<f32>,
}

/// A response containing generated text from a completion request.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    /// The generated completion text
    pub text: String,
}

impl ChatResponse for CompletionResponse {
    fn text(&self) -> Option<String> {
        Some(self.text.clone())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        None
    }
}

impl CompletionRequest {
    /// Creates a new completion request with just a prompt.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The input text to complete
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            max_tokens: None,
            temperature: None,
        }
    }

    /// Creates a builder for constructing a completion request.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The input text to complete
    pub fn builder(prompt: impl Into<String>) -> CompletionRequestBuilder {
        CompletionRequestBuilder {
            prompt: prompt.into(),
            max_tokens: None,
            temperature: None,
        }
    }
}

/// Builder for constructing completion requests with optional parameters.
#[derive(Debug, Clone)]
pub struct CompletionRequestBuilder {
    /// The input prompt text to complete
    pub prompt: String,
    /// Optional maximum number of tokens to generate
    pub max_tokens: Option<u32>,
    /// Optional temperature parameter to control randomness (0.0-1.0)
    pub temperature: Option<f32>,
}

impl CompletionRequestBuilder {
    /// Sets the maximum number of tokens to generate.
    pub fn max_tokens(mut self, val: u32) -> Self {
        self.max_tokens = Some(val);
        self
    }

    /// Sets the temperature parameter for controlling randomness.
    pub fn temperature(mut self, val: f32) -> Self {
        self.temperature = Some(val);
        self
    }

    /// Builds the completion request with the configured parameters.
    pub fn build(self) -> CompletionRequest {
        CompletionRequest {
            prompt: self.prompt,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
        }
    }
}

/// Trait for providers that support text completion requests.
#[async_trait]
pub trait CompletionProvider {
    /// Sends a completion request to generate text.
    ///
    /// # Arguments
    ///
    /// * `req` - The completion request parameters
    ///
    /// # Returns
    ///
    /// The generated completion text or an error
    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, LLMError>;
}

impl std::fmt::Display for CompletionResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text)
    }
}
