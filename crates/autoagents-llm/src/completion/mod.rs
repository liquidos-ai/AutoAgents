use async_trait::async_trait;

use crate::{
    ToolCall,
    chat::{ChatResponse, StructuredOutputFormat},
    error::LLMError,
};

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
    async fn complete(
        &self,
        req: &CompletionRequest,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError>;
}

impl std::fmt::Display for CompletionResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::LLMError;

    #[test]
    fn test_completion_request_new() {
        let request = CompletionRequest::new("Hello, world!");
        assert_eq!(request.prompt, "Hello, world!");
        assert!(request.max_tokens.is_none());
        assert!(request.temperature.is_none());
    }

    #[test]
    fn test_completion_request_builder() {
        let request = CompletionRequest::builder("Test prompt")
            .max_tokens(500)
            .temperature(0.8)
            .build();

        assert_eq!(request.prompt, "Test prompt");
        assert_eq!(request.max_tokens, Some(500));
        assert_eq!(request.temperature, Some(0.8));
    }

    #[test]
    fn test_completion_request_builder_partial() {
        let request = CompletionRequest::builder("Partial test")
            .max_tokens(100)
            .build();

        assert_eq!(request.prompt, "Partial test");
        assert_eq!(request.max_tokens, Some(100));
        assert!(request.temperature.is_none());
    }

    #[test]
    fn test_completion_request_builder_chaining() {
        let builder = CompletionRequest::builder("Chain test")
            .max_tokens(200)
            .temperature(0.5);

        let request = builder.build();
        assert_eq!(request.prompt, "Chain test");
        assert_eq!(request.max_tokens, Some(200));
        assert_eq!(request.temperature, Some(0.5));
    }

    #[test]
    fn test_completion_request_clone() {
        let request = CompletionRequest::new("Cloneable prompt");
        let cloned = request.clone();

        assert_eq!(request.prompt, cloned.prompt);
        assert_eq!(request.max_tokens, cloned.max_tokens);
        assert_eq!(request.temperature, cloned.temperature);
    }

    #[test]
    fn test_completion_request_debug() {
        let request = CompletionRequest::new("Debug test");
        let debug_str = format!("{request:?}");
        assert!(debug_str.contains("CompletionRequest"));
        assert!(debug_str.contains("Debug test"));
    }

    #[test]
    fn test_completion_response_new() {
        let response = CompletionResponse {
            text: "Generated text".to_string(),
        };
        assert_eq!(response.text, "Generated text");
    }

    #[test]
    fn test_completion_response_clone() {
        let response = CompletionResponse {
            text: "Cloneable response".to_string(),
        };
        let cloned = response.clone();
        assert_eq!(response.text, cloned.text);
    }

    #[test]
    fn test_completion_response_debug() {
        let response = CompletionResponse {
            text: "Debug response".to_string(),
        };
        let debug_str = format!("{response:?}");
        assert!(debug_str.contains("CompletionResponse"));
        assert!(debug_str.contains("Debug response"));
    }

    #[test]
    fn test_completion_response_display() {
        let response = CompletionResponse {
            text: "Display test".to_string(),
        };
        assert_eq!(response.to_string(), "Display test");
    }

    #[test]
    fn test_completion_response_chat_response_trait() {
        let response = CompletionResponse {
            text: "Chat response test".to_string(),
        };

        // Test ChatResponse trait implementation
        assert_eq!(response.text(), Some("Chat response test".to_string()));
        assert!(response.tool_calls().is_none());
    }

    #[test]
    fn test_completion_request_builder_debug() {
        let builder = CompletionRequest::builder("Builder debug")
            .max_tokens(300)
            .temperature(0.9);

        let debug_str = format!("{builder:?}");
        assert!(debug_str.contains("CompletionRequestBuilder"));
        assert!(debug_str.contains("Builder debug"));
    }

    #[test]
    fn test_completion_request_builder_clone() {
        let builder = CompletionRequest::builder("Clone test")
            .max_tokens(400)
            .temperature(0.3);

        let cloned = builder.clone();
        let request1 = builder.build();
        let request2 = cloned.build();

        assert_eq!(request1.prompt, request2.prompt);
        assert_eq!(request1.max_tokens, request2.max_tokens);
        assert_eq!(request1.temperature, request2.temperature);
    }

    #[test]
    fn test_completion_request_with_string_types() {
        let request = CompletionRequest::new(String::from("String prompt"));
        assert_eq!(request.prompt, "String prompt");

        let request2 = CompletionRequest::builder(String::from("Builder string")).build();
        assert_eq!(request2.prompt, "Builder string");
    }

    #[test]
    fn test_completion_request_zero_max_tokens() {
        let request = CompletionRequest::builder("Zero tokens")
            .max_tokens(0)
            .build();
        assert_eq!(request.max_tokens, Some(0));
    }

    #[test]
    fn test_completion_request_extreme_temperature() {
        let request = CompletionRequest::builder("Extreme temp")
            .temperature(0.0)
            .build();
        assert_eq!(request.temperature, Some(0.0));

        let request2 = CompletionRequest::builder("Extreme temp 2")
            .temperature(1.0)
            .build();
        assert_eq!(request2.temperature, Some(1.0));
    }

    #[test]
    fn test_completion_response_empty_text() {
        let response = CompletionResponse {
            text: String::default(),
        };
        assert_eq!(response.text(), Some(String::default()));
        assert_eq!(response.to_string(), "");
    }

    #[test]
    fn test_completion_response_multiline_text() {
        let multiline_text = "Line 1\nLine 2\nLine 3";
        let response = CompletionResponse {
            text: multiline_text.to_string(),
        };
        assert_eq!(response.text(), Some(multiline_text.to_string()));
        assert_eq!(response.to_string(), multiline_text);
    }

    #[test]
    fn test_completion_response_unicode_text() {
        let unicode_text = "Hello 世界! 🌍";
        let response = CompletionResponse {
            text: unicode_text.to_string(),
        };
        assert_eq!(response.text(), Some(unicode_text.to_string()));
        assert_eq!(response.to_string(), unicode_text);
    }

    // Mock provider for testing the trait
    struct MockCompletionProvider {
        should_fail: bool,
    }

    impl MockCompletionProvider {
        fn new() -> Self {
            Self { should_fail: false }
        }

        fn new_failing() -> Self {
            Self { should_fail: true }
        }
    }

    #[async_trait::async_trait]
    impl CompletionProvider for MockCompletionProvider {
        async fn complete(
            &self,
            req: &CompletionRequest,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<CompletionResponse, LLMError> {
            if self.should_fail {
                Err(LLMError::ProviderError("Mock provider error".to_string()))
            } else {
                Ok(CompletionResponse {
                    text: format!("Completed: {}", req.prompt),
                })
            }
        }
    }

    #[tokio::test]
    async fn test_completion_provider_trait_success() {
        let provider = MockCompletionProvider::new();
        let request = CompletionRequest::new("Test prompt");

        let result = provider.complete(&request, None).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.text, "Completed: Test prompt");
    }

    #[tokio::test]
    async fn test_completion_provider_trait_failure() {
        let provider = MockCompletionProvider::new_failing();
        let request = CompletionRequest::new("Test prompt");

        let result = provider.complete(&request, None).await;
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(error.to_string().contains("Mock provider error"));
    }

    #[tokio::test]
    async fn test_completion_provider_with_parameters() {
        let provider = MockCompletionProvider::new();
        let request = CompletionRequest::builder("Parameterized prompt")
            .max_tokens(100)
            .temperature(0.7)
            .build();

        let result = provider.complete(&request, None).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.text, "Completed: Parameterized prompt");
    }
}
