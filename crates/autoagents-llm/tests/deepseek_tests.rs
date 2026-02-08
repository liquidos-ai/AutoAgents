#![allow(unused_imports)]
use autoagents_llm::{
    builder::LLMBuilder,
    chat::{ChatMessage, ChatProvider},
    completion::{CompletionProvider, CompletionRequest},
    embedding::EmbeddingProvider,
    error::LLMError,
};
use std::sync::Arc;

#[cfg(feature = "deepseek")]
mod deepseek_tests {
    use super::*;
    use autoagents_llm::backends::deepseek::DeepSeek;

    fn build_test_deepseek() -> Arc<DeepSeek> {
        LLMBuilder::<DeepSeek>::new()
            .api_key("test-key")
            .model("deepseek-chat")
            .max_tokens(100)
            .temperature(0.7)
            .build()
            .expect("Failed to build DeepSeek client")
    }

    fn new_deepseek(api_key: &str, model: Option<String>) -> DeepSeek {
        DeepSeek::new(api_key.to_string(), model, None, None, None)
    }

    #[test]
    fn test_deepseek_creation() {
        let client = build_test_deepseek();
        assert_eq!(client.api_key(), "test-key");
        assert_eq!(client.model(), "deepseek-chat");
    }

    #[test]
    fn test_deepseek_builder_validation() {
        // Should fail without API key
        let result = LLMBuilder::<DeepSeek>::new().model("deepseek-chat").build();
        assert!(result.is_err());
        match result {
            Err(LLMError::InvalidRequest(msg)) => {
                assert!(msg.contains("No API key provided"));
            }
            _ => panic!("Expected InvalidRequest error"),
        }
    }

    #[test]
    fn test_deepseek_default_values() {
        let client = new_deepseek("test-key", None);
        assert_eq!(client.model(), "deepseek-chat");
        assert_eq!(client.api_key(), "test-key");
    }

    #[test]
    fn test_deepseek_custom_model() {
        let client = new_deepseek("test-key", Some("deepseek-coder".to_string()));
        assert_eq!(client.model(), "deepseek-coder");
    }

    #[test]
    fn test_deepseek_with_options() {
        let client = DeepSeek::new_with_options(
            "test-key",
            Some("https://custom.api.com/v1/".to_string()),
            Some("deepseek-chat".to_string()),
            Some(1024),
            Some(0.5),
            Some(30),
            Some(0.9),
            None,
        );
        assert_eq!(client.api_key(), "test-key");
        assert_eq!(client.model(), "deepseek-chat");
    }

    #[test]
    fn test_deepseek_builder_with_all_options() {
        let client = LLMBuilder::<DeepSeek>::new()
            .api_key("test-key")
            .model("deepseek-chat")
            .max_tokens(2048)
            .temperature(0.8)
            .timeout_seconds(60)
            .build()
            .expect("Failed to build DeepSeek client");

        assert_eq!(client.api_key(), "test-key");
        assert_eq!(client.model(), "deepseek-chat");
    }

    #[tokio::test]
    async fn test_deepseek_completion_not_implemented() {
        let client = new_deepseek("test-key", None);
        let request = CompletionRequest {
            prompt: "Hello".to_string(),
            max_tokens: None,
            temperature: None,
        };
        let result = client.complete(&request, None).await;
        assert!(result.is_err());
        match result {
            Err(LLMError::ProviderError(msg)) => {
                assert!(msg.contains("not implemented"));
            }
            _ => panic!("Expected ProviderError"),
        }
    }

    #[tokio::test]
    async fn test_deepseek_embedding_not_supported() {
        let client = new_deepseek("test-key", None);
        let result = client.embed(vec!["test".to_string()]).await;
        assert!(result.is_err());
        match result {
            Err(LLMError::ProviderError(msg)) => {
                assert!(msg.contains("not supported"));
            }
            _ => panic!("Expected ProviderError"),
        }
    }

    #[test]
    fn test_deepseek_empty_api_key_in_builder() {
        // Builder should still succeed with empty key (validation happens at API call time)
        let client = LLMBuilder::<DeepSeek>::new()
            .api_key("")
            .model("deepseek-chat")
            .build()
            .expect("Should build with empty key");
        assert_eq!(client.api_key(), "");
    }
}
