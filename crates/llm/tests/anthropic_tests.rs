#![allow(unused_imports, dead_code)]
use autoagents_llm::{
    builder::LLMBuilder,
    chat::{
        ChatMessage, ChatProvider, ChatResponse, ChatRole, MessageType, StructuredOutputFormat,
        Tool, ToolChoice,
    },
    completion::{CompletionProvider, CompletionRequest},
    error::LLMError,
    models::ModelsProvider,
    FunctionCall, ToolCall,
};
use serde_json::json;
use std::sync::Arc;

#[cfg(feature = "anthropic")]
mod anthropic_test_cases {
    use super::*;
    use autoagents_llm::backends::anthropic::Anthropic;

    fn create_test_anthropic() -> Arc<Anthropic> {
        LLMBuilder::<Anthropic>::new()
            .api_key("test-key")
            .model("claude-3-sonnet-20240229")
            .max_tokens(100)
            .temperature(0.7)
            .system("Test system prompt")
            .build()
            .expect("Failed to build Anthropic client")
    }

    #[test]
    fn test_anthropic_creation() {
        let client = create_test_anthropic();
        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.model, "claude-3-sonnet-20240229");
        assert_eq!(client.max_tokens, 100);
        assert_eq!(client.temperature, 0.7);
        assert_eq!(client.system, "Test system prompt");
    }

    #[test]
    fn test_anthropic_with_reasoning() {
        let client = LLMBuilder::<Anthropic>::new()
            .api_key("test-key")
            .reasoning(true)
            .reasoning_budget_tokens(8000)
            .build()
            .expect("Failed to build Anthropic client with reasoning");

        assert!(client.reasoning);
        assert_eq!(client.thinking_budget_tokens, Some(8000));
    }

    #[test]
    fn test_anthropic_builder_validation() {
        // Test missing API key
        let result = LLMBuilder::<Anthropic>::new()
            .model("claude-3-sonnet-20240229")
            .build();

        assert!(result.is_err());
        match result.err().unwrap() {
            LLMError::InvalidRequest(msg) => {
                assert!(msg.contains("No API key provided"));
            }
            _ => panic!("Expected InvalidRequest error"),
        }
    }

    #[test]
    fn test_anthropic_default_values() {
        let client = Anthropic::new(
            "test-key", None, None, None, None, None, None, None, None, None, None,
        );

        assert_eq!(client.model, "claude-3-sonnet-20240229");
        assert_eq!(client.max_tokens, 300);
        assert_eq!(client.temperature, 0.7);
        assert_eq!(client.system, "You are a helpful assistant.");
        assert_eq!(client.timeout_seconds, 30);
        assert!(!client.reasoning);
    }

    #[tokio::test]
    async fn test_chat_auth_error() {
        let client = Anthropic::new(
            "", // Empty API key
            None, None, None, None, None, None, None, None, None, None,
        );

        let messages = vec![ChatMessage::user().content("Hello").build()];

        let result = client.chat(&messages, None, None).await;
        assert!(result.is_err());

        match result.err().unwrap() {
            LLMError::AuthError(msg) => {
                assert_eq!(msg, "Missing Anthropic API key");
            }
            _ => panic!("Expected AuthError"),
        }
    }

    #[test]
    fn test_top_p_top_k_configuration() {
        let client = LLMBuilder::<Anthropic>::new()
            .api_key("test-key")
            .top_p(0.9)
            .top_k(40)
            .build()
            .expect("Failed to build Anthropic client with sampling params");

        assert_eq!(client.top_p, Some(0.9));
        assert_eq!(client.top_k, Some(40));
    }
}
