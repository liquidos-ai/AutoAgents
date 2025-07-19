#![allow(unused_imports)]
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

#[cfg(feature = "google")]
mod google_tests {
    use super::*;
    use autoagents_llm::backends::google::Google;

    fn create_test_google() -> Arc<Google> {
        LLMBuilder::<Google>::new()
            .api_key("test-key")
            .model("gemini-1.5-flash")
            .max_tokens(100)
            .temperature(0.7)
            .system("Test system prompt")
            .build()
            .expect("Failed to build Google client")
    }

    #[test]
    fn test_google_creation() {
        let client = create_test_google();
        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.model, "gemini-1.5-flash");
        assert_eq!(client.max_tokens, Some(100));
        assert_eq!(client.temperature, Some(0.7));
        assert_eq!(client.system, Some("Test system prompt".to_string()));
    }

    #[test]
    fn test_google_builder_validation() {
        // Test missing API key
        let result = LLMBuilder::<Google>::new()
            .model("gemini-1.5-flash")
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
    fn test_google_default_values() {
        let client = Google::new(
            "test-key", None, None, None, None, None, None, None, None, None, None,
        );

        assert_eq!(client.model, "gemini-1.5-flash");
        assert!(client.max_tokens.is_none());
        assert!(client.temperature.is_none());
        assert!(client.system.is_none());
        assert!(client.stream.is_none());
    }

    #[tokio::test]
    async fn test_chat_auth_error() {
        let client = Google::new(
            "", // Empty API key
            None, None, None, None, None, None, None, None, None, None,
        );

        let messages = vec![ChatMessage::user().content("Hello").build()];

        let result = client.chat(&messages, None).await;
        assert!(result.is_err());

        match result.err().unwrap() {
            LLMError::AuthError(msg) => {
                assert_eq!(msg, "Missing Google API key");
            }
            _ => panic!("Expected AuthError"),
        }
    }
}
