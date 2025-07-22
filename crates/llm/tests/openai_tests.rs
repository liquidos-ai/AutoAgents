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

#[cfg(feature = "openai")]
mod openai_test_cases {
    use super::*;
    use autoagents_llm::{backends::openai::OpenAI, chat::ReasoningEffort};

    fn create_test_openai() -> Arc<OpenAI> {
        LLMBuilder::<OpenAI>::new()
            .api_key("test-key")
            .model("gpt-4")
            .max_tokens(100)
            .temperature(0.7)
            .system("Test system prompt")
            .build()
            .expect("Failed to build OpenAI client")
    }

    #[test]
    fn test_openai_creation() {
        let client = create_test_openai();
        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.model, "gpt-4");
        assert_eq!(client.max_tokens, Some(100));
        assert_eq!(client.temperature, Some(0.7));
        assert_eq!(client.system, Some("Test system prompt".to_string()));
    }

    #[test]
    fn test_openai_with_custom_base_url() {
        let client = LLMBuilder::<OpenAI>::new()
            .api_key("test-key")
            .base_url("https://custom.openai.com/v1/")
            .build()
            .expect("Failed to build OpenAI client with custom URL");

        assert_eq!(client.base_url.as_str(), "https://custom.openai.com/v1/");
    }

    #[test]
    fn test_openai_builder_validation() {
        // Test missing API key
        let result = LLMBuilder::<OpenAI>::new().model("gpt-4").build();

        assert!(result.is_err());
        match result.err().unwrap() {
            LLMError::InvalidRequest(msg) => {
                assert!(msg.contains("No API key provided"));
            }
            _ => panic!("Expected InvalidRequest error"),
        }
    }

    #[test]
    fn test_openai_default_values() {
        let client = OpenAI::new(
            "test-key", None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
        );

        assert_eq!(client.model, "gpt-3.5-turbo");
        assert_eq!(client.base_url.as_str(), "https://api.openai.com/v1/");
    }

    #[test]
    fn test_embedding_configuration() {
        let client = LLMBuilder::<OpenAI>::new()
            .api_key("test-key")
            .model("text-embedding-ada-002")
            .embedding_encoding_format("float")
            .embedding_dimensions(1536)
            .build()
            .expect("Failed to build OpenAI embedding client");

        assert_eq!(client.embedding_encoding_format, Some("float".to_string()));
        assert_eq!(client.embedding_dimensions, Some(1536));
    }

    #[test]
    fn test_web_search_configuration() {
        let mut client = OpenAI::new(
            "test-key", None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
        );

        client = client.set_enable_web_search(true);
        client = client.set_web_search_context_size("large");
        client = client.set_web_search_user_location_type("approximate");
        client = client.set_web_search_user_location_approximate_country("US");
        client = client.set_web_search_user_location_approximate_city("San Francisco");
        client = client.set_web_search_user_location_approximate_region("CA");

        assert_eq!(client.enable_web_search, Some(true));
        assert_eq!(client.web_search_context_size, Some("large".to_string()));
        assert_eq!(
            client.web_search_user_location_type,
            Some("approximate".to_string())
        );
        assert_eq!(
            client.web_search_user_location_approximate_country,
            Some("US".to_string())
        );
        assert_eq!(
            client.web_search_user_location_approximate_city,
            Some("San Francisco".to_string())
        );
        assert_eq!(
            client.web_search_user_location_approximate_region,
            Some("CA".to_string())
        );
    }

    #[tokio::test]
    async fn test_chat_auth_error() {
        let client = OpenAI::new(
            "", // Empty API key
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None,
        );

        let messages = vec![ChatMessage::user().content("Hello").build()];

        let result = client.chat(&messages, None).await;
        assert!(result.is_err());

        match result.err().unwrap() {
            LLMError::AuthError(msg) => {
                assert_eq!(msg, "Missing OpenAI API key");
            }
            _ => panic!("Expected AuthError"),
        }
    }
}
