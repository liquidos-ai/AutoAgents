#![allow(unused_imports)]
use autoagents_llm::{
    builder::LLMBuilder,
    chat::{
        ChatMessage, ChatProvider, ChatResponse, ChatRole, MessageType, StructuredOutputFormat,
        Tool, ToolChoice,
    },
    completion::{CompletionProvider, CompletionRequest},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
    FunctionCall, ToolCall,
};
use serde_json::json;
use std::sync::Arc;

#[cfg(feature = "ollama")]
mod ollama_test_cases {
    use super::*;
    use autoagents_llm::backends::ollama::Ollama;

    fn create_test_ollama() -> Arc<Ollama> {
        LLMBuilder::<Ollama>::new()
            .base_url("http://localhost:11434")
            .model("llama3.1")
            .max_tokens(100)
            .temperature(0.7)
            .build()
            .expect("Failed to build Ollama client")
    }

    #[test]
    fn test_ollama_creation() {
        let client = create_test_ollama();
        assert_eq!(client.base_url, "http://localhost:11434");
        assert_eq!(client.model, "llama3.1");
        assert_eq!(client.max_tokens, Some(100));
        assert_eq!(client.temperature, Some(0.7));
    }

    #[test]
    fn test_ollama_with_api_key() {
        let client = LLMBuilder::<Ollama>::new()
            .base_url("http://localhost:11434")
            .api_key("test-api-key")
            .model("llama3.1")
            .build()
            .expect("Failed to build Ollama client with API key");

        assert_eq!(client.api_key, Some("test-api-key".to_string()));
    }

    #[test]
    fn test_ollama_default_values() {
        let client = Ollama::new(
            "http://localhost:11434",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );

        assert_eq!(client.base_url, "http://localhost:11434");
        assert_eq!(client.model, "llama3.1");
        assert!(client.api_key.is_none());
        assert!(client.max_tokens.is_none());
        assert!(client.temperature.is_none());
    }

    #[tokio::test]
    async fn test_chat_missing_base_url() {
        let client = Ollama::new(
            "", // Empty base URL
            None, None, None, None, None, None, None, None,
        );

        let messages = vec![ChatMessage::user().content("Hello").build()];

        let result = client.chat(&messages, None).await;
        assert!(result.is_err());

        match result.err().unwrap() {
            LLMError::InvalidRequest(msg) => {
                assert_eq!(msg, "Missing base_url");
            }
            _ => panic!("Expected InvalidRequest error"),
        }
    }

    #[tokio::test]
    async fn test_completion_missing_base_url() {
        let client = Ollama::new(
            "", // Empty base URL
            None, None, None, None, None, None, None, None,
        );

        let req = CompletionRequest {
            prompt: "Test prompt".to_string(),
            max_tokens: None,
            temperature: None,
        };

        let result = client.complete(&req, None).await;
        assert!(result.is_err());

        match result.err().unwrap() {
            LLMError::InvalidRequest(msg) => {
                assert_eq!(msg, "Missing base_url");
            }
            _ => panic!("Expected InvalidRequest error"),
        }
    }

    #[tokio::test]
    async fn test_embedding() {
        let client = Ollama::new(
            "", // Empty base URL will cause error
            None, None, None, None, None, None, None, None,
        );

        let result = client.embed(vec!["test text".to_string()]).await;
        assert!(result.is_err());

        match result.err().unwrap() {
            LLMError::InvalidRequest(msg) => {
                assert_eq!(msg, "Missing base_url");
            }
            _ => panic!("Expected InvalidRequest error"),
        }
    }
}
