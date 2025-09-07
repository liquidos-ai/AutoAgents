#![allow(unused_imports)]
use autoagents_llm::{
    builder::LLMBuilder,
    chat::{
        ChatMessage, ChatProvider, ChatResponse, ChatRole, FunctionTool, MessageType,
        StructuredOutputFormat, Tool, ToolChoice,
    },
    completion::{CompletionProvider, CompletionRequest},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
    FunctionCall, ToolCall,
};
use serde_json::json;
use std::sync::Arc;

#[cfg(feature = "google")]
mod google_test_cases {
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
        let client = Google::new("test-key", None, None, None, None, None, None, None);

        assert_eq!(client.model, "gemini-1.5-flash");
        assert!(client.max_tokens.is_none());
        assert!(client.temperature.is_none());
        assert!(client.system.is_none());
    }

    #[tokio::test]
    async fn test_chat_auth_error() {
        let client = Google::new(
            "", // Empty API key
            None, None, None, None, None, None, None,
        );

        let messages = vec![ChatMessage::user().content("Hello").build()];

        let result = client.chat(&messages, None, None).await;
        assert!(result.is_err());

        match result.err().unwrap() {
            LLMError::AuthError(msg) => {
                assert_eq!(msg, "Missing Google API key");
            }
            _ => panic!("Expected AuthError"),
        }
    }

    #[test]
    fn test_google_builder_with_all_options() {
        let client = LLMBuilder::<Google>::new()
            .api_key("test-key")
            .model("gemini-1.5-pro")
            .max_tokens(1000)
            .temperature(0.9)
            .top_p(0.95)
            .top_k(50)
            .timeout_seconds(120)
            .system("Custom system prompt")
            .build()
            .expect("Failed to build Google client");

        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.model, "gemini-1.5-pro");
        assert_eq!(client.max_tokens, Some(1000));
        assert_eq!(client.temperature, Some(0.9));
        assert_eq!(client.top_p, Some(0.95));
        assert_eq!(client.top_k, Some(50));
        assert_eq!(client.timeout_seconds, Some(120));
        assert_eq!(client.system, Some("Custom system prompt".to_string()));
    }

    #[test]
    fn test_google_builder_minimal() {
        let client = LLMBuilder::<Google>::new()
            .api_key("test-key")
            .build()
            .expect("Failed to build Google client");

        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.model, "gemini-1.5-flash");
        assert!(client.max_tokens.is_none());
        assert!(client.temperature.is_none());
        assert!(client.system.is_none());
    }

    #[test]
    fn test_google_new_with_custom_model() {
        let client = Google::new(
            "test-key",
            Some("gemini-1.5-pro".to_string()),
            Some(500),
            Some(0.5),
            Some(60),
            Some("Test system".to_string()),
            Some(0.8),
            Some(40),
        );

        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.model, "gemini-1.5-pro");
        assert_eq!(client.max_tokens, Some(500));
        assert_eq!(client.temperature, Some(0.5));
        assert_eq!(client.timeout_seconds, Some(60));
        assert_eq!(client.system, Some("Test system".to_string()));
        assert_eq!(client.top_p, Some(0.8));
        assert_eq!(client.top_k, Some(40));
    }

    #[test]
    fn test_google_builder_missing_model() {
        // Without model, should use default
        let client = LLMBuilder::<Google>::new()
            .api_key("test-key")
            .build()
            .expect("Failed to build Google client");

        assert_eq!(client.model, "gemini-1.5-flash");
    }

    #[tokio::test]
    async fn test_completion_auth_error() {
        let client = Google::new(
            "", // Empty API key
            None, None, None, None, None, None, None,
        );

        let request = CompletionRequest {
            prompt: "Complete this sentence".to_string(),
            max_tokens: None,
            temperature: None,
        };

        let result = client.complete(&request, None).await;
        assert!(result.is_err());

        match result.err().unwrap() {
            LLMError::AuthError(msg) => {
                assert_eq!(msg, "Missing Google API key");
            }
            _ => panic!("Expected AuthError"),
        }
    }

    #[tokio::test]
    async fn test_embedding_auth_error() {
        let client = Google::new(
            "", // Empty API key
            None, None, None, None, None, None, None,
        );

        let texts = vec!["Test text".to_string()];

        let result = client.embed(texts).await;
        assert!(result.is_err());

        match result.err().unwrap() {
            LLMError::AuthError(msg) => {
                assert_eq!(msg, "Missing Google API key");
            }
            _ => panic!("Expected AuthError"),
        }
    }

    #[tokio::test]
    async fn test_models_provider_not_implemented() {
        let client = create_test_google();
        let result = client.list_models(None).await;

        assert!(result.is_err());
        match result.err().unwrap() {
            LLMError::ProviderError(msg) => {
                assert!(msg.contains("List Models not supported"));
            }
            _ => panic!("Expected ProviderError"),
        }
    }

    #[test]
    fn test_google_builder_with_zero_values() {
        let client = Google::new(
            "test-key",
            Some("test-model".to_string()),
            Some(0),
            Some(0.0),
            Some(0),
            None,
            Some(0.0),
            Some(0),
        );

        assert_eq!(client.max_tokens, Some(0));
        assert_eq!(client.temperature, Some(0.0));
        assert_eq!(client.timeout_seconds, Some(0));
        assert_eq!(client.top_p, Some(0.0));
        assert_eq!(client.top_k, Some(0));
    }

    #[test]
    fn test_google_builder_with_extreme_values() {
        let client = Google::new(
            "test-key",
            Some("test-model".to_string()),
            Some(u32::MAX),
            Some(1.0),
            Some(u64::MAX),
            Some("Very long system prompt that contains many words and characters to test the system's ability to handle longer prompts".to_string()),
            Some(1.0),
            Some(u32::MAX),
        );

        assert_eq!(client.max_tokens, Some(u32::MAX));
        assert_eq!(client.temperature, Some(1.0));
        assert_eq!(client.timeout_seconds, Some(u64::MAX));
        assert_eq!(client.top_p, Some(1.0));
        assert_eq!(client.top_k, Some(u32::MAX));
    }

    #[tokio::test]
    async fn test_chat_with_multiple_message_types() {
        let client = Google::new(
            "", // Will trigger auth error
            None, None, None, None, None, None, None,
        );

        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                message_type: MessageType::Text,
                content: "You are a helpful assistant".to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: "Hello".to_string(),
            },
            ChatMessage {
                role: ChatRole::Assistant,
                message_type: MessageType::Text,
                content: "Hi there!".to_string(),
            },
        ];

        let result = client.chat(&messages, None, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_chat_with_tools() {
        let client = Google::new(
            "", // Will trigger auth error
            None, None, None, None, None, None, None,
        );

        let messages = vec![ChatMessage::user().content("Use a tool").build()];

        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: FunctionTool {
                name: "get_weather".to_string(),
                description: "Get weather information".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get weather for"
                        }
                    },
                    "required": ["location"]
                }),
            },
        }];

        let result = client.chat(&messages, Some(&tools), None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_chat_with_structured_output() {
        let client = Google::new(
            "", // Will trigger auth error
            None, None, None, None, None, None, None,
        );

        let messages = vec![ChatMessage::user().content("Give me JSON").build()];

        let structured_format = StructuredOutputFormat {
            name: "response_schema".to_string(),
            description: Some("Response schema".to_string()),
            schema: Some(json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "number"}
                }
            })),
            strict: Some(true),
        };

        let result = client.chat(&messages, None, Some(structured_format)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_completion_with_suffix() {
        let client = Google::new(
            "", // Will trigger auth error
            None, None, None, None, None, None, None,
        );

        let request = CompletionRequest {
            prompt: "The beginning".to_string(),
            max_tokens: None,
            temperature: None,
        };

        let result = client.complete(&request, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_completion_with_structured_output() {
        let client = Google::new(
            "", // Will trigger auth error
            None, None, None, None, None, None, None,
        );

        let request = CompletionRequest {
            prompt: "Generate JSON".to_string(),
            max_tokens: None,
            temperature: None,
        };

        let structured_format = StructuredOutputFormat {
            name: "completion_schema".to_string(),
            description: Some("Completion schema".to_string()),
            schema: Some(json!({
                "type": "object",
                "properties": {
                    "result": {"type": "string"}
                }
            })),
            strict: Some(true),
        };

        let result = client.complete(&request, Some(structured_format)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_embedding_multiple_texts() {
        let client = Google::new(
            "", // Will trigger auth error
            None, None, None, None, None, None, None,
        );

        let texts = vec![
            "First text".to_string(),
            "Second text".to_string(),
            "Third text".to_string(),
        ];

        let result = client.embed(texts).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_embedding_empty_texts() {
        let client = Google::new(
            "", // Will trigger auth error
            None, None, None, None, None, None, None,
        );

        let texts: Vec<String> = vec![];

        let result = client.embed(texts).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_google_debug_output() {
        let client = Google::new(
            "test-key",
            Some("test-model".to_string()),
            Some(100),
            Some(0.7),
            Some(30),
            Some("system".to_string()),
            Some(0.9),
            Some(50),
        );

        // Just verify we can format debug output
        let debug_str = format!("{client:?}");
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn test_google_field_access() {
        let client = create_test_google();

        // Test that all fields are accessible
        assert!(!client.api_key.is_empty());
        assert!(!client.model.is_empty());
        let _ = client.max_tokens;
        let _ = client.temperature;
        let _ = client.system;
        let _ = client.timeout_seconds;
        let _ = client.top_p;
        let _ = client.top_k;
    }

    #[tokio::test]
    async fn test_chat_stream_auth_error() {
        let client = Google::new(
            "", // Empty API key
            None, None, None, None, None, None, None,
        );

        let messages = vec![ChatMessage::user().content("Hello").build()];

        let result = client.chat_stream(&messages, None, None).await;
        assert!(result.is_err());

        match result.err().unwrap() {
            LLMError::AuthError(msg) => {
                assert_eq!(msg, "Missing Google API key");
            }
            _ => panic!("Expected AuthError"),
        }
    }

    #[test]
    fn test_google_clone_implementation() {
        let client1 = create_test_google();

        // Test we can create another instance with same values
        let client2 = Google::new(
            client1.api_key.clone(),
            Some(client1.model.clone()),
            client1.max_tokens,
            client1.temperature,
            client1.timeout_seconds,
            client1.system.clone(),
            client1.top_p,
            client1.top_k,
        );

        assert_eq!(client1.api_key, client2.api_key);
        assert_eq!(client1.model, client2.model);
        assert_eq!(client1.max_tokens, client2.max_tokens);
        assert_eq!(client1.temperature, client2.temperature);
        assert_eq!(client1.system, client2.system);
    }
}
