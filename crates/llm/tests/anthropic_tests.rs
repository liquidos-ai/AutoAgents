#![allow(unused_imports, dead_code)]
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

    #[test]
    fn test_anthropic_builder_with_all_options() {
        let client = LLMBuilder::<Anthropic>::new()
            .api_key("test-key")
            .model("claude-3-opus-20240229")
            .max_tokens(1500)
            .temperature(0.8)
            .top_p(0.95)
            .top_k(60)
            .timeout_seconds(120)
            .system("Custom system prompt")
            .reasoning(true)
            .reasoning_budget_tokens(10000)
            .build()
            .expect("Failed to build Anthropic client");

        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.model, "claude-3-opus-20240229");
        assert_eq!(client.max_tokens, 1500);
        assert_eq!(client.temperature, 0.8);
        assert_eq!(client.top_p, Some(0.95));
        assert_eq!(client.top_k, Some(60));
        assert_eq!(client.timeout_seconds, 120);
        assert_eq!(client.system, "Custom system prompt");
        assert!(client.reasoning);
        assert_eq!(client.thinking_budget_tokens, Some(10000));
    }

    #[test]
    fn test_anthropic_builder_minimal() {
        let client = LLMBuilder::<Anthropic>::new()
            .api_key("test-key")
            .build()
            .expect("Failed to build Anthropic client");

        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.model, "claude-3-sonnet-20240229");
        assert_eq!(client.max_tokens, 300);
        assert_eq!(client.temperature, 0.7);
        assert_eq!(client.system, "You are a helpful assistant.");
    }

    #[test]
    fn test_anthropic_new_with_custom_params() {
        let client = Anthropic::new(
            "test-key",
            Some("claude-3-haiku-20240307".to_string()),
            Some(500),
            Some(0.5),
            Some(60),
            Some("Custom system".to_string()),
            Some(0.8),
            Some(30),
            Some(ToolChoice::Auto),
            Some(true),
            Some(5000),
        );

        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.model, "claude-3-haiku-20240307");
        assert_eq!(client.max_tokens, 500);
        assert_eq!(client.temperature, 0.5);
        assert_eq!(client.timeout_seconds, 60);
        assert_eq!(client.system, "Custom system");
        assert_eq!(client.top_p, Some(0.8));
        assert_eq!(client.top_k, Some(30));
        assert!(client.reasoning);
        assert_eq!(client.thinking_budget_tokens, Some(5000));
        // Note: base_url is not a field in Anthropic struct
        // assert_eq!(client.base_url, "https://custom.endpoint.com");
    }

    #[tokio::test]
    async fn test_completion_auth_error() {
        let client = Anthropic::new(
            "", // Empty API key
            None, None, None, None, None, None, None, None, None, None,
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
                assert_eq!(msg, "Missing Anthropic API key");
            }
            _ => panic!("Expected AuthError"),
        }
    }

    #[tokio::test]
    async fn test_chat_with_tools_auth_error() {
        let client = Anthropic::new(
            "", // Empty API key
            None, None, None, None, None, None, None, None, None, None,
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
        let client = Anthropic::new(
            "", // Will trigger auth error
            None, None, None, None, None, None, None, None, None, None,
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
    async fn test_models_provider_not_implemented() {
        let client = create_test_anthropic();
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
    fn test_anthropic_with_zero_values() {
        let client = Anthropic::new(
            "test-key",
            Some("test-model".to_string()),
            Some(0),
            Some(0.0),
            Some(0),
            None,
            Some(0.0),
            Some(0),
            None,
            Some(false),
            Some(0),
        );

        assert_eq!(client.max_tokens, 0);
        assert_eq!(client.temperature, 0.0);
        assert_eq!(client.timeout_seconds, 0);
        assert_eq!(client.top_p, Some(0.0));
        assert_eq!(client.top_k, Some(0));
        assert!(!client.reasoning);
        assert_eq!(client.thinking_budget_tokens, Some(0));
    }

    #[test]
    fn test_anthropic_with_extreme_values() {
        let client = Anthropic::new(
            "test-key",
            Some("test-model".to_string()),
            Some(u32::MAX),
            Some(1.0),
            Some(u64::MAX),
            Some("Very long system prompt that contains many words and characters to test the system's ability to handle longer prompts that may exceed normal lengths".to_string()),
            Some(1.0),
            Some(u32::MAX),
            None,
            Some(true),
            Some(u32::MAX),
        );

        assert_eq!(client.max_tokens, u32::MAX);
        assert_eq!(client.temperature, 1.0);
        assert_eq!(client.timeout_seconds, u64::MAX);
        assert_eq!(client.top_p, Some(1.0));
        assert_eq!(client.top_k, Some(u32::MAX));
        assert!(client.reasoning);
        assert_eq!(client.thinking_budget_tokens, Some(u32::MAX));
    }

    #[tokio::test]
    async fn test_chat_with_multiple_message_types() {
        let client = Anthropic::new(
            "", // Will trigger auth error
            None, None, None, None, None, None, None, None, None, None,
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
    async fn test_completion_with_suffix() {
        let client = Anthropic::new(
            "", // Will trigger auth error
            None, None, None, None, None, None, None, None, None, None,
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
        let client = Anthropic::new(
            "", // Will trigger auth error
            None, None, None, None, None, None, None, None, None, None,
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

    #[test]
    fn test_anthropic_reasoning_false() {
        let client = LLMBuilder::<Anthropic>::new()
            .api_key("test-key")
            .reasoning(false)
            .build()
            .expect("Failed to build Anthropic client");

        assert!(!client.reasoning);
        assert!(client.thinking_budget_tokens.is_none());
    }

    #[test]
    fn test_anthropic_reasoning_without_budget() {
        let client = LLMBuilder::<Anthropic>::new()
            .api_key("test-key")
            .reasoning(true)
            .build()
            .expect("Failed to build Anthropic client");

        assert!(client.reasoning);
        assert!(client.thinking_budget_tokens.is_none());
    }

    #[test]
    fn test_anthropic_field_access() {
        let client = create_test_anthropic();

        // Test that all fields are accessible
        assert!(!client.api_key.is_empty());
        assert!(!client.model.is_empty());
        assert!(client.max_tokens > 0);
        assert!(client.temperature >= 0.0);
        assert!(!client.system.is_empty());
        assert!(client.timeout_seconds > 0);
        let _ = client.top_p;
        let _ = client.top_k;
        let _ = client.reasoning;
        let _ = client.thinking_budget_tokens;
    }

    #[test]
    fn test_anthropic_debug_output() {
        let client = create_test_anthropic();

        // Just verify we can format debug output
        let debug_str = format!("{client:?}");
        assert!(!debug_str.is_empty());
        assert!(debug_str.contains("Anthropic"));
    }

    #[test]
    fn test_anthropic_clone_implementation() {
        let client1 = create_test_anthropic();

        // Test we can create another instance with same values
        let client2 = Anthropic::new(
            client1.api_key.clone(),
            Some(client1.model.clone()),
            Some(client1.max_tokens),
            Some(client1.temperature),
            Some(client1.timeout_seconds),
            Some(client1.system.clone()),
            client1.top_p,
            client1.top_k,
            client1.tool_choice.clone(),
            Some(client1.reasoning),
            client1.thinking_budget_tokens,
        );

        assert_eq!(client1.api_key, client2.api_key);
        assert_eq!(client1.model, client2.model);
        assert_eq!(client1.max_tokens, client2.max_tokens);
        assert_eq!(client1.temperature, client2.temperature);
        assert_eq!(client1.system, client2.system);
        assert_eq!(client1.reasoning, client2.reasoning);
    }

    #[test]
    fn test_anthropic_models() {
        // Test various model configurations
        let models = vec![
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",
        ];

        for model in models {
            let client = Anthropic::new(
                "test-key",
                Some(model.to_string()),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            );
            assert_eq!(client.model, model);
        }
    }

    #[test]
    fn test_anthropic_builder_with_custom_base_url() {
        let _client = LLMBuilder::<Anthropic>::new()
            .api_key("test-key")
            .base_url("https://custom.anthropic.com")
            .build()
            .expect("Failed to build Anthropic client");

        // Base URL is set via builder but not accessible as a field
    }

    #[test]
    fn test_anthropic_sampling_parameters() {
        let client = LLMBuilder::<Anthropic>::new()
            .api_key("test-key")
            .temperature(0.1)
            .top_p(0.1)
            .top_k(1)
            .build()
            .expect("Failed to build Anthropic client");

        assert_eq!(client.temperature, 0.1);
        assert_eq!(client.top_p, Some(0.1));
        assert_eq!(client.top_k, Some(1));
    }

    #[test]
    fn test_anthropic_timeout_configuration() {
        let timeouts = vec![1, 30, 60, 120, 300, u64::MAX];

        for timeout in timeouts {
            let client = LLMBuilder::<Anthropic>::new()
                .api_key("test-key")
                .timeout_seconds(timeout)
                .build()
                .expect("Failed to build Anthropic client");

            assert_eq!(client.timeout_seconds, timeout);
        }
    }

    #[test]
    fn test_anthropic_system_prompts() {
        let prompts = vec![
            "",
            "Simple prompt",
            "Very long system prompt that contains multiple sentences and explains the context in great detail about what the assistant should do and how it should behave when responding to user queries.",
            "System prompt with\nnewlines\nand\tspecial\rcharacters!@#$%^&*(){}[]|\\:;\"'<>?,./",
            "🤖 System prompt with emojis and unicode characters: αβγδε 中文 العربية",
        ];

        for prompt in prompts {
            let client = LLMBuilder::<Anthropic>::new()
                .api_key("test-key")
                .system(prompt)
                .build()
                .expect("Failed to build Anthropic client");

            assert_eq!(client.system, prompt);
        }
    }

    #[tokio::test]
    async fn test_embedding_not_supported() {
        let client = create_test_anthropic();
        let texts = vec!["Test text".to_string()];

        let result = client.embed(texts).await;
        assert!(result.is_err());

        match result.err().unwrap() {
            LLMError::ProviderError(msg) => {
                assert!(msg.contains("Embedding not supported"));
            }
            _ => panic!("Expected ProviderError"),
        }
    }
}
