#![allow(unused_imports)]
use autoagents_llm::{
    builder::LLMBuilder,
    chat::{
        ChatMessage, ChatProvider, ChatResponse, ChatRole, FunctionTool, MessageType,
        ReasoningEffort, StructuredOutputFormat, Tool, ToolChoice,
    },
    completion::{CompletionProvider, CompletionRequest},
    embedding::EmbeddingProvider,
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
            None, None, None, None, None, None, None,
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
            None, None, None, None, None, None, None,
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
            None, None, None, None, None,
        );

        let messages = vec![ChatMessage::user().content("Hello").build()];

        let result = client.chat(&messages, None, None).await;
        assert!(result.is_err());

        match result.err().unwrap() {
            LLMError::AuthError(msg) => {
                assert_eq!(msg, "Missing OpenAI API key");
            }
            _ => panic!("Expected AuthError"),
        }
    }

    #[test]
    fn test_openai_builder_with_all_options() {
        let client = LLMBuilder::<OpenAI>::new()
            .api_key("test-key")
            .model("gpt-4-turbo")
            .max_tokens(2000)
            .temperature(0.8)
            .top_p(0.95)
            .timeout_seconds(120)
            .system("Custom system prompt")
            .base_url("https://custom.openai.com/v1/")
            .reasoning_effort(ReasoningEffort::Medium)
            .build()
            .expect("Failed to build OpenAI client");

        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.model, "gpt-4-turbo");
        assert_eq!(client.max_tokens, Some(2000));
        assert_eq!(client.temperature, Some(0.8));
        assert_eq!(client.top_p, Some(0.95));
        assert_eq!(client.timeout_seconds, Some(120));
        assert_eq!(client.system, Some("Custom system prompt".to_string()));
        assert_eq!(client.base_url.as_str(), "https://custom.openai.com/v1/");
        assert_eq!(client.reasoning_effort, Some("medium".to_string()));
    }

    #[test]
    fn test_openai_builder_minimal() {
        let client = LLMBuilder::<OpenAI>::new()
            .api_key("test-key")
            .build()
            .expect("Failed to build OpenAI client");

        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.model, "gpt-3.5-turbo");
        assert_eq!(client.base_url.as_str(), "https://api.openai.com/v1/");
        assert!(client.max_tokens.is_none());
        assert!(client.temperature.is_none());
        assert!(client.system.is_none());
    }

    #[test]
    fn test_openai_new_with_all_params() {
        let client = OpenAI::new(
            "test-key",
            Some("https://api.custom.com/v1/".to_string()),
            Some("gpt-4o".to_string()),
            Some(1000),
            Some(0.9),
            Some(90),
            Some("Test system".to_string()),
            Some(0.8),
            None, // top_k
            Some("base64".to_string()),
            Some(3072),
            None,                     // tool_choice
            Some("high".to_string()), // reasoning_effort as String
            None,                     // voice
            Some(true),
            Some("medium".to_string()),
            Some("exact".to_string()),
            None, // country
            None, // city
            None, // region
        );

        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.model, "gpt-4o");
        assert_eq!(client.max_tokens, Some(1000));
        assert_eq!(client.temperature, Some(0.9));
        assert_eq!(client.top_p, Some(0.8));
        assert_eq!(client.timeout_seconds, Some(90));
        assert_eq!(client.system, Some("Test system".to_string()));
        assert_eq!(client.base_url.as_str(), "https://api.custom.com/v1/");
        assert_eq!(client.embedding_encoding_format, Some("base64".to_string()));
        assert_eq!(client.embedding_dimensions, Some(3072));
        assert_eq!(client.enable_web_search, Some(true));
        assert_eq!(client.web_search_context_size, Some("medium".to_string()));
        assert_eq!(
            client.web_search_user_location_type,
            Some("exact".to_string())
        );
        assert_eq!(client.reasoning_effort, Some("high".to_string()));
    }

    #[tokio::test]
    async fn test_completion_auth_error() {
        let client = OpenAI::new(
            "", // Empty API key
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None,
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
                assert_eq!(msg, "Missing OpenAI API key");
            }
            _ => panic!("Expected AuthError"),
        }
    }

    #[tokio::test]
    async fn test_embedding_auth_error() {
        let client = OpenAI::new(
            "", // Empty API key
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None,
        );

        let texts = vec!["Test text".to_string()];

        let result = client.embed(texts).await;
        assert!(result.is_err());

        match result.err().unwrap() {
            LLMError::AuthError(msg) => {
                assert_eq!(msg, "Missing OpenAI API key");
            }
            _ => panic!("Expected AuthError"),
        }
    }

    #[tokio::test]
    async fn test_chat_with_tools_auth_error() {
        let client = OpenAI::new(
            "", // Empty API key
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None,
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
        let client = OpenAI::new(
            "", // Will trigger auth error
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None,
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

    #[test]
    fn test_openai_web_search_methods() {
        let mut client = OpenAI::new(
            "test-key", None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None,
        );

        // Test all web search setter methods
        client = client.set_enable_web_search(true);
        client = client.set_web_search_context_size("small");
        client = client.set_web_search_user_location_type("exact");
        client = client.set_web_search_user_location_approximate_country("CA");
        client = client.set_web_search_user_location_approximate_city("Toronto");
        client = client.set_web_search_user_location_approximate_region("ON");

        assert_eq!(client.enable_web_search, Some(true));
        assert_eq!(client.web_search_context_size, Some("small".to_string()));
        assert_eq!(
            client.web_search_user_location_type,
            Some("exact".to_string())
        );
        assert_eq!(
            client.web_search_user_location_approximate_country,
            Some("CA".to_string())
        );
        assert_eq!(
            client.web_search_user_location_approximate_city,
            Some("Toronto".to_string())
        );
        assert_eq!(
            client.web_search_user_location_approximate_region,
            Some("ON".to_string())
        );
    }

    #[test]
    fn test_openai_web_search_chaining() {
        let client = OpenAI::new(
            "test-key", None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None,
        )
        .set_enable_web_search(false)
        .set_web_search_context_size("large")
        .set_web_search_user_location_type("approximate");

        assert_eq!(client.enable_web_search, Some(false));
        assert_eq!(client.web_search_context_size, Some("large".to_string()));
        assert_eq!(
            client.web_search_user_location_type,
            Some("approximate".to_string())
        );
    }

    #[test]
    fn test_openai_models() {
        // Test various model configurations
        let models = vec![
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "o1-preview",
            "o1-mini",
        ];

        for model in models {
            let client = OpenAI::new(
                "test-key",
                None,
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
    fn test_openai_reasoning_efforts() {
        let efforts = vec![
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::High,
        ];

        for effort in efforts {
            let effort_clone = effort.clone();
            let client = LLMBuilder::<OpenAI>::new()
                .api_key("test-key")
                .reasoning_effort(effort)
                .build()
                .expect("Failed to build OpenAI client");

            assert_eq!(
                client.reasoning_effort,
                Some(match effort_clone {
                    ReasoningEffort::Low => "low".to_string(),
                    ReasoningEffort::Medium => "medium".to_string(),
                    ReasoningEffort::High => "high".to_string(),
                })
            );
        }
    }

    #[test]
    fn test_openai_with_zero_values() {
        let client = OpenAI::new(
            "test-key",
            None,
            Some("test-model".to_string()),
            Some(0),
            Some(0.0),
            Some(0),
            None,
            Some(0.0),
            None,
            None,
            Some(0),
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
            None,
            None,
        );

        assert_eq!(client.max_tokens, Some(0));
        assert_eq!(client.temperature, Some(0.0));
        assert_eq!(client.top_p, Some(0.0));
        assert_eq!(client.timeout_seconds, Some(0));
        assert_eq!(client.embedding_dimensions, Some(0));
        assert_eq!(client.enable_web_search, Some(false));
    }

    #[test]
    fn test_openai_with_extreme_values() {
        let client = OpenAI::new(
            "test-key",
            Some("https://very-long-custom-endpoint-url.example.com/api/v1/chat/completions".to_string()),
            Some("test-model".to_string()),
            Some(u32::MAX),
            Some(2.0),
            Some(u64::MAX),
            Some("Very long system prompt that contains many words and characters to test the system's ability to handle longer prompts that may exceed normal lengths and contain various special characters and unicode symbols".to_string()),
            Some(1.0),
            None,
            Some("float".to_string()),
            Some(u32::MAX),
            None,
            Some("high".to_string()),
            None,
            Some(true),
            Some("extra-large".to_string()),
            Some("precise".to_string()),
            None,
            None,
            None,
        );

        assert_eq!(client.max_tokens, Some(u32::MAX));
        assert_eq!(client.temperature, Some(2.0));
        assert_eq!(client.top_p, Some(1.0));
        assert_eq!(client.timeout_seconds, Some(u64::MAX));
        assert_eq!(client.embedding_dimensions, Some(u32::MAX));
        assert_eq!(client.reasoning_effort, Some("high".to_string()));
    }

    #[tokio::test]
    async fn test_chat_with_multiple_message_types() {
        let client = OpenAI::new(
            "", // Will trigger auth error
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None,
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
        let client = OpenAI::new(
            "", // Will trigger auth error
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None,
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
    async fn test_embedding_multiple_texts() {
        let client = OpenAI::new(
            "", // Will trigger auth error
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None,
        );

        let texts = vec![
            "First text".to_string(),
            "Second text".to_string(),
            "Third text".to_string(),
        ];

        let result = client.embed(texts).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_openai_field_access() {
        let client = create_test_openai();

        // Test that all fields are accessible
        assert!(!client.api_key.is_empty());
        assert!(!client.model.is_empty());
        assert!(client.max_tokens.is_some());
        assert!(client.temperature.is_some());
        assert!(client.system.is_some());
        assert!(!client.base_url.as_str().is_empty());
        let _ = client.top_p;
        let _ = client.timeout_seconds;
        let _ = client.embedding_encoding_format;
        let _ = client.embedding_dimensions;
        let _ = client.enable_web_search;
        let _ = client.web_search_context_size;
        let _ = client.web_search_user_location_type;
        let _ = client.web_search_user_location_approximate_country;
        let _ = client.web_search_user_location_approximate_city;
        let _ = client.web_search_user_location_approximate_region;
        let _ = client.reasoning_effort;
    }

    // Debug trait not implemented for OpenAI struct

    #[test]
    fn test_openai_clone_implementation() {
        let client1 = create_test_openai();

        // Test we can create another instance with same values
        let client2 = OpenAI::new(
            client1.api_key.clone(),
            Some(client1.base_url.to_string()),
            Some(client1.model.clone()),
            client1.max_tokens,
            client1.temperature,
            client1.timeout_seconds,
            client1.system.clone(),
            client1.top_p,
            None,
            client1.embedding_encoding_format.clone(),
            client1.embedding_dimensions,
            None,
            client1.reasoning_effort.clone(),
            None,
            client1.enable_web_search,
            client1.web_search_context_size.clone(),
            client1.web_search_user_location_type.clone(),
            None,
            None,
            None,
        );

        assert_eq!(client1.api_key, client2.api_key);
        assert_eq!(client1.model, client2.model);
        assert_eq!(client1.max_tokens, client2.max_tokens);
        assert_eq!(client1.temperature, client2.temperature);
        assert_eq!(client1.system, client2.system);
        assert_eq!(client1.base_url, client2.base_url);
    }

    #[tokio::test]
    async fn test_models_provider_auth_error() {
        let client = OpenAI::new(
            "", // Empty API key
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None,
        );

        let result = client.list_models(None).await;
        assert!(result.is_err());

        match result.err().unwrap() {
            LLMError::AuthError(msg) => {
                assert_eq!(msg, "Missing OpenAI API key");
            }
            _ => panic!("Expected AuthError"),
        }
    }

    #[test]
    fn test_openai_penalty_bounds() {
        // Test that we can set penalties within valid bounds
        let penalties = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

        for _penalty in penalties {
            let _client = LLMBuilder::<OpenAI>::new()
                .api_key("test-key")
                .build()
                .expect("Failed to build OpenAI client");
        }
    }

    #[test]
    fn test_openai_embedding_formats() {
        let formats = vec!["float", "base64"];

        for format in formats {
            let client = LLMBuilder::<OpenAI>::new()
                .api_key("test-key")
                .embedding_encoding_format(format)
                .build()
                .expect("Failed to build OpenAI client");

            assert_eq!(client.embedding_encoding_format, Some(format.to_string()));
        }
    }

    #[test]
    fn test_openai_logit_bias() {
        let _logit_bias = json!({
            "50256": -100,
            "198": -50,
            "220": 10
        });

        let _client = LLMBuilder::<OpenAI>::new()
            .api_key("test-key")
            .build()
            .expect("Failed to build OpenAI client");
    }

    #[tokio::test]
    async fn test_completion_with_structured_output() {
        let client = OpenAI::new(
            "", // Will trigger auth error
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None,
        );

        let request = CompletionRequest {
            prompt: "Generate JSON".to_string(),
            max_tokens: None,
            temperature: None,
        };

        let structured_format = StructuredOutputFormat {
            name: "result_schema".to_string(),
            description: Some("Result schema".to_string()),
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
    fn test_openai_embedding_dimensions() {
        let dimensions = vec![256, 512, 1024, 1536, 3072];

        for dim in dimensions {
            let client = LLMBuilder::<OpenAI>::new()
                .api_key("test-key")
                .embedding_dimensions(dim)
                .build()
                .expect("Failed to build OpenAI client");

            assert_eq!(client.embedding_dimensions, Some(dim));
        }
    }

    #[test]
    fn test_openai_system_prompts() {
        let prompts = vec![
            "",
            "Simple prompt",
            "Very long system prompt that contains multiple sentences and explains the context in great detail about what the assistant should do and how it should behave when responding to user queries and handling various types of requests.",
            "System prompt with\nnewlines\nand\tspecial\rcharacters!@#$%^&*(){}[]|\\:;\"'<>?,./",
            "🤖 System prompt with emojis and unicode characters: αβγδε 中文 العربية русский",
        ];

        for prompt in prompts {
            let client = LLMBuilder::<OpenAI>::new()
                .api_key("test-key")
                .system(prompt)
                .build()
                .expect("Failed to build OpenAI client");

            assert_eq!(client.system, Some(prompt.to_string()));
        }
    }
}
