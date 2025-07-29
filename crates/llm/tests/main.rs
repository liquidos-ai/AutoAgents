//! Main test file that includes all backend-specific test modules

#[cfg(feature = "anthropic")]
mod anthropic_tests;

#[cfg(feature = "openai")]
mod openai_tests;

#[cfg(feature = "google")]
mod google_tests;

#[cfg(feature = "ollama")]
mod ollama_tests;

#[cfg(feature = "azure_openai")]
mod azure_openai_tests;

#[cfg(any(
    feature = "deepseek",
    feature = "xai",
    feature = "phind",
    feature = "groq"
))]
mod other_backends_tests;

// Integration tests that test common functionality across all backends
#[cfg(test)]
mod common_tests {
    use autoagents_llm::{
        builder::LLMBackend,
        chat::{ChatMessage, ChatMessageBuilder, ChatRole, ImageMime, MessageType},
        FunctionCall, ToolCall,
    };
    use serde_json::json;

    #[test]
    fn test_chat_message_builder() {
        // Test user message
        let msg = ChatMessage::user().content("Hello").build();
        assert_eq!(msg.role, ChatRole::User);
        assert_eq!(msg.content, "Hello");
        matches!(msg.message_type, MessageType::Text);

        // Test assistant message
        let msg = ChatMessage::assistant().content("Response").build();
        assert_eq!(msg.role, ChatRole::Assistant);
        assert_eq!(msg.content, "Response");

        // Test system message
        let msg = ChatMessageBuilder::new(ChatRole::System)
            .content("System prompt")
            .build();
        assert_eq!(msg.role, ChatRole::System);
        assert_eq!(msg.content, "System prompt");

        // Test tool message
        let msg = ChatMessageBuilder::new(ChatRole::Tool)
            .content("Tool output")
            .build();
        assert_eq!(msg.role, ChatRole::Tool);
        assert_eq!(msg.content, "Tool output");
    }

    #[test]
    fn test_message_with_image() {
        let image_data = vec![0xFF, 0xD8, 0xFF]; // JPEG header
        let msg = ChatMessage::user()
            .content("Look at this image")
            .image(ImageMime::JPEG, image_data.clone())
            .build();

        match &msg.message_type {
            MessageType::Image((mime, data)) => {
                assert_eq!(mime.mime_type(), "image/jpeg");
                assert_eq!(data, &image_data);
            }
            _ => panic!("Expected Image message type"),
        }
    }

    #[test]
    fn test_message_with_image_url() {
        let url = "https://example.com/image.png";
        let msg = ChatMessage::user()
            .content("Check this URL")
            .image_url(url)
            .build();

        match &msg.message_type {
            MessageType::ImageURL(image_url) => {
                assert_eq!(image_url, url);
            }
            _ => panic!("Expected ImageURL message type"),
        }
    }

    #[test]
    fn test_message_with_pdf() {
        let pdf_data = vec![0x25, 0x50, 0x44, 0x46]; // PDF header
        let msg = ChatMessage::user()
            .content("Analyze this PDF")
            .pdf(pdf_data.clone())
            .build();

        match &msg.message_type {
            MessageType::Pdf(data) => {
                assert_eq!(data, &pdf_data);
            }
            _ => panic!("Expected Pdf message type"),
        }
    }

    #[test]
    fn test_message_with_tool_calls() {
        let tool_calls = vec![ToolCall {
            id: "call_123".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "get_weather".to_string(),
                arguments: json!({"location": "New York"}).to_string(),
            },
        }];

        let msg = ChatMessage::assistant()
            .content("I'll check the weather")
            .tool_use(tool_calls.clone())
            .build();

        match &msg.message_type {
            MessageType::ToolUse(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].id, "call_123");
                assert_eq!(calls[0].function.name, "get_weather");
            }
            _ => panic!("Expected ToolUse message type"),
        }
    }

    #[test]
    fn test_message_with_tool_responses() {
        let tool_responses = vec![ToolCall {
            id: "call_123".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "get_weather".to_string(),
                arguments: "Sunny, 72°F".to_string(),
            },
        }];

        let msg = ChatMessageBuilder::new(ChatRole::Tool)
            .content("")
            .tool_result(tool_responses.clone())
            .build();

        match &msg.message_type {
            MessageType::ToolResult(responses) => {
                assert_eq!(responses.len(), 1);
                assert_eq!(responses[0].id, "call_123");
                assert_eq!(responses[0].function.arguments, "Sunny, 72°F");
            }
            _ => panic!("Expected ToolResult message type"),
        }
    }

    #[test]
    fn test_llm_backend_enum() {
        // Test Display implementation via Debug
        assert_eq!(format!("{:?}", LLMBackend::OpenAI), "OpenAI");
        assert_eq!(format!("{:?}", LLMBackend::Anthropic), "Anthropic");
        assert_eq!(format!("{:?}", LLMBackend::Ollama), "Ollama");
        assert_eq!(format!("{:?}", LLMBackend::DeepSeek), "DeepSeek");
        assert_eq!(format!("{:?}", LLMBackend::XAI), "XAI");
        assert_eq!(format!("{:?}", LLMBackend::Phind), "Phind");
        assert_eq!(format!("{:?}", LLMBackend::Google), "Google");
        assert_eq!(format!("{:?}", LLMBackend::Groq), "Groq");
        assert_eq!(format!("{:?}", LLMBackend::AzureOpenAI), "AzureOpenAI");

        // Test FromStr implementation
        use std::str::FromStr;
        assert!(matches!(
            LLMBackend::from_str("openai").unwrap(),
            LLMBackend::OpenAI
        ));
        assert!(matches!(
            LLMBackend::from_str("anthropic").unwrap(),
            LLMBackend::Anthropic
        ));
        assert!(matches!(
            LLMBackend::from_str("ollama").unwrap(),
            LLMBackend::Ollama
        ));
        assert!(matches!(
            LLMBackend::from_str("deepseek").unwrap(),
            LLMBackend::DeepSeek
        ));
        assert!(matches!(
            LLMBackend::from_str("xai").unwrap(),
            LLMBackend::XAI
        ));
        assert!(matches!(
            LLMBackend::from_str("phind").unwrap(),
            LLMBackend::Phind
        ));
        assert!(matches!(
            LLMBackend::from_str("google").unwrap(),
            LLMBackend::Google
        ));
        assert!(matches!(
            LLMBackend::from_str("groq").unwrap(),
            LLMBackend::Groq
        ));
        assert!(matches!(
            LLMBackend::from_str("azure-openai").unwrap(),
            LLMBackend::AzureOpenAI
        ));

        // Test invalid backend
        assert!(LLMBackend::from_str("invalid").is_err());
    }

    #[test]
    fn test_structured_output_format() {
        use autoagents_llm::chat::StructuredOutputFormat;

        let schema = StructuredOutputFormat {
            name: "PersonInfo".to_string(),
            description: Some("Information about a person".to_string()),
            schema: Some(json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Person's full name"
                    },
                    "age": {
                        "type": "integer",
                        "description": "Person's age"
                    },
                    "email": {
                        "type": "string",
                        "format": "email",
                        "description": "Person's email address"
                    }
                },
                "required": ["name", "age"],
                "additionalProperties": false
            })),
            strict: Some(true),
        };

        assert_eq!(schema.name, "PersonInfo");
        assert_eq!(
            schema.description,
            Some("Information about a person".to_string())
        );
        assert!(schema.schema.is_some());
        assert_eq!(schema.strict, Some(true));

        // Test with minimal schema
        let minimal_schema = StructuredOutputFormat {
            name: "SimpleSchema".to_string(),
            description: None,
            schema: None,
            strict: None,
        };

        assert_eq!(minimal_schema.name, "SimpleSchema");
        assert!(minimal_schema.description.is_none());
        assert!(minimal_schema.schema.is_none());
        assert!(minimal_schema.strict.is_none());
    }

    #[test]
    fn test_function_and_tool_structures() {
        use autoagents_llm::chat::{FunctionTool, Tool};

        let function = FunctionTool {
            name: "calculate_sum".to_string(),
            description: "Calculate the sum of two numbers".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number"
                    }
                },
                "required": ["a", "b"]
            }),
        };

        assert_eq!(function.name, "calculate_sum");
        assert_eq!(function.description, "Calculate the sum of two numbers");
        assert!(function.parameters.is_object());

        let tool = Tool {
            tool_type: "function".to_string(),
            function,
        };

        assert_eq!(tool.tool_type, "function");
        assert_eq!(tool.function.name, "calculate_sum");
    }
}

// Test module for testing feature flag configurations
#[cfg(test)]
mod feature_tests {
    #[test]
    #[cfg(feature = "full")]
    fn test_full_feature_enables_all_backends() {
        assert!(cfg!(feature = "openai"));
        assert!(cfg!(feature = "anthropic"));
        assert!(cfg!(feature = "ollama"));
        assert!(cfg!(feature = "deepseek"));
        assert!(cfg!(feature = "xai"));
        assert!(cfg!(feature = "phind"));
        assert!(cfg!(feature = "google"));
        assert!(cfg!(feature = "groq"));
        assert!(cfg!(feature = "azure_openai"));
    }
}
