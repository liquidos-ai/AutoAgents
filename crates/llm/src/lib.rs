//! AutoAgents LLM is a unified interface for interacting with Large Language Model providers.
//!
//! # Overview
//! This crate provides a consistent API for working with different LLM backends by abstracting away
//! provider-specific implementation details. It supports:
//!
//! - Chat-based interactions
//! - Text completion
//! - Embeddings generation
//! - Multiple providers (OpenAI, Anthropic, etc.)
//! - Request validation and retry logic
//!
//! # Architecture
//! The crate is organized into modules that handle different aspects of LLM interactions:

use chat::Tool;
use serde::{Deserialize, Serialize};

/// Backend implementations for supported LLM providers like OpenAI, Anthropic, etc.
pub mod backends;

/// Builder pattern for configuring and instantiating LLM providers
pub mod builder;

/// Chat-based interactions with language models (e.g. ChatGPT style)
pub mod chat;

/// Text completion capabilities (e.g. GPT-3 style completion)
pub mod completion;

/// Vector embeddings generation for text
pub mod embedding;

/// Error types and handling
pub mod error;

/// Evaluator for LLM providers
pub mod evaluator;

/// Secret store for storing API keys and other sensitive information
pub mod secret_store;

/// Listing models support
pub mod models;

mod tool;
pub use tool::{ToolCallError, ToolInputT, ToolT};

/// Core trait that all LLM providers must implement, combining chat, completion
/// and embedding capabilities into a unified interface
pub trait LLMProvider:
    chat::ChatProvider
    + completion::CompletionProvider
    + embedding::EmbeddingProvider
    + models::ModelsProvider
    + Send
    + Sync
    + 'static
{
    fn tools(&self) -> Option<&[Tool]> {
        None
    }
}

/// Tool call represents a function call that an LLM wants to make.
/// This is a standardized structure used across all providers.
#[derive(Debug, Deserialize, Serialize, Clone, Eq, PartialEq)]
pub struct ToolCall {
    /// The ID of the tool call.
    pub id: String,
    /// The type of the tool call (usually "function").
    #[serde(rename = "type")]
    pub call_type: String,
    /// The function to call.
    pub function: FunctionCall,
}

/// FunctionCall contains details about which function to call and with what arguments.
#[derive(Debug, Deserialize, Serialize, Clone, Eq, PartialEq)]
pub struct FunctionCall {
    /// The name of the function to call.
    pub name: String,
    /// The arguments to pass to the function, typically serialized as a JSON string.
    pub arguments: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::ChatProvider;
    use crate::completion::CompletionProvider;
    use crate::embedding::EmbeddingProvider;
    use async_trait::async_trait;
    use serde_json::json;

    #[test]
    fn test_tool_call_creation() {
        let tool_call = ToolCall {
            id: "call_123".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "test_function".to_string(),
                arguments: "{\"param\": \"value\"}".to_string(),
            },
        };

        assert_eq!(tool_call.id, "call_123");
        assert_eq!(tool_call.call_type, "function");
        assert_eq!(tool_call.function.name, "test_function");
        assert_eq!(tool_call.function.arguments, "{\"param\": \"value\"}");
    }

    #[test]
    fn test_tool_call_serialization() {
        let tool_call = ToolCall {
            id: "call_456".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "serialize_test".to_string(),
                arguments: "{\"test\": true}".to_string(),
            },
        };

        let serialized = serde_json::to_string(&tool_call).unwrap();
        let deserialized: ToolCall = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.id, "call_456");
        assert_eq!(deserialized.call_type, "function");
        assert_eq!(deserialized.function.name, "serialize_test");
        assert_eq!(deserialized.function.arguments, "{\"test\": true}");
    }

    #[test]
    fn test_tool_call_equality() {
        let tool_call1 = ToolCall {
            id: "call_1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "equal_test".to_string(),
                arguments: "{}".to_string(),
            },
        };

        let tool_call2 = ToolCall {
            id: "call_1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "equal_test".to_string(),
                arguments: "{}".to_string(),
            },
        };

        let tool_call3 = ToolCall {
            id: "call_2".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "equal_test".to_string(),
                arguments: "{}".to_string(),
            },
        };

        assert_eq!(tool_call1, tool_call2);
        assert_ne!(tool_call1, tool_call3);
    }

    #[test]
    fn test_tool_call_clone() {
        let tool_call = ToolCall {
            id: "clone_test".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "test_clone".to_string(),
                arguments: "{\"clone\": true}".to_string(),
            },
        };

        let cloned = tool_call.clone();
        assert_eq!(tool_call, cloned);
        assert_eq!(tool_call.id, cloned.id);
        assert_eq!(tool_call.function.name, cloned.function.name);
    }

    #[test]
    fn test_tool_call_debug() {
        let tool_call = ToolCall {
            id: "debug_test".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "debug_function".to_string(),
                arguments: "{}".to_string(),
            },
        };

        let debug_str = format!("{tool_call:?}");
        assert!(debug_str.contains("ToolCall"));
        assert!(debug_str.contains("debug_test"));
        assert!(debug_str.contains("debug_function"));
    }

    #[test]
    fn test_function_call_creation() {
        let function_call = FunctionCall {
            name: "test_function".to_string(),
            arguments: "{\"param1\": \"value1\", \"param2\": 42}".to_string(),
        };

        assert_eq!(function_call.name, "test_function");
        assert_eq!(
            function_call.arguments,
            "{\"param1\": \"value1\", \"param2\": 42}"
        );
    }

    #[test]
    fn test_function_call_serialization() {
        let function_call = FunctionCall {
            name: "serialize_function".to_string(),
            arguments: "{\"data\": [1, 2, 3]}".to_string(),
        };

        let serialized = serde_json::to_string(&function_call).unwrap();
        let deserialized: FunctionCall = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.name, "serialize_function");
        assert_eq!(deserialized.arguments, "{\"data\": [1, 2, 3]}");
    }

    #[test]
    fn test_function_call_equality() {
        let func1 = FunctionCall {
            name: "equal_func".to_string(),
            arguments: "{}".to_string(),
        };

        let func2 = FunctionCall {
            name: "equal_func".to_string(),
            arguments: "{}".to_string(),
        };

        let func3 = FunctionCall {
            name: "different_func".to_string(),
            arguments: "{}".to_string(),
        };

        assert_eq!(func1, func2);
        assert_ne!(func1, func3);
    }

    #[test]
    fn test_function_call_clone() {
        let function_call = FunctionCall {
            name: "clone_func".to_string(),
            arguments: "{\"clone\": \"test\"}".to_string(),
        };

        let cloned = function_call.clone();
        assert_eq!(function_call, cloned);
        assert_eq!(function_call.name, cloned.name);
        assert_eq!(function_call.arguments, cloned.arguments);
    }

    #[test]
    fn test_function_call_debug() {
        let function_call = FunctionCall {
            name: "debug_func".to_string(),
            arguments: "{}".to_string(),
        };

        let debug_str = format!("{function_call:?}");
        assert!(debug_str.contains("FunctionCall"));
        assert!(debug_str.contains("debug_func"));
    }

    #[test]
    fn test_tool_call_with_empty_values() {
        let tool_call = ToolCall {
            id: String::new(),
            call_type: String::new(),
            function: FunctionCall {
                name: String::new(),
                arguments: String::new(),
            },
        };

        assert!(tool_call.id.is_empty());
        assert!(tool_call.call_type.is_empty());
        assert!(tool_call.function.name.is_empty());
        assert!(tool_call.function.arguments.is_empty());
    }

    #[test]
    fn test_tool_call_with_complex_arguments() {
        let complex_args = json!({
            "nested": {
                "array": [1, 2, 3],
                "object": {
                    "key": "value"
                }
            },
            "simple": "string"
        });

        let tool_call = ToolCall {
            id: "complex_call".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "complex_function".to_string(),
                arguments: complex_args.to_string(),
            },
        };

        let serialized = serde_json::to_string(&tool_call).unwrap();
        let deserialized: ToolCall = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.id, "complex_call");
        assert_eq!(deserialized.function.name, "complex_function");
        // Arguments should be preserved as string
        assert!(deserialized.function.arguments.contains("nested"));
        assert!(deserialized.function.arguments.contains("array"));
    }

    #[test]
    fn test_tool_call_with_unicode() {
        let tool_call = ToolCall {
            id: "unicode_call".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "unicode_function".to_string(),
                arguments: "{\"message\": \"Hello 世界! 🌍\"}".to_string(),
            },
        };

        let serialized = serde_json::to_string(&tool_call).unwrap();
        let deserialized: ToolCall = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.id, "unicode_call");
        assert_eq!(deserialized.function.name, "unicode_function");
        assert!(deserialized.function.arguments.contains("Hello 世界! 🌍"));
    }

    #[test]
    fn test_tool_call_large_arguments() {
        let large_arg = "x".repeat(10000);
        let tool_call = ToolCall {
            id: "large_call".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "large_function".to_string(),
                arguments: format!("{{\"large_param\": \"{large_arg}\"}}"),
            },
        };

        let serialized = serde_json::to_string(&tool_call).unwrap();
        let deserialized: ToolCall = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.id, "large_call");
        assert_eq!(deserialized.function.name, "large_function");
        assert!(deserialized.function.arguments.len() > 10000);
    }

    // Mock LLM provider for testing
    struct MockLLMProvider;

    #[async_trait]
    impl chat::ChatProvider for MockLLMProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[chat::ChatMessage],
            _tools: Option<&[chat::Tool]>,
            _json_schema: Option<chat::StructuredOutputFormat>,
        ) -> Result<Box<dyn chat::ChatResponse>, error::LLMError> {
            Ok(Box::new(MockChatResponse {
                text: Some("Mock response".to_string()),
            }))
        }
    }

    #[async_trait]
    impl completion::CompletionProvider for MockLLMProvider {
        async fn complete(
            &self,
            _req: &completion::CompletionRequest,
            _json_schema: Option<chat::StructuredOutputFormat>,
        ) -> Result<completion::CompletionResponse, error::LLMError> {
            Ok(completion::CompletionResponse {
                text: "Mock completion".to_string(),
            })
        }
    }

    #[async_trait]
    impl embedding::EmbeddingProvider for MockLLMProvider {
        async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, error::LLMError> {
            let mut embeddings = Vec::new();
            for (i, _) in input.iter().enumerate() {
                embeddings.push(vec![i as f32, (i + 1) as f32]);
            }
            Ok(embeddings)
        }
    }

    #[async_trait]
    impl models::ModelsProvider for MockLLMProvider {}

    impl LLMProvider for MockLLMProvider {
        fn tools(&self) -> Option<&[chat::Tool]> {
            None
        }
    }

    struct MockChatResponse {
        text: Option<String>,
    }

    impl chat::ChatResponse for MockChatResponse {
        fn text(&self) -> Option<String> {
            self.text.clone()
        }

        fn tool_calls(&self) -> Option<Vec<ToolCall>> {
            None
        }
    }

    impl std::fmt::Debug for MockChatResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "MockChatResponse")
        }
    }

    impl std::fmt::Display for MockChatResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.text.as_deref().unwrap_or(""))
        }
    }

    #[tokio::test]
    async fn test_llm_provider_trait_chat() {
        let provider = MockLLMProvider;
        let messages = vec![chat::ChatMessage::user().content("Test").build()];

        let response = provider.chat(&messages, None).await.unwrap();
        assert_eq!(response.text(), Some("Mock response".to_string()));
    }

    #[tokio::test]
    async fn test_llm_provider_trait_completion() {
        let provider = MockLLMProvider;
        let request = completion::CompletionRequest::new("Test prompt");

        let response = provider.complete(&request, None).await.unwrap();
        assert_eq!(response.text, "Mock completion");
    }

    #[tokio::test]
    async fn test_llm_provider_trait_embedding() {
        let provider = MockLLMProvider;
        let input = vec!["First".to_string(), "Second".to_string()];

        let embeddings = provider.embed(input).await.unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0], vec![0.0, 1.0]);
        assert_eq!(embeddings[1], vec![1.0, 2.0]);
    }

    #[test]
    fn test_llm_provider_tools() {
        let provider = MockLLMProvider;
        assert!(provider.tools().is_none());
    }
}
