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

use std::fmt::Display;

use serde::{Deserialize, Serialize};

// The OpenAI Responses backend supports a WASI Preview2 (`wasm32-wasip2`)
// HTTP transport via the `wasi-http` feature using `golem-wasi-http` over the
// `wasi:http` host interface. The unsupported combinations below produce
// precise, actionable compile errors.

// 1. Any HTTP provider feature on non-WASI browser wasm (e.g. `wasm32-unknown-unknown`)
//    is unsupported: there is no socket/HTTP host interface available there.
#[cfg(all(
    target_arch = "wasm32",
    not(target_os = "wasi"),
    any(
        feature = "openai",
        feature = "anthropic",
        feature = "ollama",
        feature = "deepseek",
        feature = "xai",
        feature = "phind",
        feature = "google",
        feature = "groq",
        feature = "azure_openai",
        feature = "openrouter",
        feature = "minimax"
    )
))]
compile_error!(
    "autoagents-llm HTTP provider backends are not supported on non-WASI wasm32 targets \
(such as wasm32-unknown-unknown). Use a native target, or target wasm32-wasip2 with the \
`wasi-http` feature for the OpenAI Responses backend."
);

// 2. WASI Preview1 (`wasm32-wasip1`) has no Preview2 HTTP host interface; the only
//    WASI HTTP transport shipped by this crate requires Preview2.
#[cfg(all(
    target_arch = "wasm32",
    target_os = "wasi",
    target_env = "p1",
    any(
        feature = "openai",
        feature = "anthropic",
        feature = "ollama",
        feature = "deepseek",
        feature = "xai",
        feature = "phind",
        feature = "google",
        feature = "groq",
        feature = "azure_openai",
        feature = "openrouter",
        feature = "minimax"
    )
))]
compile_error!(
    "autoagents-llm HTTP provider backends are not supported on wasm32-wasip1. \
WASI HTTP requires the Preview2 target (wasm32-wasip2); rebuild with `--target wasm32-wasip2` \
and the `wasi-http` feature for the OpenAI Responses backend."
);

// 3. On WASI Preview2 the OpenAI Responses backend requires the `wasi-http` feature
//    to pull in the `wasip2` HTTP bindings.
#[cfg(all(
    target_arch = "wasm32",
    target_os = "wasi",
    target_env = "p2",
    feature = "openai",
    not(feature = "wasi-http")
))]
compile_error!(
    "autoagents-llm OpenAI Responses backend on wasm32-wasip2 requires the `wasi-http` feature. \
Rebuild with `--features openai,wasi-http`."
);

// 4. Other HTTP providers are not yet wired to the WASI Preview2 transport.
#[cfg(all(
    target_arch = "wasm32",
    target_os = "wasi",
    target_env = "p2",
    any(
        feature = "anthropic",
        feature = "ollama",
        feature = "deepseek",
        feature = "xai",
        feature = "phind",
        feature = "google",
        feature = "groq",
        feature = "azure_openai",
        feature = "openrouter",
        feature = "minimax"
    )
))]
compile_error!(
    "autoagents-llm only supports the OpenAI Responses backend on wasm32-wasip2 in this release. \
Remove the non-openai HTTP provider features, or build for a native target."
);

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

/// Shared configuration constants.
pub mod config;

/// Centralized HTTP response handling for provider backends.
#[cfg(any(
    not(target_arch = "wasm32"),
    all(
        target_arch = "wasm32",
        target_os = "wasi",
        target_env = "p2",
        feature = "wasi-http"
    )
))]
pub mod http;

/// Evaluator for LLM providers
pub mod evaluator;

/// Secret store for storing API keys and other sensitive information
#[cfg(not(target_arch = "wasm32"))]
pub mod secret_store;

/// Listing models support
pub mod models;

mod protocol;
pub mod providers;

/// Composable optimization pipeline for LLM providers.
pub mod pipeline;

/// Built-in optimization passes (cache, etc.). Not available on WASM.
#[cfg(all(not(target_arch = "wasm32"), feature = "optim"))]
pub mod optim;

/// Direct WASI Preview2 (`wasm32-wasip2`) HTTP transport used by the OpenAI
/// Responses backend when the `wasi-http` feature is enabled.
#[cfg(all(
    target_arch = "wasm32",
    target_os = "wasi",
    target_env = "p2",
    feature = "wasi-http"
))]
mod wasi_http;

//Re-export for convenience
pub use async_trait::async_trait;
pub use chat::SamplingOverrides;

/// Unit config for providers with no provider-specific options.
#[derive(Debug, Default, Clone)]
pub struct NoConfig;

/// Provides an associated configuration type for LLM provider builders.
///
/// Implement this alongside [`LLMProvider`] to expose provider-specific builder
/// options. Use [`NoConfig`] for providers with no special options.
pub trait HasConfig {
    /// Provider-specific configuration type.
    type Config: Default + Send + Sync + 'static;
}

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

impl Display for ToolCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ToolCall {{ id: {}, type: {}, function: {:?} }}",
            self.id, self.call_type, self.function
        )
    }
}

/// FunctionCall contains details about which function to call and with what arguments.
#[derive(Debug, Deserialize, Serialize, Clone, Eq, PartialEq)]
pub struct FunctionCall {
    /// The name of the function to call.
    pub name: String,
    /// The arguments to pass to the function, typically serialized as a JSON string.
    pub arguments: String,
}

/// Default value for call_type field in ToolCall
pub fn default_call_type() -> String {
    "function".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HasConfig;
    use crate::chat::{ChatMessage, ChatProvider, ChatResponse, StructuredOutputFormat, Tool};
    use crate::completion::CompletionProvider;
    use crate::embedding::EmbeddingProvider;
    use crate::error::LLMError;
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
            id: String::default(),
            call_type: String::default(),
            function: FunctionCall {
                name: String::default(),
                arguments: String::default(),
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
        async fn chat(
            &self,
            _messages: &[ChatMessage],
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            Ok(Box::new(MockChatResponse {
                text: Some("Mock response".into()),
            }))
        }

        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            Ok(Box::new(MockChatResponse {
                text: Some("Mock response".into()),
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

    impl LLMProvider for MockLLMProvider {}

    impl HasConfig for MockLLMProvider {
        type Config = NoConfig;
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
    async fn test_chat_and_sampling_default_impl_ignores_sampling() {
        // Default trait impl delegates to chat_with_tools, dropping sampling.
        // Backends without per-call sampling support must not error when
        // overrides are passed — `Some(SamplingOverrides::...)` is silently
        // ignored. Asserts the contract documented on
        // `chat_with_tools_and_sampling`.
        let provider = MockLLMProvider;
        let messages = vec![chat::ChatMessage::user().content("Test").build()];

        // None override path: identical to chat_with_tools.
        let baseline = provider.chat(&messages, None).await.unwrap();
        let none_override = provider
            .chat_and_sampling(&messages, None, None)
            .await
            .unwrap();
        assert_eq!(baseline.text(), none_override.text());

        // Some override path: also succeeds (mock ignores it).
        let with_overrides = provider
            .chat_and_sampling(
                &messages,
                None,
                Some(&chat::SamplingOverrides {
                    temperature: Some(0.0),
                    top_p: Some(0.9),
                    max_tokens: Some(64),
                }),
            )
            .await
            .unwrap();
        assert_eq!(with_overrides.text(), Some("Mock response".to_string()));
    }

    #[tokio::test]
    async fn test_chat_with_tools_and_sampling_default_impl_ignores_sampling() {
        let provider = MockLLMProvider;
        let messages = vec![chat::ChatMessage::user().content("Test").build()];

        let response = provider
            .chat_with_tools_and_sampling(
                &messages,
                None,
                None,
                Some(&chat::SamplingOverrides::with_temperature(0.0)),
            )
            .await
            .unwrap();
        assert_eq!(response.text(), Some("Mock response".to_string()));
    }

    #[test]
    fn test_sampling_overrides_helpers() {
        let empty = chat::SamplingOverrides::empty();
        assert_eq!(empty, chat::SamplingOverrides::default());
        assert!(empty.temperature.is_none());
        assert!(empty.top_p.is_none());
        assert!(empty.max_tokens.is_none());

        let temp = chat::SamplingOverrides::with_temperature(0.0);
        assert_eq!(temp.temperature, Some(0.0));
        assert_eq!(temp.top_p, None);

        let top_p = chat::SamplingOverrides::with_top_p(0.95);
        assert_eq!(top_p.top_p, Some(0.95));
        assert_eq!(top_p.temperature, None);

        let max_tok = chat::SamplingOverrides::with_max_tokens(128);
        assert_eq!(max_tok.max_tokens, Some(128));

        // SamplingOverrides is re-exported at crate root.
        let from_root = SamplingOverrides::with_temperature(0.5);
        assert_eq!(from_root.temperature, Some(0.5));
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
}
