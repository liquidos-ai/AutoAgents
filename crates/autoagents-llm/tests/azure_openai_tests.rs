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

#[cfg(feature = "azure_openai")]
mod azure_openai_test_cases {
    use super::*;
    use autoagents_llm::{backends::azure_openai::AzureOpenAI, chat::ReasoningEffort};

    fn create_test_azure_openai() -> Arc<AzureOpenAI> {
        LLMBuilder::<AzureOpenAI>::new()
            .api_key("test-key")
            .api_version("2023-05-15")
            .deployment_id("gpt-4-deployment")
            .base_url("https://myresource.openai.azure.com")
            .model("gpt-4")
            .max_tokens(100)
            .temperature(0.7)
            .build()
            .expect("Failed to build Azure OpenAI client")
    }

    #[test]
    fn test_azure_openai_creation() {
        let client = create_test_azure_openai();
        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.api_version, "2023-05-15");
        assert_eq!(client.model, "gpt-4");
        assert_eq!(client.max_tokens, Some(100));
        assert_eq!(client.temperature, Some(0.7));
        assert_eq!(
            client.base_url.as_str(),
            "https://myresource.openai.azure.com/openai/deployments/gpt-4-deployment/"
        );
    }

    #[test]
    fn test_azure_openai_builder_validation() {
        // Test missing API key
        let result = LLMBuilder::<AzureOpenAI>::new()
            .api_version("2023-05-15")
            .deployment_id("gpt-4-deployment")
            .base_url("https://myresource.openai.azure.com/")
            .build();

        assert!(result.is_err());
        match result.err().unwrap() {
            LLMError::InvalidRequest(msg) => {
                assert!(msg.contains("No API key provided"));
            }
            _ => panic!("Expected InvalidRequest error"),
        }

        // Test missing API version
        let result = LLMBuilder::<AzureOpenAI>::new()
            .api_key("test-key")
            .deployment_id("gpt-4-deployment")
            .base_url("https://myresource.openai.azure.com/")
            .build();

        assert!(result.is_err());
        match result.err().unwrap() {
            LLMError::InvalidRequest(msg) => {
                assert!(msg.contains("No API version provided"));
            }
            _ => panic!("Expected InvalidRequest error"),
        }

        // Test missing deployment ID
        let result = LLMBuilder::<AzureOpenAI>::new()
            .api_key("test-key")
            .api_version("2023-05-15")
            .base_url("https://myresource.openai.azure.com/")
            .build();

        assert!(result.is_err());
        match result.err().unwrap() {
            LLMError::InvalidRequest(msg) => {
                assert!(msg.contains("No deployment ID provided"));
            }
            _ => panic!("Expected InvalidRequest error"),
        }

        // Test missing base URL
        let result = LLMBuilder::<AzureOpenAI>::new()
            .api_key("test-key")
            .api_version("2023-05-15")
            .deployment_id("gpt-4-deployment")
            .build();

        assert!(result.is_err());
        match result.err().unwrap() {
            LLMError::InvalidRequest(msg) => {
                assert!(msg.contains("No API endpoint provided"));
            }
            _ => panic!("Expected InvalidRequest error"),
        }
    }

    #[test]
    fn test_azure_openai_default_values() {
        let client = AzureOpenAI::new(
            "test-key",
            "2023-05-15",
            "gpt-4-deployment",
            "https://myresource.openai.azure.com/",
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

        assert_eq!(client.model, "gpt-3.5-turbo");
        assert!(client.max_tokens.is_none());
        assert!(client.temperature.is_none());
    }

    #[test]
    fn test_embedding_configuration() {
        let client = LLMBuilder::<AzureOpenAI>::new()
            .api_key("test-key")
            .api_version("2023-05-15")
            .deployment_id("text-embedding-ada-002")
            .base_url("https://myresource.openai.azure.com/")
            .model("text-embedding-ada-002")
            .embedding_encoding_format("float")
            .embedding_dimensions(1536)
            .build()
            .expect("Failed to build Azure OpenAI embedding client");

        assert_eq!(client.embedding_encoding_format, Some("float".to_string()));
        assert_eq!(client.embedding_dimensions, Some(1536));
    }

    #[test]
    fn test_reasoning_configuration() {
        let client = LLMBuilder::<AzureOpenAI>::new()
            .api_key("test-key")
            .api_version("2023-05-15")
            .deployment_id("gpt-4-deployment")
            .base_url("https://myresource.openai.azure.com/")
            .reasoning_effort(ReasoningEffort::High)
            .build()
            .expect("Failed to build Azure OpenAI client with reasoning");

        assert_eq!(
            client.reasoning_effort,
            Some(ReasoningEffort::High.to_string())
        );
    }

    #[tokio::test]
    async fn test_chat_auth_error() {
        let client = AzureOpenAI::new(
            "", // Empty API key
            "2023-05-15",
            "gpt-4-deployment",
            "https://myresource.openai.azure.com/",
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

        let messages = vec![ChatMessage::user().content("Hello").build()];

        let result = client.chat(&messages, None).await;
        assert!(result.is_err());

        match result.err().unwrap() {
            LLMError::AuthError(msg) => {
                assert_eq!(msg, "Missing Azure OpenAI API key");
            }
            _ => panic!("Expected AuthError"),
        }
    }

    #[test]
    fn test_timeout_configuration() {
        let client = LLMBuilder::<AzureOpenAI>::new()
            .api_key("test-key")
            .api_version("2023-05-15")
            .deployment_id("gpt-4-deployment")
            .base_url("https://myresource.openai.azure.com")
            .timeout_seconds(60)
            .build()
            .expect("Failed to build Azure OpenAI client with timeout");

        assert_eq!(client.timeout_seconds, Some(60));
    }

    #[test]
    fn test_url_construction() {
        let client = AzureOpenAI::new(
            "test-key",
            "2023-05-15",
            "my-deployment",
            "https://myresource.openai.azure.com",
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

        assert_eq!(
            client.base_url.as_str(),
            "https://myresource.openai.azure.com/openai/deployments/my-deployment/"
        );
    }
}
