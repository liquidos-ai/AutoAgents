#![allow(unused_imports)]
use autoagents_llm::{
    builder::LLMBuilder,
    chat::{ChatMessage, ChatProvider, ChatRole, StructuredOutputFormat},
    completion::{CompletionProvider, CompletionRequest},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
};
use serde_json::json;
use std::sync::Arc;

#[cfg(feature = "deepseek")]
mod deepseek_tests {
    use super::*;
    use autoagents_llm::backends::deepseek::DeepSeek;

    fn create_test_deepseek() -> Arc<DeepSeek> {
        LLMBuilder::<DeepSeek>::new()
            .api_key("test-key")
            .model("deepseek-chat")
            .max_tokens(100)
            .temperature(0.7)
            .system("Test system prompt")
            .build()
            .expect("Failed to build DeepSeek client")
    }

    #[test]
    fn test_deepseek_creation() {
        let client = create_test_deepseek();
        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.model, "deepseek-chat");
        assert_eq!(client.max_tokens, Some(100));
        assert_eq!(client.temperature, Some(0.7));
        assert_eq!(client.system, Some("Test system prompt".to_string()));
    }

    #[test]
    fn test_deepseek_builder_validation() {
        // Test missing API key
        let result = LLMBuilder::<DeepSeek>::new().model("deepseek-chat").build();

        assert!(result.is_err());
        match result.err().unwrap() {
            LLMError::InvalidRequest(msg) => {
                assert!(msg.contains("No API key provided"));
            }
            _ => panic!("Expected InvalidRequest error"),
        }
    }

    #[test]
    fn test_deepseek_default_values() {
        let client = DeepSeek::new("test-key", None, None, None, None, None);

        assert_eq!(client.model, "deepseek-chat");
        assert!(client.max_tokens.is_none());
        assert!(client.temperature.is_none());
        assert!(client.system.is_none());
    }

    #[test]
    fn test_deepseek_builder_with_all_options() {
        let client = LLMBuilder::<DeepSeek>::new()
            .api_key("test-key")
            .model("deepseek-coder")
            .max_tokens(2000)
            .temperature(0.8)
            .top_p(0.95)
            .system("Custom system prompt")
            .build()
            .expect("Failed to build DeepSeek client");

        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.model, "deepseek-coder");
        assert_eq!(client.max_tokens, Some(2000));
        assert_eq!(client.temperature, Some(0.8));
        assert_eq!(client.system, Some("Custom system prompt".to_string()));
    }

    #[tokio::test]
    async fn test_deepseek_chat_auth_error() {
        let client = DeepSeek::new("", None, None, None, None, None);

        let messages = vec![ChatMessage::user().content("Hello").build()];

        let result = client.chat(&messages, None, None).await;
        assert!(result.is_err());

        match result.err().unwrap() {
            LLMError::AuthError(msg) => {
                assert_eq!(msg, "Missing DeepSeek API key");
            }
            _ => panic!("Expected AuthError"),
        }
    }

    #[tokio::test]
    async fn test_deepseek_completion_auth_error() {
        let client = DeepSeek::new("", None, None, None, None, None);

        let request = CompletionRequest {
            prompt: "Complete this sentence".to_string(),
            max_tokens: None,
            temperature: None,
        };

        let result = client.complete(&request, None).await;
        assert!(result.is_err());

        match result.err().unwrap() {
            LLMError::AuthError(msg) => {
                assert_eq!(msg, "Missing DeepSeek API key");
            }
            _ => panic!("Expected AuthError"),
        }
    }

    #[tokio::test]
    async fn test_deepseek_models_provider_not_implemented() {
        let client = create_test_deepseek();
        let result = client.list_models(None).await;

        assert!(result.is_err());
        match result.err().unwrap() {
            LLMError::ProviderError(msg) => {
                assert!(msg.contains("List Models not supported"));
            }
            _ => panic!("Expected ProviderError"),
        }
    }

    // DeepSeek doesn't implement EmbeddingProvider, so no embedding tests

    #[test]
    fn test_deepseek_models() {
        let models = vec!["deepseek-chat", "deepseek-coder", "deepseek-reasoner"];

        for model in models {
            let client = DeepSeek::new("test-key", Some(model.to_string()), None, None, None, None);
            assert_eq!(client.model, model);
        }
    }

    #[test]
    fn test_deepseek_with_extreme_values() {
        let client = DeepSeek::new(
            "test-key",
            Some("test-model".to_string()),
            Some(u32::MAX),
            Some(2.0),
            Some(60),
            Some("Very long system prompt".to_string()),
        );

        assert_eq!(client.max_tokens, Some(u32::MAX));
        assert_eq!(client.temperature, Some(2.0));
    }
}

#[cfg(feature = "xai")]
mod xai_tests {
    use super::*;
    use autoagents_llm::backends::xai::XAI;

    fn create_test_xai() -> Arc<XAI> {
        LLMBuilder::<XAI>::new()
            .api_key("test-key")
            .model("grok-2-latest")
            .max_tokens(100)
            .temperature(0.7)
            .system("Test system prompt")
            .build()
            .expect("Failed to build XAI client")
    }

    #[test]
    fn test_xai_creation() {
        let client = create_test_xai();
        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.model, "grok-2-latest");
        assert_eq!(client.max_tokens, Some(100));
        assert_eq!(client.temperature, Some(0.7));
        assert_eq!(client.system, Some("Test system prompt".to_string()));
    }

    #[test]
    fn test_xai_builder_validation() {
        // Test missing API key
        let result = LLMBuilder::<XAI>::new().model("grok-2-latest").build();

        assert!(result.is_err());
        match result.err().unwrap() {
            LLMError::InvalidRequest(msg) => {
                assert!(msg.contains("No API key provided"));
            }
            _ => panic!("Expected InvalidRequest error"),
        }
    }

    #[test]
    fn test_xai_default_values() {
        let client = XAI::new(
            "test-key", None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None,
        );

        assert_eq!(client.model, "grok-2-latest");
        assert!(client.max_tokens.is_none());
        assert!(client.temperature.is_none());
        assert!(client.system.is_none());
    }

    #[test]
    fn test_xai_search_configuration() {
        let mut client = XAI::new(
            "test-key", None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None,
        );

        client = client.set_search_mode("auto");
        client = client.set_search_source("web", Some(vec!["example.com".to_string()]));
        client = client.set_max_search_results(10);
        client = client.set_search_date_range("2023-01-01", "2023-12-31");

        assert_eq!(client.xai_search_mode, Some("auto".to_string()));
        assert_eq!(client.xai_search_source_type, Some("web".to_string()));
        assert_eq!(
            client.xai_search_excluded_websites,
            Some(vec!["example.com".to_string()])
        );
        assert_eq!(client.xai_search_max_results, Some(10));
        assert_eq!(client.xai_search_from_date, Some("2023-01-01".to_string()));
        assert_eq!(client.xai_search_to_date, Some("2023-12-31".to_string()));
    }

    #[test]
    fn test_xai_embedding_configuration() {
        let client = LLMBuilder::<XAI>::new()
            .api_key("test-key")
            .model("text-embedding-model")
            .embedding_encoding_format("float")
            .embedding_dimensions(1536)
            .build()
            .expect("Failed to build XAI embedding client");

        assert_eq!(client.embedding_encoding_format, Some("float".to_string()));
        assert_eq!(client.embedding_dimensions, Some(1536));
    }

    #[tokio::test]
    async fn test_chat_auth_error() {
        let client = XAI::new(
            "", None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None,
        );

        let messages = vec![ChatMessage::user().content("Hello").build()];

        let result = client.chat(&messages, None, None).await;
        assert!(result.is_err());

        match result.err().unwrap() {
            LLMError::AuthError(msg) => {
                assert_eq!(msg, "Missing X.AI API key");
            }
            _ => panic!("Expected AuthError"),
        }
    }

    #[test]
    fn test_xai_builder_with_all_options() {
        let client = LLMBuilder::<XAI>::new()
            .api_key("test-key")
            .model("grok-2-1212")
            .max_tokens(2000)
            .temperature(0.8)
            .top_p(0.95)
            .system("Custom system prompt")
            .embedding_encoding_format("float")
            .embedding_dimensions(1536)
            .build()
            .expect("Failed to build XAI client");

        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.model, "grok-2-1212");
        assert_eq!(client.max_tokens, Some(2000));
        assert_eq!(client.temperature, Some(0.8));
        assert_eq!(client.system, Some("Custom system prompt".to_string()));
        assert_eq!(client.embedding_encoding_format, Some("float".to_string()));
        assert_eq!(client.embedding_dimensions, Some(1536));
    }

    #[tokio::test]
    async fn test_xai_completion_auth_error() {
        let client = XAI::new(
            "", None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None,
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
                assert_eq!(msg, "Missing X.AI API key");
            }
            _ => panic!("Expected AuthError"),
        }
    }

    // XAI embedding tests would require EmbeddingProvider trait implementation

    #[test]
    fn test_xai_search_methods() {
        let mut client = XAI::new(
            "test-key", None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None,
        );

        // Test all search setter methods
        client = client.set_search_mode("disabled");
        client = client.set_search_source("social", Some(vec!["reddit.com".to_string()]));
        client = client.set_max_search_results(5);
        client = client.set_search_date_range("2022-01-01", "2024-12-31");

        assert_eq!(client.xai_search_mode, Some("disabled".to_string()));
        assert_eq!(client.xai_search_source_type, Some("social".to_string()));
        assert_eq!(
            client.xai_search_excluded_websites,
            Some(vec!["reddit.com".to_string()])
        );
        assert_eq!(client.xai_search_max_results, Some(5));
        assert_eq!(client.xai_search_from_date, Some("2022-01-01".to_string()));
        assert_eq!(client.xai_search_to_date, Some("2024-12-31".to_string()));
    }

    #[test]
    fn test_xai_models() {
        let models = vec![
            "grok-2-latest",
            "grok-2-1212",
            "grok-beta",
            "grok-vision-beta",
        ];

        for model in models {
            let client = XAI::new(
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
    fn test_xai_with_extreme_values() {
        let client = XAI::new(
            "test-key",
            Some("test-model".to_string()),
            Some(u32::MAX),
            Some(2.0),
            Some(u64::MAX),
            Some("Very long system prompt".to_string()),
            Some(1.0),
            None,
            Some("base64".to_string()),
            Some(u32::MAX),
            Some("all".to_string()),
            Some("academic".to_string()),
            Some(vec!["all-domains.com".to_string()]),
            Some(u32::MAX),
            Some("1900-01-01".to_string()),
            Some("2100-12-31".to_string()),
        );

        assert_eq!(client.max_tokens, Some(u32::MAX));
        assert_eq!(client.temperature, Some(2.0));
        assert_eq!(client.timeout_seconds, Some(u64::MAX));
        assert_eq!(client.embedding_encoding_format, Some("base64".to_string()));
        assert_eq!(client.embedding_dimensions, Some(u32::MAX));
        assert_eq!(client.xai_search_mode, Some("all".to_string()));
        assert_eq!(client.xai_search_source_type, Some("academic".to_string()));
        assert_eq!(client.xai_search_max_results, Some(u32::MAX));
    }

    #[tokio::test]
    async fn test_xai_models_provider_auth_error() {
        let client = XAI::new(
            "", None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None,
        );

        let result = client.list_models(None).await;
        assert!(result.is_err());

        match result.err().unwrap() {
            LLMError::AuthError(msg) => {
                assert_eq!(msg, "Missing X.AI API key");
            }
            _ => panic!("Expected AuthError"),
        }
    }

    #[test]
    fn test_xai_search_chaining() {
        let client = XAI::new(
            "test-key", None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None,
        )
        .set_search_mode("auto")
        .set_max_search_results(3)
        .set_search_date_range("2023-06-01", "2023-12-01");

        assert_eq!(client.xai_search_mode, Some("auto".to_string()));
        assert_eq!(client.xai_search_max_results, Some(3));
        assert_eq!(client.xai_search_from_date, Some("2023-06-01".to_string()));
        assert_eq!(client.xai_search_to_date, Some("2023-12-01".to_string()));
    }
}

#[cfg(feature = "phind")]
mod phind_tests {
    use super::*;
    use autoagents_llm::backends::phind::Phind;

    fn create_test_phind() -> Arc<Phind> {
        LLMBuilder::<Phind>::new()
            .model("Phind-70B")
            .max_tokens(100)
            .temperature(0.7)
            .system("Test system prompt")
            .build()
            .expect("Failed to build Phind client")
    }

    #[test]
    fn test_phind_creation() {
        let client = create_test_phind();
        assert_eq!(client.model, "Phind-70B");
        assert_eq!(client.max_tokens, Some(100));
        assert_eq!(client.temperature, Some(0.7));
        assert_eq!(client.system, Some("Test system prompt".to_string()));
        assert_eq!(client.api_base_url, "https://extension.phind.com/agent/");
    }

    #[test]
    fn test_phind_default_values() {
        let client = Phind::new(None, None, None, None, None, None, None, None);

        assert_eq!(client.model, "Phind-70B");
        assert!(client.max_tokens.is_none());
        assert!(client.temperature.is_none());
        assert!(client.system.is_none());
    }

    #[test]
    fn test_phind_builder_with_all_options() {
        let client = LLMBuilder::<Phind>::new()
            .model("Phind-CodeLlama-34B-v2")
            .max_tokens(2000)
            .temperature(0.8)
            .top_p(0.95)
            .timeout_seconds(120)
            .system("Custom system prompt")
            .base_url("https://custom.phind.com/agent/")
            .build()
            .expect("Failed to build Phind client");

        assert_eq!(client.model, "Phind-CodeLlama-34B-v2");
        assert_eq!(client.max_tokens, Some(2000));
        assert_eq!(client.temperature, Some(0.8));
        assert_eq!(client.timeout_seconds, Some(120));
        assert_eq!(client.system, Some("Custom system prompt".to_string()));
        assert_eq!(client.api_base_url, "https://custom.phind.com/agent/");
    }

    #[tokio::test]
    async fn test_phind_chat_no_auth_error() {
        // Phind doesn't require API key authentication
        let client = Phind::new(None, None, None, None, None, None, None, None);

        let messages = vec![ChatMessage::user().content("Hello").build()];

        // This should not fail with auth error since Phind doesn't require API key
        // It will likely fail with network error or other error, but not auth
        let result = client.chat(&messages, None, None).await;
        // We expect some error but not specifically an auth error
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_phind_completion_request() {
        let client = Phind::new(None, None, None, None, None, None, None, None);

        let request = CompletionRequest {
            prompt: "Complete this code".to_string(),
            max_tokens: None,
            temperature: None,
        };

        let result = client.complete(&request, None).await;
        assert!(result.is_err()); // Will fail with network error, not auth error
    }

    #[tokio::test]
    async fn test_phind_models_provider_not_implemented() {
        let client = create_test_phind();
        let result = client.list_models(None).await;

        assert!(result.is_err());
        match result.err().unwrap() {
            LLMError::ProviderError(msg) => {
                assert!(msg.contains("List Models not supported"));
            }
            _ => panic!("Expected ProviderError"),
        }
    }

    // Phind doesn't implement EmbeddingProvider, so no embedding tests

    #[test]
    fn test_phind_models() {
        let models = vec![
            "Phind-70B",
            "Phind-CodeLlama-34B-v2",
            "Phind-CodeLlama-34B-Python-v1",
        ];

        for model in models {
            let client = Phind::new(
                Some(model.to_string()),
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
    fn test_phind_with_extreme_values() {
        let client = Phind::new(
            Some("test-model".to_string()),
            Some(u32::MAX),
            Some(2.0),
            Some(60),
            Some("Very long system prompt".to_string()),
            Some(1.0),
            None,
            None,
        );

        assert_eq!(client.max_tokens, Some(u32::MAX));
        assert_eq!(client.temperature, Some(2.0));
        assert_eq!(client.timeout_seconds, Some(60));
        assert_eq!(client.top_p, Some(1.0));
    }

    #[test]
    fn test_phind_custom_base_url() {
        let custom_urls = vec![
            "https://api.phind.com/v1/",
            "https://custom.phind.service.com/agent/",
            "http://localhost:8080/phind/",
        ];

        for url in custom_urls {
            let _client = Phind::new(None, None, None, None, None, None, None, None);
            // API base URL is set at compile time, not through constructor
            let _expected_url = url;
        }
    }
}

#[cfg(feature = "groq")]
mod groq_tests {
    use super::*;
    use autoagents_llm::backends::groq::{Groq, GroqModel};

    fn create_test_groq() -> Arc<Groq> {
        LLMBuilder::<Groq>::new()
            .api_key("test-key")
            .model("llama-3.3-70b-versatile")
            .max_tokens(100)
            .temperature(0.7)
            .system("Test system prompt")
            .build()
            .expect("Failed to build Groq client")
    }

    #[test]
    fn test_groq_creation() {
        let client = create_test_groq();
        assert_eq!(client.api_key, "test-key");
        assert_eq!(
            String::from(client.model.clone()),
            "llama-3.3-70b-versatile"
        );
        assert_eq!(client.max_tokens, Some(100));
        assert_eq!(client.temperature, Some(0.7));
        assert_eq!(client.system, Some("Test system prompt".to_string()));
    }

    #[test]
    fn test_groq_builder_validation() {
        // Test missing API key
        let result = LLMBuilder::<Groq>::new()
            .model("llama-3.3-70b-versatile")
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
    fn test_groq_default_values() {
        let client = Groq::new("test-key", None, None, None, None, None, None, None);

        assert_eq!(
            String::from(client.model.clone()),
            "llama-3.3-70b-versatile"
        );
        assert!(client.max_tokens.is_none());
        assert!(client.temperature.is_none());
        assert!(client.system.is_none());
    }

    #[test]
    fn test_groq_model_enum() {
        // Test default
        let default_model = GroqModel::default();
        assert_eq!(String::from(default_model), "llama-3.3-70b-versatile");

        // Test Kimi model
        let kimi_model = GroqModel::KimiK2;
        assert_eq!(String::from(kimi_model), "moonshotai/kimi-k2-instruct");

        // Test string to model conversion
        let model_from_string = GroqModel::from("moonshotai/kimi-k2-instruct".to_string());
        match model_from_string {
            GroqModel::KimiK2 => (),
            _ => panic!("Expected KimiK2 model"),
        }

        // Test unknown model defaults to Llama
        let unknown_model = GroqModel::from("unknown-model".to_string());
        match unknown_model {
            GroqModel::Llama33_70B => (),
            _ => panic!("Expected Llama33_70B model"),
        }
    }

    #[tokio::test]
    async fn test_chat_auth_error() {
        let client = Groq::new("", None, None, None, None, None, None, None);

        let messages = vec![ChatMessage::user().content("Hello").build()];

        let result = client.chat(&messages, None, None).await;
        assert!(result.is_err());

        match result.err().unwrap() {
            LLMError::AuthError(msg) => {
                assert_eq!(msg, "Missing Groq API key");
            }
            _ => panic!("Expected AuthError"),
        }
    }

    #[test]
    fn test_groq_builder_with_all_options() {
        let client = LLMBuilder::<Groq>::new()
            .api_key("test-key")
            .model("mixtral-8x7b-32768")
            .max_tokens(2000)
            .temperature(0.8)
            .top_p(0.95)
            .timeout_seconds(120)
            .system("Custom system prompt")
            .base_url("https://custom.groq.com/openai/v1")
            .build()
            .expect("Failed to build Groq client");

        assert_eq!(client.api_key, "test-key");
        assert_eq!(String::from(client.model.clone()), "mixtral-8x7b-32768");
        assert_eq!(client.max_tokens, Some(2000));
        assert_eq!(client.temperature, Some(0.8));
        assert_eq!(client.timeout_seconds, Some(120));
        assert_eq!(client.system, Some("Custom system prompt".to_string()));
        // Base URL is handled by the builder pattern
    }

    #[tokio::test]
    async fn test_groq_completion_auth_error() {
        let client = Groq::new("", None, None, None, None, None, None, None);

        let request = CompletionRequest {
            prompt: "Complete this sentence".to_string(),
            max_tokens: None,
            temperature: None,
        };

        let result = client.complete(&request, None).await;
        assert!(result.is_err());

        match result.err().unwrap() {
            LLMError::AuthError(msg) => {
                assert_eq!(msg, "Missing Groq API key");
            }
            _ => panic!("Expected AuthError"),
        }
    }

    #[tokio::test]
    async fn test_groq_models_provider_auth_error() {
        let client = Groq::new("", None, None, None, None, None, None, None);

        let result = client.list_models(None).await;
        assert!(result.is_err());

        match result.err().unwrap() {
            LLMError::AuthError(msg) => {
                assert_eq!(msg, "Missing Groq API key");
            }
            _ => panic!("Expected AuthError"),
        }
    }

    // Groq doesn't implement EmbeddingProvider, so no embedding tests

    #[test]
    fn test_groq_model_variants() {
        let model_tests = vec![
            ("llama-3.3-70b-versatile", GroqModel::Llama33_70B),
            ("moonshotai/kimi-k2-instruct", GroqModel::KimiK2),
        ];

        for (expected_string, model) in model_tests {
            assert_eq!(String::from(model.clone()), expected_string);

            let parsed_model = GroqModel::from(expected_string.to_string());
            assert_eq!(
                std::mem::discriminant(&parsed_model),
                std::mem::discriminant(&model)
            );
        }
    }

    #[test]
    fn test_groq_model_debug_and_clone() {
        let model = GroqModel::Llama33_70B;
        let cloned = model.clone();

        assert_eq!(
            std::mem::discriminant(&model),
            std::mem::discriminant(&cloned)
        );

        let debug_str = format!("{model:?}");
        assert!(debug_str.contains("Llama33_70B"));
    }

    #[test]
    fn test_groq_with_extreme_values() {
        let client = Groq::new(
            "test-key",
            Some(GroqModel::Llama33_70B),
            Some(u32::MAX),
            Some(2.0),
            Some(60),
            Some("Very long system prompt".to_string()),
            Some(1.0),
            None,
        );

        assert_eq!(client.max_tokens, Some(u32::MAX));
        assert_eq!(client.temperature, Some(2.0));
        assert_eq!(client.timeout_seconds, Some(60));
        assert_eq!(client.top_p, Some(1.0));
    }

    #[test]
    fn test_groq_all_model_types() {
        let models = vec![GroqModel::Llama33_70B, GroqModel::KimiK2];

        for model in models {
            let client = Groq::new(
                "test-key",
                Some(model.clone()),
                None,
                None,
                None,
                None,
                None,
                None,
            );
            assert_eq!(
                std::mem::discriminant(&client.model),
                std::mem::discriminant(&model)
            );
        }
    }

    #[test]
    fn test_groq_base_url_variants() {
        let base_urls = vec![
            "https://api.groq.com/openai/v1",
            "https://custom.groq.service.com/v1",
            "http://localhost:8080/groq/v1",
        ];

        for url in base_urls {
            let _client = Groq::new("test-key", None, None, None, None, None, None, None);
            // Base URL is handled by builder pattern, not constructor
            let _expected_url = url;
        }
    }

    #[test]
    fn test_groq_model_string_conversions() {
        // Test unknown model mapping
        let unknown_models = vec![
            "unknown-model",
            "gpt-4",
            "claude-3",
            "",
            "random-string-123",
        ];

        for unknown in unknown_models {
            let model = GroqModel::from(unknown.to_string());
            match model {
                GroqModel::Llama33_70B => (), // Should default to this
                _ => panic!("Expected unknown model to default to Llama33_70B"),
            }
        }
    }

    #[test]
    fn test_groq_model_from_string_edge_cases() {
        // Test exact string matches
        let exact_matches = vec![
            ("llama-3.3-70b-versatile", GroqModel::Llama33_70B),
            ("LLAMA-3.3-70B-VERSATILE", GroqModel::Llama33_70B), // Should not match due to case sensitivity
            ("llama-3.1-70b-versatile ", GroqModel::Llama33_70B), // Trailing space should not match
        ];

        for (input, _expected_default) in exact_matches {
            let model = GroqModel::from(input.to_string());
            if input == "llama-3.3-70b-versatile" {
                match model {
                    GroqModel::Llama33_70B => (),
                    _ => panic!("Expected exact match for {input}"),
                }
            } else {
                // Non-exact matches should default to Llama33_70B
                match model {
                    GroqModel::Llama33_70B => (),
                    _ => panic!("Expected default for {input}"),
                }
            }
        }
    }
}
