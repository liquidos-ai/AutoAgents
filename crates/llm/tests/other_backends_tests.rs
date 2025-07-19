#![allow(unused_imports)]
use autoagents_llm::{
    builder::LLMBuilder,
    chat::{ChatMessage, ChatProvider, ChatRole, StructuredOutputFormat},
    completion::{CompletionProvider, CompletionRequest},
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
        let client = DeepSeek::new("test-key", None, None, None, None, None, None);

        assert_eq!(client.model, "deepseek-chat");
        assert!(client.max_tokens.is_none());
        assert!(client.temperature.is_none());
        assert!(client.system.is_none());
        assert!(client.stream.is_none());
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
            None, None, None, None, None,
        );

        assert_eq!(client.model, "grok-2-latest");
        assert!(client.max_tokens.is_none());
        assert!(client.temperature.is_none());
        assert!(client.system.is_none());
        assert!(client.stream.is_none());
    }

    #[test]
    fn test_xai_search_configuration() {
        let mut client = XAI::new(
            "test-key", None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None,
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
            None, None, None,
        );

        let messages = vec![ChatMessage::user().content("Hello").build()];

        let result = client.chat(&messages, None).await;
        assert!(result.is_err());

        match result.err().unwrap() {
            LLMError::AuthError(msg) => {
                assert_eq!(msg, "Missing X.AI API key");
            }
            _ => panic!("Expected AuthError"),
        }
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
        assert_eq!(
            client.api_base_url,
            "https://https.extension.phind.com/agent/"
        );
    }

    #[test]
    fn test_phind_default_values() {
        let client = Phind::new(None, None, None, None, None, None, None, None);

        assert_eq!(client.model, "Phind-70B");
        assert!(client.max_tokens.is_none());
        assert!(client.temperature.is_none());
        assert!(client.system.is_none());
        assert!(client.stream.is_none());
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
        let client = Groq::new("test-key", None, None, None, None, None, None, None, None);

        assert_eq!(
            String::from(client.model.clone()),
            "llama-3.3-70b-versatile"
        );
        assert!(client.max_tokens.is_none());
        assert!(client.temperature.is_none());
        assert!(client.system.is_none());
        assert!(client.stream.is_none());
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
        let client = Groq::new("", None, None, None, None, None, None, None, None);

        let messages = vec![ChatMessage::user().content("Hello").build()];

        let result = client.chat(&messages, None).await;
        assert!(result.is_err());

        match result.err().unwrap() {
            LLMError::AuthError(msg) => {
                assert_eq!(msg, "Missing Groq API key");
            }
            _ => panic!("Expected AuthError"),
        }
    }
}
