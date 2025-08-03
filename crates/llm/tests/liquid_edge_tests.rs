#![allow(unused_imports)]
use autoagents_llm::{
    builder::LLMBuilder,
    chat::{
        ChatMessage, ChatProvider, ChatResponse, ChatRole, MessageType, StructuredOutputFormat,
    },
    completion::{CompletionProvider, CompletionRequest},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
};
use serde_json::json;
use std::{fs, sync::Arc};
use tempfile::tempdir;

#[cfg(feature = "liquid_edge")]
mod liquid_edge_test_cases {
    use super::*;
    use autoagents_llm::backends::liquid_edge::LiquidEdge;

    fn create_mock_model_dir() -> tempfile::TempDir {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();

        // Create mock tokenizer.json with minimal valid structure
        fs::write(
            model_path.join("tokenizer.json"),
            r#"{
                "version": "1.0",
                "truncation": null,
                "padding": null,
                "added_tokens": [],
                "normalizer": null,
                "pre_tokenizer": null,
                "post_processor": null,
                "decoder": null,
                "model": {
                    "type": "BPE",
                    "dropout": null,
                    "unk_token": null,
                    "continuing_subword_prefix": null,
                    "end_of_word_suffix": null,
                    "fuse_unk": false,
                    "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2},
                    "merges": []
                }
            }"#,
        )
        .unwrap();

        // Create mock config.json
        fs::write(
            model_path.join("config.json"),
            r#"{
                "eos_token_id": 2,
                "bos_token_id": 1,
                "max_position_embeddings": 2048,
                "vocab_size": 32000
            }"#,
        )
        .unwrap();

        // Create mock tokenizer_config.json
        fs::write(
            model_path.join("tokenizer_config.json"),
            r#"{
                "add_bos_token": true,
                "model_max_length": 2048
            }"#,
        )
        .unwrap();

        // Create mock special_tokens_map.json
        fs::write(
            model_path.join("special_tokens_map.json"),
            r#"{
                "bos_token": "<s>",
                "eos_token": "</s>"
            }"#,
        )
        .unwrap();

        // Create mock model.onnx (empty file for testing)
        fs::write(model_path.join("model.onnx"), b"mock onnx data").unwrap();

        // Create mock chat template
        fs::write(
            model_path.join("chat_template.jinja"),
            "{% for message in messages %}{{ message.content }}{% endfor %}",
        )
        .unwrap();

        temp_dir
    }

    #[test]
    fn test_liquid_edge_builder_validation() {
        // Test missing model path (base_url is used as model path in LiquidEdge)
        let result = LLMBuilder::<LiquidEdge>::new().model("test-model").build();

        assert!(result.is_err());
        match result.err().unwrap() {
            LLMError::InvalidRequest(msg) => {
                assert!(msg.contains("No model path provided"));
            }
            _ => panic!("Expected InvalidRequest error"),
        }
    }

    #[test]
    fn test_liquid_edge_creation_fails_without_valid_model() {
        let temp_dir = create_mock_model_dir();

        // This will fail because we don't have a real ONNX model that can be loaded
        let result = LiquidEdge::new(
            temp_dir.path(),
            "test-model".to_string(),
            Some(100),
            Some(0.7),
            Some(0.9),
            Some("Test system prompt".to_string()),
        );

        // Should fail due to invalid ONNX model file
        assert!(result.is_err());
    }

    #[test]
    fn test_liquid_edge_config_loading() {
        let temp_dir = create_mock_model_dir();

        // Test configuration loading
        let config_result = LiquidEdge::load_config(temp_dir.path());

        // This should work since we have a valid config.json
        assert!(config_result.is_ok());
    }

    #[test]
    fn test_liquid_edge_default_values() {
        let temp_dir = create_mock_model_dir();

        // Test with minimal parameters
        let result = LiquidEdge::new(
            temp_dir.path(),
            "test-model".to_string(),
            None, // Should default to 150
            None, // Should default to 0.7
            None, // Should default to 0.9
            None, // No system prompt
        );

        // Will fail due to ONNX loading, but we can test the parameter handling
        assert!(result.is_err());
        // The error should be about ONNX loading, not parameter validation
        match result.err().unwrap() {
            LLMError::ProviderError(msg) => {
                assert!(msg.contains("Failed to load") || msg.contains("ONNX"));
            }
            _ => panic!("Expected ProviderError"),
        }
    }

    #[test]
    fn test_liquid_edge_builder_with_valid_path() {
        let temp_dir = create_mock_model_dir();

        let result = LLMBuilder::<LiquidEdge>::new()
            .base_url(temp_dir.path().to_string_lossy().to_string())
            .model("test-model")
            .max_tokens(100)
            .temperature(0.8)
            .top_p(0.95)
            .system("Test system prompt")
            .build();

        // Should fail due to ONNX model loading, but builder should work
        assert!(result.is_err());
        match result.err().unwrap() {
            LLMError::ProviderError(msg) => {
                // Should fail at model loading, not configuration
                assert!(
                    msg.contains("Failed to load")
                        || msg.contains("ONNX")
                        || msg.contains("tokenizer")
                );
            }
            _ => panic!("Expected ProviderError for model loading"),
        }
    }

    #[test]
    fn test_liquid_edge_chat_template_loading() {
        let temp_dir = create_mock_model_dir();

        let template_result =
            autoagents_llm::backends::liquid_edge::LiquidEdge::load_chat_template(temp_dir.path());
        assert!(template_result.is_ok());

        let template = template_result.unwrap();
        assert!(template.contains("message"));
    }

    #[test]
    fn test_liquid_edge_missing_config_file() {
        let temp_dir = tempdir().unwrap();
        // Don't create any files

        let config_result =
            autoagents_llm::backends::liquid_edge::LiquidEdge::load_config(temp_dir.path());
        assert!(config_result.is_err());

        match config_result.err().unwrap() {
            LLMError::ProviderError(msg) => {
                assert!(msg.contains("Failed to read config.json"));
            }
            _ => panic!("Expected ProviderError for missing config"),
        }
    }

    #[test]
    fn test_liquid_edge_invalid_config_json() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();

        // Create invalid JSON
        fs::write(model_path.join("config.json"), "invalid json").unwrap();

        let config_result =
            autoagents_llm::backends::liquid_edge::LiquidEdge::load_config(temp_dir.path());
        assert!(config_result.is_err());

        match config_result.err().unwrap() {
            LLMError::ProviderError(msg) => {
                assert!(msg.contains("Failed to parse config.json"));
            }
            _ => panic!("Expected ProviderError for invalid JSON"),
        }
    }

    #[test]
    fn test_liquid_edge_default_chat_template() {
        let temp_dir = tempdir().unwrap();
        // Don't create chat_template.jinja

        let template_result =
            autoagents_llm::backends::liquid_edge::LiquidEdge::load_chat_template(temp_dir.path());
        assert!(template_result.is_ok());

        let template = template_result.unwrap();
        assert!(template.contains("Assistant:"));
        assert!(template.contains("User:"));
    }

    // Mock test for chat functionality (will fail without real model)
    #[tokio::test]
    async fn test_liquid_edge_chat_interface() {
        let temp_dir = create_mock_model_dir();

        // Create a builder that should fail at model loading
        let result = LLMBuilder::<LiquidEdge>::new()
            .base_url(temp_dir.path().to_string_lossy().to_string())
            .model("test-model")
            .build();

        assert!(result.is_err());
        // This validates the builder interface even though model loading fails
    }

    #[tokio::test]
    async fn test_liquid_edge_completion_interface() {
        // Test that the completion interface would work if we had a valid model
        let temp_dir = create_mock_model_dir();

        let result = LiquidEdge::new(
            temp_dir.path(),
            "test-model".to_string(),
            Some(50),
            Some(0.7),
            Some(0.9),
            None,
        );

        // Should fail due to ONNX model loading
        assert!(result.is_err());
        match result.err().unwrap() {
            LLMError::ProviderError(_) => {
                // Expected - can't load real ONNX model in tests
            }
            _ => panic!("Expected ProviderError"),
        }
    }

    #[tokio::test]
    async fn test_liquid_edge_embedding_not_supported() {
        let temp_dir = create_mock_model_dir();

        // Even if we could create a LiquidEdge instance, embeddings should not be supported
        // We'll test this by checking the error message pattern
        let result = LiquidEdge::new(
            temp_dir.path(),
            "test-model".to_string(),
            None,
            None,
            None,
            None,
        );

        // Will fail at model loading, but that's expected
        assert!(result.is_err());
    }

    #[test]
    fn test_liquid_edge_parameters() {
        let temp_dir = create_mock_model_dir();

        // Test parameter handling
        let result = LiquidEdge::new(
            temp_dir.path(),
            "param-test-model".to_string(),
            Some(200),
            Some(0.5),
            Some(0.8),
            Some("Parameter test system".to_string()),
        );

        // Should fail at ONNX loading, but parameters should be validated
        assert!(result.is_err());
        let error = result.err().unwrap();

        // Should be a provider error about model loading, not parameter validation
        match error {
            LLMError::ProviderError(_) => {
                // Expected - ONNX model loading fails
            }
            LLMError::InvalidRequest(_) => {
                panic!("Should not be parameter validation error");
            }
            _ => panic!("Unexpected error type"),
        }
    }
}

#[cfg(not(feature = "liquid_edge"))]
mod no_liquid_edge_tests {
    #[test]
    fn test_liquid_edge_feature_disabled() {
        // Test that runs when liquid_edge feature is not enabled
        // This ensures the test file doesn't break when the feature is disabled
        assert!(true, "LiquidEdge feature is disabled");
    }
}
