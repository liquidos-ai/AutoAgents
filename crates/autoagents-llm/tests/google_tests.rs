#![allow(unused_imports)]
use autoagents_llm::{builder::LLMBuilder, error::LLMError};
use std::sync::Arc;

#[cfg(feature = "google")]
mod google_test_cases {
    use super::*;
    use autoagents_llm::backends::google::Google;

    fn build_google() -> Arc<Google> {
        LLMBuilder::<Google>::new()
            .api_key("test-key")
            .model("gemini-1.5-flash")
            .max_tokens(100)
            .temperature(0.7)
            .build()
            .expect("Failed to build Google client")
    }

    #[test]
    fn test_google_creation() {
        let client = build_google();
        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.model, "gemini-1.5-flash");
    }

    #[test]
    fn test_google_builder_validation() {
        let result = LLMBuilder::<Google>::new()
            .model("gemini-1.5-flash")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_google_default_values() {
        let client = Google::new("test-key", None, None, None, None, None, None);
        assert_eq!(client.model, "gemini-1.5-flash");
    }
}
