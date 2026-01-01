#![allow(unused_imports)]
use autoagents_llm::{
    builder::LLMBuilder,
    chat::{ChatMessage, ChatProvider},
    error::LLMError,
};
use std::sync::Arc;

#[cfg(feature = "openai")]
mod openai_test_cases {
    use super::*;
    use autoagents_llm::backends::openai::OpenAI;

    fn build_test_openai() -> Arc<OpenAI> {
        LLMBuilder::<OpenAI>::new()
            .api_key("test-key")
            .model("gpt-4o")
            .max_tokens(100)
            .temperature(0.7)
            .build()
            .expect("Failed to build OpenAI client")
    }

    fn new_openai(api_key: &str, model: Option<String>) -> Result<OpenAI, LLMError> {
        OpenAI::new(
            api_key.to_string(),
            None,
            model,
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
            None,
        )
    }

    #[test]
    fn test_openai_creation() {
        let client = build_test_openai();
        assert_eq!(client.api_key(), "test-key");
        assert_eq!(client.model(), "gpt-4o");
    }

    #[test]
    fn test_openai_builder_validation() {
        let result = LLMBuilder::<OpenAI>::new().model("gpt-4").build();
        assert!(result.is_err());
    }

    #[test]
    fn test_openai_default_values() {
        let client = new_openai("test-key", None).expect("should build");
        assert_eq!(client.model(), "gpt-4.1-nano");
        assert_eq!(client.base_url().as_str(), "https://api.openai.com/v1/");
    }

    #[tokio::test]
    async fn test_chat_auth_error() {
        let client = new_openai("", None);
        assert!(client.is_err());
    }
}
