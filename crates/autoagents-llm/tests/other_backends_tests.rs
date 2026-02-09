#![allow(unused_imports)]
use autoagents_llm::{builder::LLMBuilder, error::LLMError};
use std::sync::Arc;

#[cfg(feature = "xai")]
mod xai_tests {
    use super::*;
    use autoagents_llm::backends::xai::XAI;

    #[test]
    fn test_xai_builds() {
        let client = LLMBuilder::<XAI>::new()
            .api_key("test-key")
            .model("grok-2-latest")
            .build()
            .expect("Failed to build XAI client");
        assert_eq!(client.model, "grok-2-latest");
    }
}

#[cfg(feature = "phind")]
mod phind_tests {
    use super::*;
    use autoagents_llm::backends::phind::Phind;

    #[test]
    fn test_phind_builds() {
        let client = LLMBuilder::<Phind>::new()
            .api_key("test-key")
            .model("phind-codellama-34b")
            .build()
            .expect("Failed to build Phind client");
        assert_eq!(client.model, "phind-codellama-34b");
    }
}

#[cfg(feature = "groq")]
mod groq_tests {
    use super::*;
    use autoagents_llm::backends::groq::Groq;

    #[test]
    fn test_groq_builds() {
        let client = LLMBuilder::<Groq>::new()
            .api_key("test-key")
            .model("llama3-8b-8192")
            .build()
            .expect("Failed to build Groq client");
        assert_eq!(client.model, "llama3-8b-8192");
    }
}
