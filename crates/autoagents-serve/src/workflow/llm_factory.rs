use crate::config::ModelConfig;
use crate::error::{Result, WorkflowError};
use autoagents::llm::{
    backends::{anthropic::Anthropic, groq::Groq, ollama::Ollama, openai::OpenAI},
    builder::LLMBuilder,
    LLMProvider,
};
use std::sync::Arc;

pub struct LLMFactory;

impl LLMFactory {
    pub fn create_llm(config: &ModelConfig) -> Result<Arc<dyn LLMProvider>> {
        let provider_str = config.provider.to_lowercase();

        match provider_str.as_str() {
            "openai" => Self::create_openai(config),
            "anthropic" => Self::create_anthropic(config),
            "ollama" => Self::create_ollama(config),
            "groq" => Self::create_groq(config),
            _ => Err(WorkflowError::InvalidModelConfig(format!(
                "Unsupported provider: {}",
                config.provider
            ))),
        }
    }

    fn create_openai(config: &ModelConfig) -> Result<Arc<dyn LLMProvider>> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| WorkflowError::ConfigError("OPENAI_API_KEY not set".to_string()))?;

        let mut builder = LLMBuilder::<OpenAI>::new()
            .api_key(&api_key)
            .model(&config.model_name);

        if let Some(params) = &config.parameters {
            if let Some(temp) = params.temperature {
                builder = builder.temperature(temp);
            }
            if let Some(max_tokens) = params.max_tokens {
                builder = builder.max_tokens(max_tokens);
            }
            if let Some(top_p) = params.top_p {
                builder = builder.top_p(top_p);
            }
        }

        let llm = builder.build()?;
        Ok(llm as Arc<dyn LLMProvider>)
    }

    fn create_anthropic(config: &ModelConfig) -> Result<Arc<dyn LLMProvider>> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| WorkflowError::ConfigError("ANTHROPIC_API_KEY not set".to_string()))?;

        let mut builder = LLMBuilder::<Anthropic>::new()
            .api_key(&api_key)
            .model(&config.model_name);

        if let Some(params) = &config.parameters {
            if let Some(temp) = params.temperature {
                builder = builder.temperature(temp);
            }
            if let Some(max_tokens) = params.max_tokens {
                builder = builder.max_tokens(max_tokens);
            }
            if let Some(top_p) = params.top_p {
                builder = builder.top_p(top_p);
            }
        }

        let llm = builder.build()?;
        Ok(llm as Arc<dyn LLMProvider>)
    }

    fn create_ollama(config: &ModelConfig) -> Result<Arc<dyn LLMProvider>> {
        let base_url = config
            .backend
            .base_url
            .as_deref()
            .unwrap_or("http://localhost:11434");

        let mut builder = LLMBuilder::<Ollama>::new()
            .base_url(base_url)
            .model(&config.model_name);

        if let Some(params) = &config.parameters {
            if let Some(temp) = params.temperature {
                builder = builder.temperature(temp);
            }
            if let Some(top_p) = params.top_p {
                builder = builder.top_p(top_p);
            }
            if let Some(top_k) = params.top_k {
                builder = builder.top_k(top_k);
            }
        }

        let llm = builder.build()?;
        Ok(llm as Arc<dyn LLMProvider>)
    }

    fn create_groq(config: &ModelConfig) -> Result<Arc<dyn LLMProvider>> {
        let api_key = std::env::var("GROQ_API_KEY")
            .map_err(|_| WorkflowError::ConfigError("GROQ_API_KEY not set".to_string()))?;

        let mut builder = LLMBuilder::<Groq>::new()
            .api_key(&api_key)
            .model(&config.model_name);

        if let Some(params) = &config.parameters {
            if let Some(temp) = params.temperature {
                builder = builder.temperature(temp);
            }
            if let Some(max_tokens) = params.max_tokens {
                builder = builder.max_tokens(max_tokens);
            }
        }

        let llm = builder.build()?;
        Ok(llm as Arc<dyn LLMProvider>)
    }
}
