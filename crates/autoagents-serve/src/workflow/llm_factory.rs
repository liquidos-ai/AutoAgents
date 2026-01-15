use crate::config::ModelConfig;
use crate::error::{Result, WorkflowError};
use autoagents::llm::{
    backends::{anthropic::Anthropic, groq::Groq, ollama::Ollama, openai::OpenAI},
    builder::LLMBuilder,
    LLMProvider,
};
use autoagents_mistral_rs::models::ModelType as MistralModelType;
use autoagents_mistral_rs::{GgufQuant, IsqType, MistralRsProvider, ModelSource};
use std::sync::Arc;

pub struct LLMFactory;

impl LLMFactory {
    pub async fn create_llm(config: &ModelConfig) -> Result<Arc<dyn LLMProvider>> {
        Self::create_llm_with_cache(config, None).await
    }

    pub async fn create_llm_with_cache(
        config: &ModelConfig,
        cached_model: Option<Arc<dyn LLMProvider>>,
    ) -> Result<Arc<dyn LLMProvider>> {
        // Return cached model if available
        if let Some(model) = cached_model {
            log::debug!("Using preloaded model for provider: {}", config.provider);
            return Ok(model);
        }

        let provider_str = config.provider.to_lowercase();

        match provider_str.as_str() {
            "openai" => Self::create_openai(config),
            "anthropic" => Self::create_anthropic(config),
            "ollama" => Self::create_ollama(config),
            "groq" => Self::create_groq(config),
            "mistral" | "mistralrs" | "mistral-rs" => Self::create_mistral_rs(config).await,
            _ => Err(WorkflowError::InvalidModelConfig(format!(
                "Unsupported provider: {}",
                config.provider
            ))),
        }
    }

    #[allow(clippy::result_large_err)]
    fn create_openai(config: &ModelConfig) -> Result<Arc<dyn LLMProvider>> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| WorkflowError::ConfigError("OPENAI_API_KEY not set".to_string()))?;

        let model_name = config.model_name.as_ref().ok_or_else(|| {
            WorkflowError::InvalidModelConfig("model_name is required for OpenAI".to_string())
        })?;

        let mut builder = LLMBuilder::<OpenAI>::new()
            .api_key(&api_key)
            .model(model_name);

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

    #[allow(clippy::result_large_err)]
    fn create_anthropic(config: &ModelConfig) -> Result<Arc<dyn LLMProvider>> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| WorkflowError::ConfigError("ANTHROPIC_API_KEY not set".to_string()))?;

        let model_name = config.model_name.as_ref().ok_or_else(|| {
            WorkflowError::InvalidModelConfig("model_name is required for Anthropic".to_string())
        })?;

        let mut builder = LLMBuilder::<Anthropic>::new()
            .api_key(&api_key)
            .model(model_name);

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

    #[allow(clippy::result_large_err)]
    fn create_ollama(config: &ModelConfig) -> Result<Arc<dyn LLMProvider>> {
        let base_url = config
            .backend
            .base_url
            .as_deref()
            .unwrap_or("http://localhost:11434");

        let model_name = config.model_name.as_ref().ok_or_else(|| {
            WorkflowError::InvalidModelConfig("model_name is required for Ollama".to_string())
        })?;

        let mut builder = LLMBuilder::<Ollama>::new()
            .base_url(base_url)
            .model(model_name);

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

    #[allow(clippy::result_large_err)]
    fn create_groq(config: &ModelConfig) -> Result<Arc<dyn LLMProvider>> {
        let api_key = std::env::var("GROQ_API_KEY")
            .map_err(|_| WorkflowError::ConfigError("GROQ_API_KEY not set".to_string()))?;

        let model_name = config.model_name.as_ref().ok_or_else(|| {
            WorkflowError::InvalidModelConfig("model_name is required for Groq".to_string())
        })?;

        let mut builder = LLMBuilder::<Groq>::new()
            .api_key(&api_key)
            .model(model_name);

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

    async fn create_mistral_rs(config: &ModelConfig) -> Result<Arc<dyn LLMProvider>> {
        let params = config.parameters.as_ref().ok_or_else(|| {
            WorkflowError::InvalidModelConfig(
                "MistralRs requires parameters section in config".to_string(),
            )
        })?;

        // Determine model source
        let model_source = if let Some(model_dir) = &params.model_dir {
            // GGUF local model
            let quant = Self::parse_gguf_quant(params.quant.as_deref().unwrap_or("q4_k_m"))?;
            let file_name = format!("model-{}.gguf", quant.file_suffix());

            ModelSource::Gguf {
                model_dir: model_dir.clone(),
                files: vec![file_name],
                tokenizer: None,
                chat_template: None,
            }
        } else if let Some(source) = &config.source {
            // HuggingFace model
            let model_type = match params.model_type.as_deref() {
                Some("text") => MistralModelType::Text,
                Some("vision") => MistralModelType::Vision,
                _ => MistralModelType::Auto,
            };

            ModelSource::HuggingFace {
                repo_id: source.clone(),
                revision: None,
                model_type,
            }
        } else {
            return Err(WorkflowError::InvalidModelConfig(
                "MistralRs requires either 'source' (HF repo) or 'model_dir' (local GGUF) in parameters".to_string(),
            ));
        };

        // Build provider
        let mut builder = MistralRsProvider::builder().model_source(model_source);

        // Apply parameters
        if let Some(temp) = params.temperature {
            builder = builder.temperature(temp);
        }
        if let Some(max_tokens) = params.max_tokens {
            builder = builder.max_tokens(max_tokens);
        }

        // ISQ quantization for HuggingFace models
        if config.source.is_some() && params.model_dir.is_none() {
            if let Some(quant_str) = &params.quant {
                let isq = Self::parse_isq_quant(quant_str)?;
                builder = builder.with_isq(isq);
            }
        }

        // Paged attention
        if params.paged_attention.unwrap_or(false) {
            builder = builder.with_paged_attention();
        }

        // Verbose logging
        if params.verbose.unwrap_or(false) {
            builder = builder.with_logging();
        }

        // Build and convert to trait object
        let provider = builder.build().await.map_err(|e| {
            WorkflowError::InvalidModelConfig(format!("Failed to build MistralRs provider: {}", e))
        })?;

        Ok(Arc::new(provider) as Arc<dyn LLMProvider>)
    }

    #[allow(clippy::result_large_err)]
    fn parse_gguf_quant(quant_str: &str) -> Result<GgufQuant> {
        match quant_str.to_lowercase().as_str() {
            "q4" | "q4_k_m" | "q4km" => Ok(GgufQuant::Q4_K_M),
            "q4_k_s" | "q4ks" => Ok(GgufQuant::Q4_K_S),
            "q5" | "q5_k_m" | "q5km" => Ok(GgufQuant::Q5_K_M),
            "q5_k_s" | "q5ks" => Ok(GgufQuant::Q5_K_S),
            "q8" | "q8_0" => Ok(GgufQuant::Q8_0),
            "f16" => Ok(GgufQuant::F16),
            "f32" => Ok(GgufQuant::F32),
            _ => Err(WorkflowError::InvalidModelConfig(format!(
                "Unknown GGUF quantization: {}",
                quant_str
            ))),
        }
    }

    #[allow(clippy::result_large_err)]
    fn parse_isq_quant(quant_str: &str) -> Result<IsqType> {
        match quant_str.to_lowercase().as_str() {
            "q4" | "q4_0" => Ok(IsqType::Q4_0),
            "q4_1" => Ok(IsqType::Q4_1),
            "q5_0" => Ok(IsqType::Q5_0),
            "q5_1" => Ok(IsqType::Q5_1),
            "q8" | "q8_0" => Ok(IsqType::Q8_0),
            "q8_1" => Ok(IsqType::Q8_1),
            _ => Err(WorkflowError::InvalidModelConfig(format!(
                "Unknown ISQ quantization: {}",
                quant_str
            ))),
        }
    }
}
