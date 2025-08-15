//! LiquidEdge backend implementation for local model inference.
//!
//! This backend uses the liquid-edge inference runtime to run LLM models locally.
//! It handles tokenization, text generation, and sampling specifically for LLMs.

use crate::{
    builder::LLMBuilder,
    chat::{
        ChatMessage, ChatProvider, ChatResponse, ChatRole, MessageType, StructuredOutputFormat,
    },
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
    LLMProvider, ToolCall,
};
use async_trait::async_trait;
use liquid_edge::{EdgeError, InferenceInput, InferenceRuntime};
use minijinja::{context, Environment};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{path::Path, sync::Arc};
use tokenizers::Tokenizer;

/// LiquidEdge backend for local LLM inference
pub struct LiquidEdge {
    inference_runtime: tokio::sync::Mutex<InferenceRuntime>,
    tokenizer: Tokenizer,
    model_config: ModelConfig,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    system: Option<String>,
    chat_template: Option<String>,
}

/// Model configuration for LLM inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: u32,
    pub max_position_embeddings: u32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
}

/// Generation configuration for text generation
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub do_sample: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 50,
            temperature: 0.7,
            top_p: 0.9,
            do_sample: true,
        }
    }
}

/// Response wrapper for LiquidEdge chat responses
#[derive(Debug)]
pub struct LiquidEdgeResponse {
    text: String,
}

impl ChatResponse for LiquidEdgeResponse {
    fn text(&self) -> Option<String> {
        Some(self.text.clone())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        None // Tool calls not supported yet
    }
}

impl std::fmt::Display for LiquidEdgeResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text)
    }
}

impl LiquidEdge {
    /// Create a new LiquidEdge instance from a model with a specific device
    pub async fn from_model_with_device(
        model: Box<dyn liquid_edge::Model>,
        device: liquid_edge::Device,
        _model_name: String,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        system: Option<String>,
    ) -> Result<Self, LLMError> {
        let model_path = model.model_path().to_path_buf();

        // Load inference runtime with the model and device
        let inference_runtime = InferenceRuntime::from_model_with_device(model, device)
            .await
            .map_err(|e| LLMError::ProviderError(format!("Failed to load model: {e}")))?;

        Self::from_runtime(
            inference_runtime,
            model_path,
            max_tokens,
            temperature,
            top_p,
            system,
        )
        .await
    }

    /// Create a new LiquidEdge instance from a model (uses CPU device by default)
    pub async fn from_model(
        model: Box<dyn liquid_edge::Model>,
        _model_name: String,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        system: Option<String>,
    ) -> Result<Self, LLMError> {
        let model_path = model.model_path().to_path_buf();

        // Load inference runtime with the model
        let inference_runtime = InferenceRuntime::from_model(model)
            .await
            .map_err(|e| LLMError::ProviderError(format!("Failed to load model: {e}")))?;

        Self::from_runtime(
            inference_runtime,
            model_path,
            max_tokens,
            temperature,
            top_p,
            system,
        )
        .await
    }

    /// Common initialization logic for LiquidEdge
    async fn from_runtime(
        inference_runtime: InferenceRuntime,
        model_path: std::path::PathBuf,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        system: Option<String>,
    ) -> Result<Self, LLMError> {
        // Load tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| LLMError::ProviderError(format!("Failed to load tokenizer: {e}")))?;

        // Load model config
        let config_path = model_path.join("config.json");
        let config_content = std::fs::read_to_string(&config_path)
            .map_err(|e| LLMError::ProviderError(format!("Failed to read config.json: {e}")))?;

        let config_json: Value = serde_json::from_str(&config_content)
            .map_err(|e| LLMError::ProviderError(format!("Failed to parse config.json: {e}")))?;

        let model_config = ModelConfig {
            vocab_size: config_json
                .get("vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(32000) as u32,
            max_position_embeddings: config_json
                .get("max_position_embeddings")
                .and_then(|v| v.as_u64())
                .unwrap_or(2048) as u32,
            bos_token_id: config_json
                .get("bos_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32),
            eos_token_id: config_json
                .get("eos_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32),
            pad_token_id: config_json
                .get("pad_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32),
        };

        // Extract chat template from tokenizer or config
        let chat_template = Self::load_chat_template(model_path, &config_json);

        Ok(Self {
            inference_runtime: tokio::sync::Mutex::new(inference_runtime),
            tokenizer,
            model_config,
            max_tokens: max_tokens.unwrap_or(50),
            temperature: temperature.unwrap_or(0.7),
            top_p: top_p.unwrap_or(0.9),
            system,
            chat_template,
        })
    }

    /// Load chat template from chat_template.jinja file, tokenizer.json or config.json
    fn load_chat_template<P: AsRef<Path>>(model_path: P, config: &Value) -> Option<String> {
        let model_path = model_path.as_ref();

        // First try to load from chat_template.jinja file
        let jinja_template_path = model_path.join("chat_template.jinja");
        if jinja_template_path.exists() {
            if let Ok(template_content) = std::fs::read_to_string(&jinja_template_path) {
                log::debug!("Loaded chat template from chat_template.jinja");
                return Some(template_content);
            }
        }

        // Fallback to tokenizer.json
        let tokenizer_path = model_path.join("tokenizer.json");
        if tokenizer_path.exists() {
            if let Ok(tokenizer_content) = std::fs::read_to_string(&tokenizer_path) {
                if let Ok(tokenizer_json) = serde_json::from_str::<Value>(&tokenizer_content) {
                    if let Some(chat_template) =
                        tokenizer_json.get("chat_template").and_then(|v| v.as_str())
                    {
                        log::debug!("Loaded chat template from tokenizer.json");
                        return Some(chat_template.to_string());
                    }
                }
            }
        }

        // Fallback to config.json
        if let Some(chat_template) = config.get("chat_template").and_then(|v| v.as_str()) {
            log::debug!("Loaded chat template from config.json");
            return Some(chat_template.to_string());
        }

        log::debug!("No chat template found");
        None
    }

    // TODO: DO Better
    /// Format messages into a prompt using Jinja2 chat template
    fn format_messages(&self, messages: &[ChatMessage]) -> String {
        // Prepare all messages including system message if provided
        let mut all_messages = Vec::new();

        // Add system message from constructor if provided and no system message in messages
        if let Some(system) = &self.system {
            let has_system = messages.iter().any(|m| matches!(m.role, ChatRole::System));
            if !has_system {
                all_messages.push(ChatMessage {
                    role: ChatRole::System,
                    message_type: MessageType::Text,
                    content: system.clone(),
                });
            }
        }

        // Add provided messages
        all_messages.extend_from_slice(messages);

        // Use Jinja2 chat template if available
        match self.apply_jinja_template(&all_messages) {
            Ok(formatted) => {
                log::debug!("Using Jinja2 chat template");
                formatted
            }
            Err(e) => {
                log::error!("Chat template required but not available or failed: {e}");
                log::error!("Please provide a chat_template.jinja file in the model directory");
                // Return a minimal error-indicating response
                "Error: No chat template found. Please add chat_template.jinja file to model directory.".to_string()
            }
        }
    }

    /// Apply Jinja2 chat template
    fn apply_jinja_template(&self, messages: &[ChatMessage]) -> Result<String, LLMError> {
        let template_str = self
            .chat_template
            .as_ref()
            .ok_or_else(|| LLMError::ProviderError("No chat template available".to_string()))?;

        // Create Jinja2 environment
        let mut env = Environment::new();

        // Convert ChatMessage to template format
        let template_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|msg| {
                let role = match msg.role {
                    ChatRole::System => "system",
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                    ChatRole::Tool => "tool",
                };

                serde_json::json!({
                    "role": role,
                    "content": msg.content
                })
            })
            .collect();

        // Add template to environment
        env.add_template("chat", template_str)
            .map_err(|e| LLMError::ProviderError(format!("Failed to parse chat template: {e}")))?;

        // Render template with messages
        let template = env
            .get_template("chat")
            .map_err(|e| LLMError::ProviderError(format!("Failed to get chat template: {e}")))?;

        let rendered = template
            .render(context! {
                messages => template_messages,
                add_generation_prompt => true,
                bos_token => "<s>",
                eos_token => "</s>",
                system_message => self.system.as_deref().unwrap_or(""),
            })
            .map_err(|e| LLMError::ProviderError(format!("Failed to render chat template: {e}")))?;

        Ok(rendered)
    }

    /// Generate text using the inference runtime
    async fn generate_text(
        &self,
        prompt: &str,
        config: GenerationConfig,
    ) -> Result<String, LLMError> {
        // Tokenize input
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| LLMError::ProviderError(format!("Tokenization failed: {e}")))?;
        let input_tokens: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();

        log::info!(
            "Starting LLM generation with {} input tokens, max_new_tokens: {}",
            input_tokens.len(),
            config.max_new_tokens
        );

        let mut output_tokens = input_tokens.clone();
        let max_length = input_tokens.len() + config.max_new_tokens as usize;

        let mut runtime = self.inference_runtime.lock().await;

        for step in 0..config.max_new_tokens {
            if output_tokens.len() >= max_length {
                log::info!("Reached max length, stopping generation");
                break;
            }

            log::debug!("Generation step {}/{}", step + 1, config.max_new_tokens);

            // Prepare inputs for inference
            let seq_len = output_tokens.len();
            let mut inference_input = InferenceInput::new();

            // Add input_ids
            let input_ids_json = Value::Array(
                output_tokens
                    .iter()
                    .map(|&x| Value::Number(x.into()))
                    .collect(),
            );
            inference_input = inference_input.add_input("input_ids".to_string(), input_ids_json);

            // Add attention_mask
            let attention_mask: Vec<Value> = vec![Value::Number(1.into()); seq_len];
            inference_input = inference_input
                .add_input("attention_mask".to_string(), Value::Array(attention_mask));

            // Add position_ids
            let position_ids: Vec<Value> = (0..seq_len as i64)
                .map(|x| Value::Number(x.into()))
                .collect();
            inference_input =
                inference_input.add_input("position_ids".to_string(), Value::Array(position_ids));

            // Run inference
            log::debug!("Running inference...");
            let output = runtime
                .infer(inference_input)
                .map_err(|e| LLMError::ProviderError(format!("Inference failed: {e}")))?;
            log::debug!("Inference completed");

            // Get logits from output
            let logits = output
                .get_output("logits")
                .ok_or_else(|| LLMError::ProviderError("No logits output found".to_string()))?;

            // Extract logits for the last token
            let logits_array = logits.as_array().ok_or_else(|| {
                LLMError::ProviderError("Logits output is not an array".to_string())
            })?;

            // Get logits for the last token (assuming shape [batch, seq_len, vocab_size])
            let vocab_size = self.model_config.vocab_size as usize;
            let last_token_start = (seq_len - 1) * vocab_size;
            let last_token_end = last_token_start + vocab_size;

            if logits_array.len() < last_token_end {
                return Err(LLMError::ProviderError("Invalid logits shape".to_string()));
            }

            let last_token_logits: Vec<f32> = logits_array[last_token_start..last_token_end]
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();

            log::debug!(
                "Got logits for last token, size: {}",
                last_token_logits.len()
            );

            // Sample next token
            let next_token = if config.do_sample {
                self.sample_token(&last_token_logits, config.temperature, config.top_p)?
            } else {
                // Greedy decoding
                last_token_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i as i64)
                    .ok_or_else(|| {
                        LLMError::ProviderError("Failed to find max logit".to_string())
                    })?
            };

            log::debug!("Generated token: {next_token}");

            // Check for EOS token
            if let Some(eos_id) = self.model_config.eos_token_id {
                if next_token == eos_id as i64 {
                    log::info!("Generated EOS token, stopping generation");
                    break;
                }
            }

            output_tokens.push(next_token);
        }

        log::info!(
            "Generation completed. Total tokens: {}, generated: {}",
            output_tokens.len(),
            output_tokens.len() - input_tokens.len()
        );

        // Decode generated tokens (skip input tokens)
        let generated_tokens: Vec<u32> = output_tokens[input_tokens.len()..]
            .iter()
            .map(|&x| x as u32)
            .collect();
        let generated_text = self
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| LLMError::ProviderError(format!("Failed to decode tokens: {e}")))?;

        log::info!("Generated text: {generated_text}");
        Ok(generated_text)
    }

    /// Sample next token using top-p sampling
    fn sample_token(&self, logits: &[f32], temperature: f32, top_p: f32) -> Result<i64, LLMError> {
        use rand::Rng;

        // Apply temperature
        let scaled_logits: Vec<f32> = logits.iter().map(|x| x / temperature).collect();

        // Convert to probabilities (softmax)
        let max_logit = scaled_logits
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f32> = scaled_logits
            .iter()
            .map(|x| (x - max_logit).exp())
            .collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|x| x / sum_exp).collect();

        // Top-p sampling
        let mut sorted_indices: Vec<usize> = (0..probs.len()).collect();
        sorted_indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

        let mut cumulative_prob = 0.0;
        let mut cutoff_index = probs.len();

        for (i, &idx) in sorted_indices.iter().enumerate() {
            cumulative_prob += probs[idx];
            if cumulative_prob >= top_p {
                cutoff_index = i + 1;
                break;
            }
        }

        // Sample from the top-p distribution
        let mut rng = rand::thread_rng();
        let random_value: f32 = rng.gen();

        let mut cumulative = 0.0;
        for &idx in sorted_indices.iter().take(cutoff_index) {
            cumulative += probs[idx];
            if random_value <= cumulative {
                return Ok(idx as i64);
            }
        }

        // Fallback to the most probable token
        Ok(sorted_indices[0] as i64)
    }
}

#[async_trait]
impl ChatProvider for LiquidEdge {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        _tools: Option<&[crate::chat::Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let mut modified_messages = messages.to_vec();

        // Add JSON format instruction if schema is provided
        if let Some(schema) = &json_schema {
            let default_schema = serde_json::json!({});
            let schema_json = schema.schema.as_ref().unwrap_or(&default_schema);
            let schema_str =
                serde_json::to_string_pretty(schema_json).unwrap_or_else(|_| "{}".to_string());

            //TODO: Improve
            let json_instruction = format!(
                "You must respond with valid JSON that matches this schema: {schema_str}. Only return the JSON, no additional text.").to_string();

            modified_messages.insert(
                0,
                ChatMessage {
                    role: ChatRole::System,
                    message_type: MessageType::Text,
                    content: json_instruction,
                },
            );
        }

        let prompt = self.format_messages(&modified_messages);

        let generation_config = GenerationConfig {
            max_new_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            do_sample: true,
        };

        let response_text = self.generate_text(&prompt, generation_config).await?;
        let cleaned_response = response_text.trim().to_string();

        Ok(Box::new(LiquidEdgeResponse {
            text: if cleaned_response.is_empty() {
                "I'm here to help! What would you like to know?".to_string()
            } else {
                cleaned_response
            },
        }))
    }

    async fn chat(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.chat_with_tools(messages, None, json_schema).await
    }
}

#[async_trait]
impl CompletionProvider for LiquidEdge {
    async fn complete(
        &self,
        req: &CompletionRequest,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        let generation_config = GenerationConfig {
            max_new_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            do_sample: true,
        };

        let text = self.generate_text(&req.prompt, generation_config).await?;
        Ok(CompletionResponse { text })
    }
}

#[async_trait]
impl EmbeddingProvider for LiquidEdge {
    async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::ProviderError(
            "Embedding not supported by LiquidEdge backend".to_string(),
        ))
    }
}

#[async_trait]
impl ModelsProvider for LiquidEdge {
    async fn list_models(
        &self,
        _request: Option<&crate::models::ModelListRequest>,
    ) -> Result<Box<dyn crate::models::ModelListResponse>, LLMError> {
        Err(LLMError::ProviderError(
            "Model listing not supported by LiquidEdge backend".to_string(),
        ))
    }
}

impl LLMProvider for LiquidEdge {
    fn tools(&self) -> Option<&[crate::chat::Tool]> {
        None
    }
}

impl LLMBuilder<LiquidEdge> {
    pub async fn build(self) -> Result<Arc<LiquidEdge>, LLMError> {
        let model_name = self
            .model
            .unwrap_or_else(|| "liquid-edge-model".to_string());

        let liquid_edge = if let Some(edge_model) = self.edge_model {
            if let Some(device) = self.edge_device {
                LiquidEdge::from_model_with_device(
                    edge_model,
                    device,
                    model_name,
                    self.max_tokens,
                    self.temperature,
                    self.top_p,
                    self.system,
                )
                .await?
            } else {
                LiquidEdge::from_model(
                    edge_model,
                    model_name,
                    self.max_tokens,
                    self.temperature,
                    self.top_p,
                    self.system,
                )
                .await?
            }
        } else {
            return Err(LLMError::InvalidRequest(
                "edge_model must be provided for LiquidEdge".to_string(),
            ));
        };

        Ok(Arc::new(liquid_edge))
    }
}

// Convert EdgeError to LLMError
impl From<EdgeError> for LLMError {
    fn from(err: EdgeError) -> Self {
        LLMError::ProviderError(format!("LiquidEdge error: {err}"))
    }
}
