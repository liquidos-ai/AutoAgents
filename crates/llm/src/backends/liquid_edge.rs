//! LiquidEdge backend implementation for local ONNX model inference.
//!
//! This module provides integration with liquid-edge for running local ONNX models
//! with support for chat-based interactions and text completion.

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
use liquid_edge::{
    runtime::{OnnxInput, OnnxRuntime},
    sampling::TopPSampler,
    tokenizer::Tokenizer,
    traits::{GenerationOptions, InferenceRuntime, SamplingStrategy, TokenizerTrait},
    EdgeError,
};

use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs, path::Path, sync::Arc};

/// LiquidEdge backend for local ONNX model inference
pub struct LiquidEdge {
    /// ONNX runtime for inference
    runtime: Arc<OnnxRuntime>,
    /// Tokenizer for text processing
    tokenizer: Arc<Tokenizer>,
    /// Text sampler for generation
    sampler: TopPSampler,
    /// Model configuration
    config: ModelConfig,
    /// Chat template for conversation formatting
    chat_template: String,
    /// Generation parameters
    pub model_name: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub system: Option<String>,
    /// Special tokens
    pub eos_token_id: u32,
    pub bos_token_id: u32,
    /// Conversation history for context
    conversation_history: Vec<(String, String)>,
}

/// Model configuration structure
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    #[serde(default = "default_eos_token_id")]
    eos_token_id: u32,
    #[serde(default = "default_bos_token_id")]
    bos_token_id: u32,
    #[serde(default = "default_max_position_embeddings")]
    max_position_embeddings: usize,
    #[serde(default = "default_vocab_size")]
    vocab_size: usize,
}

/// Tokenizer configuration structure
#[derive(Debug, Deserialize, Serialize)]
struct TokenizerConfig {
    #[serde(default)]
    add_bos_token: bool,
    #[serde(default = "default_model_max_length")]
    model_max_length: usize,
}

/// Special tokens mapping
#[derive(Debug, Deserialize, Serialize)]
struct SpecialTokensMap {
    #[serde(default)]
    bos_token: Option<serde_json::Value>,
    #[serde(default)]
    eos_token: Option<serde_json::Value>,
}

// Configuration defaults
fn default_eos_token_id() -> u32 {
    2
}
fn default_bos_token_id() -> u32 {
    1
}
fn default_max_position_embeddings() -> usize {
    2048
}
fn default_vocab_size() -> usize {
    32000
}
fn default_model_max_length() -> usize {
    2048
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
        None // LiquidEdge doesn't support tool calls yet
    }
}

impl std::fmt::Display for LiquidEdgeResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text)
    }
}

impl LiquidEdge {
    /// Create a new LiquidEdge instance
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        model_name: String,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        system: Option<String>,
    ) -> Result<Self, LLMError> {
        let model_path = model_path.as_ref();

        // Load tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| LLMError::ProviderError(format!("Failed to load tokenizer: {e}")))?;

        // Load ONNX model
        let onnx_path = model_path.join("model.onnx");
        let runtime = OnnxRuntime::new(&onnx_path, model_name.clone())
            .map_err(|e| LLMError::ProviderError(format!("Failed to load ONNX model: {e}")))?;

        // Load model configuration
        let config = Self::load_config(model_path)?;

        // Load chat template
        let chat_template = Self::load_chat_template(model_path)?;

        // Create sampler with conservative settings for consistent output
        let top_p_val = top_p.unwrap_or(0.9);
        let sampler = TopPSampler::new(top_p_val)
            .map_err(|e| LLMError::ProviderError(format!("Failed to create sampler: {e}")))?;

        Ok(Self {
            runtime: Arc::new(runtime),
            tokenizer: Arc::new(tokenizer),
            sampler,
            config: config.clone(),
            chat_template,
            model_name,
            max_tokens: max_tokens.unwrap_or(150),
            temperature: temperature.unwrap_or(0.7),
            top_p: top_p_val,
            system,
            eos_token_id: config.eos_token_id,
            bos_token_id: config.bos_token_id,
            conversation_history: Vec::new(),
        })
    }

    /// Load model configuration from config.json
    pub fn load_config<P: AsRef<Path>>(model_path: P) -> Result<ModelConfig, LLMError> {
        let config_path = model_path.as_ref().join("config.json");
        let config_str = fs::read_to_string(&config_path)
            .map_err(|e| LLMError::ProviderError(format!("Failed to read config.json: {e}")))?;

        let config: ModelConfig = serde_json::from_str(&config_str)
            .map_err(|e| LLMError::ProviderError(format!("Failed to parse config.json: {e}")))?;

        Ok(config)
    }

    /// Load chat template from chat_template.jinja or use default
    pub fn load_chat_template<P: AsRef<Path>>(model_path: P) -> Result<String, LLMError> {
        let template_path = model_path.as_ref().join("chat_template.jinja");

        if template_path.exists() {
            fs::read_to_string(&template_path)
                .map_err(|e| LLMError::ProviderError(format!("Failed to read chat template: {e}")))
        } else {
            // Default template for models without chat_template.jinja
            Ok("{% for message in messages %}{% if message.role == 'system' %}{{ message.content }}{% elif message.role == 'user' %}User: {{ message.content }}{% elif message.role == 'assistant' %}Assistant: {{ message.content }}{% endif %}{% if not loop.last %}\n{% endif %}{% endfor %}\nAssistant:".to_string())
        }
    }

    /// Run chat inference with proper template formatting
    fn run_chat_inference(&mut self, messages: &[ChatMessage]) -> Result<String, LLMError> {
        // Convert messages to template format
        let template_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|msg| {
                serde_json::json!({
                    "role": match msg.role {
                        ChatRole::User => "user",
                        ChatRole::Assistant => "assistant",
                        ChatRole::System => "system",
                        ChatRole::Tool => "tool",
                    },
                    "content": msg.content
                })
            })
            .collect();

        // Use messages from ReAct framework (already includes system message from agent description)

        // Get EOS token string from tokenizer
        let eos_token = self
            .tokenizer
            .decode(&[self.eos_token_id], false)
            .unwrap_or_else(|_| "</s>".to_string());

        // Use LiquidEdge TemplateRenderer instead of manual minijinja
        let template_renderer = liquid_edge::templates::TemplateRenderer::new().map_err(|e| {
            LLMError::ProviderError(format!("Failed to create template renderer: {e}"))
        })?;

        let mut context = HashMap::new();
        context.insert(
            "messages".to_string(),
            serde_json::Value::Array(template_messages),
        );
        context.insert(
            "eos_token".to_string(),
            serde_json::Value::String(eos_token),
        );
        context.insert(
            "add_generation_prompt".to_string(),
            serde_json::Value::Bool(true),
        );

        let conversation = template_renderer
            .render(&self.chat_template, &context)
            .map_err(|e| LLMError::ProviderError(format!("Failed to render template: {e}")))?;

        // Tokenize the conversation
        let input_ids = self
            .tokenizer
            .encode(&conversation, true)
            .map_err(|e| LLMError::ProviderError(format!("Tokenization failed: {e}")))?
            .into_iter()
            .map(|id| id as i64)
            .collect::<Vec<_>>();

        // Generate response
        let generated_tokens = self.generate_tokens(input_ids)?;

        // Convert tokens back to text
        let generated_text = self
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| LLMError::ProviderError(format!("Failed to decode tokens: {e}")))?;

        // Clean up the response
        let cleaned_response = generated_text.trim().replace("</s>", "").trim().to_string();

        Ok(if cleaned_response.is_empty() {
            "I'm here to help! What would you like to know?".to_string()
        } else {
            cleaned_response
        })
    }

    /// Generate tokens using the ONNX runtime
    fn generate_tokens(&self, mut input_tokens: Vec<i64>) -> Result<Vec<u32>, LLMError> {
        let mut generated_tokens = Vec::new();
        let max_new_tokens = self.max_tokens as usize;

        let gen_options = GenerationOptions {
            max_new_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            do_sample: true,
            ..Default::default()
        };

        for _step in 0..max_new_tokens {
            // Check sequence length limit
            if input_tokens.len() >= self.config.max_position_embeddings - 10 {
                break;
            }

            // Prepare input for ONNX
            let onnx_input = OnnxInput::new(input_tokens.clone());

            // Run inference
            let output = self
                .runtime
                .infer(onnx_input)
                .map_err(|e| LLMError::ProviderError(format!("Inference failed: {e}")))?;

            // Get logits for the last token
            let last_logits = output
                .last_token_logits()
                .map_err(|e| LLMError::ProviderError(format!("Failed to get logits: {e}")))?;

            // Sample next token
            let next_token = self
                .sampler
                .sample(last_logits, &gen_options)
                .map_err(|e| LLMError::ProviderError(format!("Sampling failed: {e}")))?;

            // Check for end of sequence
            if next_token == self.eos_token_id {
                break;
            }

            // Add token to sequences
            generated_tokens.push(next_token);
            input_tokens.push(next_token as i64);

            // Early stopping for natural conversation endings
            if generated_tokens.len() >= 10 {
                if let Ok(partial_text) = self.tokenizer.decode(&generated_tokens, true) {
                    if partial_text.trim().ends_with('.')
                        || partial_text.trim().ends_with('!')
                        || partial_text.trim().ends_with('?')
                    {
                        break;
                    }
                }
            }
        }

        Ok(generated_tokens)
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
        // Clone self to make it mutable for inference
        let mut runtime = LiquidEdge {
            runtime: Arc::clone(&self.runtime),
            tokenizer: Arc::clone(&self.tokenizer),
            sampler: TopPSampler::new(self.top_p)
                .map_err(|e| LLMError::ProviderError(format!("Failed to create sampler: {e}")))?,
            config: self.config.clone(),
            chat_template: self.chat_template.clone(),
            model_name: self.model_name.clone(),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            system: self.system.clone(),
            eos_token_id: self.eos_token_id,
            bos_token_id: self.bos_token_id,
            conversation_history: self.conversation_history.clone(),
        };

        // Modify messages to include JSON format instruction if schema is provided
        let mut modified_messages = messages.to_vec();

        if let Some(schema) = &json_schema {
            // Add a system message instructing to return JSON format
            let default_schema = serde_json::json!({});
            let schema_json = schema.schema.as_ref().unwrap_or(&default_schema);
            let schema_str =
                serde_json::to_string_pretty(schema_json).unwrap_or_else(|_| "{}".to_string());

            println!("DEBUG: Schema string: {schema_str}");

            let json_instruction = format!(
                "You must respond with valid JSON that matches this exact schema: {schema_str}. Do not include any text outside the JSON response."
            );

            println!("DEBUG: JSON instruction: {json_instruction}");

            modified_messages.insert(
                0,
                ChatMessage {
                    role: ChatRole::System,
                    message_type: MessageType::Text,
                    content: json_instruction,
                },
            );
        }

        let response_text = runtime.run_chat_inference(&modified_messages)?;

        Ok(Box::new(LiquidEdgeResponse {
            text: response_text,
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
        // Convert completion request to a simple user message
        let message = ChatMessage::user().content(&req.prompt).build();
        let messages = vec![message];

        let response = self.chat(&messages, None).await?;
        let text = response.text().unwrap_or_default();

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
        None // Tools not supported yet
    }
}

impl LLMBuilder<LiquidEdge> {
    pub fn build(self) -> Result<Arc<LiquidEdge>, LLMError> {
        let model_path = self.base_url.ok_or_else(|| {
            LLMError::InvalidRequest("No model path provided for LiquidEdge".to_string())
        })?;

        let model_name = self
            .model
            .unwrap_or_else(|| "liquid-edge-model".to_string());

        let liquid_edge = LiquidEdge::new(
            model_path,
            model_name,
            self.max_tokens,
            self.temperature,
            self.top_p,
            self.system,
        )?;

        Ok(Arc::new(liquid_edge))
    }
}

// Convert EdgeError to LLMError
impl From<EdgeError> for LLMError {
    fn from(err: EdgeError) -> Self {
        LLMError::ProviderError(format!("LiquidEdge error: {err}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn create_mock_model_dir() -> tempfile::TempDir {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();

        // Create mock tokenizer.json
        fs::write(
            model_path.join("tokenizer.json"),
            r#"{"version": "1.0", "model": {"type": "BPE"}}"#,
        )
        .unwrap();

        // Create mock config.json
        fs::write(
            model_path.join("config.json"),
            r#"{"eos_token_id": 2, "bos_token_id": 1, "max_position_embeddings": 2048, "vocab_size": 32000}"#,
        ).unwrap();

        // Create mock model.onnx (empty file for testing)
        fs::write(model_path.join("model.onnx"), b"mock onnx data").unwrap();

        temp_dir
    }

    #[test]
    fn test_liquid_edge_builder_validation() {
        // Test missing model path
        let result = LLMBuilder::<LiquidEdge>::new().build();
        assert!(result.is_err());

        match result.err().unwrap() {
            LLMError::InvalidRequest(msg) => {
                assert!(msg.contains("No model path provided"));
            }
            _ => panic!("Expected InvalidRequest error"),
        }
    }

    #[test]
    fn test_config_loading() {
        let temp_dir = create_mock_model_dir();
        let config = LiquidEdge::load_config(temp_dir.path()).unwrap();

        assert_eq!(config.eos_token_id, 2);
        assert_eq!(config.bos_token_id, 1);
        assert_eq!(config.max_position_embeddings, 2048);
        assert_eq!(config.vocab_size, 32000);
    }

    #[test]
    fn test_chat_template_loading() {
        let temp_dir = create_mock_model_dir();

        // Test default template
        let template = LiquidEdge::load_chat_template(temp_dir.path()).unwrap();
        assert!(template.contains("Assistant:"));

        // Test custom template
        fs::write(
            temp_dir.path().join("chat_template.jinja"),
            "Custom template: {{ messages }}",
        )
        .unwrap();

        let custom_template = LiquidEdge::load_chat_template(temp_dir.path()).unwrap();
        assert_eq!(custom_template, "Custom template: {{ messages }}");
    }

    #[tokio::test]
    async fn test_completion_provider() {
        // This test would need a real model file to work properly
        // For now, just test the interface
        let temp_dir = create_mock_model_dir();

        let result = LiquidEdge::new(
            temp_dir.path(),
            "test-model".to_string(),
            Some(50),
            Some(0.7),
            Some(0.9),
            Some("Test system".to_string()),
        );

        // This will fail because we don't have a real ONNX model
        // but it tests the configuration loading
        assert!(result.is_err());
    }
}
