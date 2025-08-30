use crate::console_log;
use crate::llama::Model;
use async_trait::async_trait;
use autoagents::llm::{
    chat::{
        ChatMessage, ChatProvider, ChatResponse, ChatRole, MessageType, StreamChoice, StreamDelta,
        StreamResponse, StructuredOutputFormat, Tool, Usage,
    },
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
    LLMProvider, ToolCall,
};
use futures::stream;
use futures::stream::{Stream, StreamExt};
use minijinja::{context, Environment};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

/// TinyLlama model provider for AutoAgents
pub struct LlamaProvider {
    model: Arc<Mutex<Model>>,
    temperature: f64,
    top_p: f64,
    repeat_penalty: f32,
    repeat_last_n: usize,
    seed: u64,
}

impl LlamaProvider {
    /// Create a new LlamaProvider with a loaded TinyLlama model
    pub fn new(
        model: Model,
        temperature: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: Option<f32>,
        repeat_last_n: Option<usize>,
        seed: Option<u64>,
    ) -> Self {
        Self {
            model: Arc::new(Mutex::new(model)),
            temperature: temperature.unwrap_or(0.7),
            top_p: top_p.unwrap_or(0.9),
            repeat_penalty: repeat_penalty.unwrap_or(1.0),
            repeat_last_n: repeat_last_n.unwrap_or(64),
            seed: seed.unwrap_or(42),
        }
    }

    fn format_messages(
        &self,
        template: String,
        _system_message: String, // Unused, kept for compatibility
        messages: &[ChatMessage],
    ) -> String {
        // Use all messages as provided
        let all_messages = Vec::from(messages);

        // Use Jinja2 chat template
        match self.apply_jinja_template(template, &all_messages) {
            Ok(formatted) => formatted,
            Err(e) => {
                format!("Error: No chat template found. Please add chat_template.jinja file to model directory. {}", e)
            }
        }
    }

    /// Apply Jinja2 chat template
    fn apply_jinja_template(
        &self,
        template: String,
        messages: &[ChatMessage],
    ) -> Result<String, LLMError> {
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

        console_log!("Template: {}", template);

        // Add template to environment
        env.add_template("chat", &template)
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
                eos_token => "</s>"
            })
            .map_err(|e| LLMError::ProviderError(format!("Failed to render chat template: {e}")))?;

        Ok(rendered)
    }
}

/// Response structure for TinyLlama chat responses
#[derive(Debug, Clone)]
pub struct LlamaChatResponse {
    pub text: Option<String>,
    pub usage: Option<Usage>,
}

impl ChatResponse for LlamaChatResponse {
    fn text(&self) -> Option<String> {
        self.text.clone()
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        None
    }

    fn usage(&self) -> Option<Usage> {
        self.usage.clone()
    }
}

impl std::fmt::Display for LlamaChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text.as_deref().unwrap_or(""))
    }
}

/// TODO: Does not support tool calling
#[async_trait]
impl ChatProvider for LlamaProvider {
    async fn chat(
        &self,
        messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        // Convert messages to proper TinyLlama format using Jinja template
        let prompt = self.format_messages(
            include_str!("../models/chat_template.jinja").into(),
            String::new(), // No separate system message needed, it's handled in the template
            messages,
        );

        let mut model_guard = self.model.lock().unwrap();

        // Initialize the model with the prompt
        let initial_response = model_guard
            .init_with_prompt(
                prompt,
                self.temperature,
                self.top_p,
                self.repeat_penalty,
                self.repeat_last_n,
                self.seed,
            )
            .map_err(|e| LLMError::Generic(format!("{:?}", e)))?;

        let mut full_response = initial_response;

        // Generate tokens until we get a complete response or hit max tokens
        let max_tokens = 512; // Configurable limit
        let mut token_count = 0;

        while token_count < max_tokens {
            match model_guard.next_token() {
                Ok(token) => {
                    console_log!("Generated token: '{}'", token);

                    // Check for stop conditions first (before any cleaning)
                    if token == "</s>"
                        || token == "<|endoftext|>"
                        || token.starts_with("<|user|>")
                        || token.starts_with("<|system|>")
                        || token.starts_with("<|assistant|>")
                    {
                        console_log!("Stop condition met, breaking generation");
                        break;
                    }

                    // Skip empty tokens but continue generation
                    if token.is_empty() {
                        console_log!("Empty token, skipping but continuing...");
                        continue;
                    }

                    // Only clean up malformed partial tokens, not complete content
                    let cleaned_token = if token.contains("<|") && !token.starts_with("<|") {
                        // Remove partial special tokens that might appear mid-generation
                        token
                            .replace("<|user|>", "")
                            .replace("<|assistant|>", "")
                            .replace("<|system|>", "")
                    } else {
                        token
                    };

                    if cleaned_token.is_empty() {
                        continue;
                    }

                    full_response.push_str(&cleaned_token);
                    token_count += 1;
                }
                Err(e) => {
                    return Err(LLMError::Generic(format!(
                        "Token generation error: {:?}",
                        e
                    )));
                }
            }
        }

        let usage = Usage {
            prompt_tokens: 0, // We don't have exact token counts from the candle model
            completion_tokens: token_count,
            total_tokens: token_count,
            completion_tokens_details: None,
            prompt_tokens_details: None,
        };

        Ok(Box::new(LlamaChatResponse {
            text: Some(full_response),
            usage: Some(usage),
        }))
    }

    /// Streaming chat implementation that returns structured response chunks
    async fn chat_stream_struct(
        &self,
        messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>,
        LLMError,
    > {
        // Convert messages to proper TinyLlama format using Jinja template
        let prompt = self.format_messages(
            include_str!("../models/chat_template.jinja").into(),
            String::new(), // No separate system message needed, it's handled in the template
            messages,
        );
        console_log!("Generated Prompt: {:?}", prompt);

        // Clone the model Arc to use in the async stream
        let model_arc = Arc::clone(&self.model);
        let temperature = self.temperature;
        let top_p = self.top_p;
        let repeat_penalty = self.repeat_penalty;
        let repeat_last_n = self.repeat_last_n;
        let seed = self.seed;

        // Create an async stream that yields StreamResponse objects for each token
        let stream = stream::unfold(
            (model_arc, prompt, true, 0u32),
            move |(model_arc, prompt, is_first, token_count)| async move {
                let max_tokens = 512; // Configurable limit

                if token_count >= max_tokens {
                    return None;
                }

                // Clone model_arc before using it to avoid borrow checker issues
                let model_arc_clone = Arc::clone(&model_arc);
                let result = {
                    let mut model_guard = model_arc_clone.lock().unwrap();

                    if is_first {
                        // Initialize the model with the prompt on first call
                        match model_guard.init_with_prompt(
                            prompt.clone(),
                            temperature,
                            top_p,
                            repeat_penalty,
                            repeat_last_n,
                            seed,
                        ) {
                            Ok(initial_token) => {
                                console_log!("Initial token: '{}'", initial_token);

                                // Check for stop conditions first
                                if initial_token == "</s>"
                                    || initial_token == "<|endoftext|>"
                                    || initial_token.starts_with("<|user|>")
                                    || initial_token.starts_with("<|system|>")
                                    || initial_token.starts_with("<|assistant|>")
                                {
                                    console_log!(
                                        "Stop condition met for initial token: '{}'",
                                        initial_token
                                    );
                                    return None;
                                }

                                // Skip empty tokens but don't stop the stream
                                if initial_token.is_empty() {
                                    console_log!("Empty initial token, skipping...");
                                    // Continue to next token instead of stopping
                                    return Some((
                                        Ok(StreamResponse {
                                            choices: vec![StreamChoice {
                                                delta: StreamDelta {
                                                    content: Some("".to_string()),
                                                    tool_calls: None,
                                                },
                                            }],
                                            usage: None,
                                        }),
                                        (model_arc, prompt, false, token_count + 1),
                                    ));
                                }

                                // Only clean up malformed partial tokens
                                let cleaned_token = if initial_token.contains("<|")
                                    && !initial_token.starts_with("<|")
                                {
                                    initial_token
                                        .replace("<|user|>", "")
                                        .replace("<|assistant|>", "")
                                        .replace("<|system|>", "")
                                } else {
                                    initial_token
                                };

                                if cleaned_token.is_empty() {
                                    return None;
                                }
                                Ok(cleaned_token)
                            }
                            Err(e) => {
                                console_log!("Init error: {:?}", e);
                                Err(LLMError::Generic(format!("Init error: {:?}", e)))
                            }
                        }
                    } else {
                        // Generate the next token
                        match model_guard.next_token() {
                            Ok(token) => {
                                console_log!("Streamed token: '{}'", token);

                                // Check for stop conditions first
                                if token == "</s>"
                                    || token == "<|endoftext|>"
                                    || token.starts_with("<|user|>")
                                    || token.starts_with("<|system|>")
                                    || token.starts_with("<|assistant|>")
                                {
                                    console_log!("Stop condition met for token: '{}'", token);
                                    return None;
                                }

                                // Skip empty tokens but continue generation
                                if token.is_empty() {
                                    console_log!("Empty token, skipping...");
                                    return Some((
                                        Ok(StreamResponse {
                                            choices: vec![StreamChoice {
                                                delta: StreamDelta {
                                                    content: Some("".to_string()),
                                                    tool_calls: None,
                                                },
                                            }],
                                            usage: None,
                                        }),
                                        (model_arc, prompt, false, token_count + 1),
                                    ));
                                }

                                // Only clean up malformed partial tokens
                                let cleaned_token =
                                    if token.contains("<|") && !token.starts_with("<|") {
                                        token
                                            .replace("<|user|>", "")
                                            .replace("<|assistant|>", "")
                                            .replace("<|system|>", "")
                                    } else {
                                        token
                                    };

                                if cleaned_token.is_empty() {
                                    return None;
                                }
                                Ok(cleaned_token)
                            }
                            Err(e) => {
                                console_log!("Token generation error: {:?}", e);
                                Err(LLMError::Generic(format!(
                                    "Token generation error: {:?}",
                                    e
                                )))
                            }
                        }
                    }
                };

                match result {
                    Ok(token) => {
                        console_log!("Llama provider yielding token to stream: '{}'", token);
                        let stream_response = StreamResponse {
                            choices: vec![StreamChoice {
                                delta: StreamDelta {
                                    content: Some(token.clone()),
                                    tool_calls: None,
                                },
                            }],
                            usage: None, // Usage is only sent in the final chunk
                        };

                        Some((
                            Ok(stream_response),
                            (model_arc, prompt, false, token_count + 1),
                        ))
                    }
                    Err(e) => Some((Err(e), (model_arc, prompt, false, token_count + 1))),
                }
            },
        );

        Ok(Box::pin(stream))
    }
}

#[async_trait]
impl CompletionProvider for LlamaProvider {
    async fn complete(
        &self,
        req: &CompletionRequest,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        let mut model_guard = self.model.lock().unwrap();

        // Initialize the model with the prompt
        let initial_response = model_guard
            .init_with_prompt(
                req.prompt.clone(),
                self.temperature,
                self.top_p,
                self.repeat_penalty,
                self.repeat_last_n,
                self.seed,
            )
            .map_err(|e| LLMError::Generic(format!("{:?}", e)))?;

        let mut full_response = initial_response;

        // Generate tokens until we get a complete response or hit max tokens
        let max_tokens = req.max_tokens.unwrap_or(512);
        let mut token_count = 0;

        while token_count < max_tokens {
            match model_guard.next_token() {
                Ok(token) => {
                    // Check for stop conditions first
                    if token == "</s>"
                        || token == "<|endoftext|>"
                        || token.is_empty()
                        || token.starts_with("<|user|>")
                        || token.starts_with("<|system|>")
                        || token.starts_with("<|assistant|>")
                    {
                        break;
                    }

                    // Only clean up malformed partial tokens
                    let cleaned_token = if token.contains("<|") && !token.starts_with("<|") {
                        token
                            .replace("<|user|>", "")
                            .replace("<|assistant|>", "")
                            .replace("<|system|>", "")
                    } else {
                        token
                    };

                    if cleaned_token.is_empty() {
                        continue;
                    }
                    full_response.push_str(&cleaned_token);
                    token_count += 1;
                }
                Err(e) => {
                    return Err(LLMError::Generic(format!(
                        "Token generation error: {:?}",
                        e
                    )));
                }
            }
        }

        Ok(CompletionResponse {
            text: full_response,
        })
    }
}

#[async_trait]
impl EmbeddingProvider for LlamaProvider {
    async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::Generic(
            "TinyLlama model does not support embeddings".to_string(),
        ))
    }
}

#[async_trait]
impl ModelsProvider for LlamaProvider {}

impl LLMProvider for LlamaProvider {}
