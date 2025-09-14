use std::marker::PhantomData;
use std::path::PathBuf;
use std::pin::Pin;
// use crate::backend::burn_backend_types::InferenceBackend;
use crate::backend::burn_backend_types::InferenceBackend;
use crate::model::llama::{
    generation::{GenerationError, Sampler, TopP},
    tokenizer::SentencePieceTokenizer,
    Llama, LlamaConfig, TinyLlamaVersion,
};
use crate::utils::spawn_future;
use autoagents_llm::chat::{
    ChatMessage, ChatProvider, ChatResponse, StreamResponse, StructuredOutputFormat, Tool,
};
use autoagents_llm::completion::{CompletionProvider, CompletionRequest, CompletionResponse};
use autoagents_llm::embedding::EmbeddingProvider;
use autoagents_llm::error::LLMError;
use autoagents_llm::models::ModelsProvider;
use autoagents_llm::{async_trait, LLMProvider};
use burn::backend::NdArray;
use burn::prelude::Backend;
use futures::Stream;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

/// TinyLlama model wrapper for LLM provider
pub struct TinyLlama<B: Backend> {
    llama: Arc<tokio::sync::Mutex<Llama<InferenceBackend, SentencePieceTokenizer>>>,
    config: GenerationConfig,
    marker: PhantomData<B>,
}

#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub seed: u64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            seed: 42,
        }
    }
}

#[async_trait]
impl<B: Backend> CompletionProvider for TinyLlama<B> {
    async fn complete(
        &self,
        req: &CompletionRequest,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        let mut llama = self.llama.lock().await;
        llama.reset();

        let temperature = req
            .temperature
            .map(|t| t as f64)
            .unwrap_or(self.config.temperature);
        let max_tokens = req
            .max_tokens
            .map(|t| t as usize)
            .unwrap_or(self.config.max_tokens);

        let mut sampler = if temperature > 0.0 {
            crate::model::llama::generation::Sampler::TopP(
                crate::model::llama::generation::TopP::new(self.config.top_p, self.config.seed),
            )
        } else {
            crate::model::llama::generation::Sampler::Argmax
        };

        let result = llama
            .generate(&req.prompt, max_tokens, temperature, &mut sampler)
            .map_err(|e| LLMError::Generic(format!("Generation error: {:?}", e)))?;

        Ok(CompletionResponse {
            text: result.result,
        })
    }
}

#[async_trait]
impl<B: Backend> ChatProvider for TinyLlama<B> {
    async fn chat(
        &self,
        messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        // Format messages into TinyLlama chat format
        let mut prompt = String::new();
        for message in messages {
            prompt.push_str(&format!("<|{}|>\n{}</s>\n", message.role, message.content));
        }
        prompt.push_str("<|assistant|>\n");

        let mut llama = self.llama.lock().await;
        llama.reset();

        let mut sampler = if self.config.temperature > 0.0 {
            crate::model::llama::generation::Sampler::TopP(
                crate::model::llama::generation::TopP::new(self.config.top_p, self.config.seed),
            )
        } else {
            crate::model::llama::generation::Sampler::Argmax
        };

        let result = llama
            .generate(
                &prompt,
                self.config.max_tokens,
                self.config.temperature,
                &mut sampler,
            )
            .map_err(|e| LLMError::Generic(format!("Generation error: {:?}", e)))?;

        Ok(Box::new(SimpleChatResponse {
            content: result.result,
            tokens_used: result.tokens,
        }))
    }

    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError> {
        use futures::stream::StreamExt;

        // Reuse chat_stream_struct and extract content from StreamResponse
        let struct_stream = self
            .chat_stream_struct(messages, tools, json_schema)
            .await?;

        let content_stream = struct_stream.filter_map(|result| async move {
            match result {
                Ok(stream_response) => {
                    // Extract content from the first choice's delta
                    if let Some(choice) = stream_response.choices.first() {
                        if let Some(content) = &choice.delta.content {
                            return Some(Ok(content.clone()));
                        }
                    }
                    // Skip chunks without content (like final usage chunks)
                    None
                }
                Err(e) => Some(Err(e)),
            }
        });

        Ok(Box::pin(content_stream))
    }

    async fn chat_stream_struct(
        &self,
        messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>, LLMError>
    {
        use autoagents_llm::chat::{StreamChoice, StreamDelta};
        use futures::stream::{self, StreamExt};
        use tokio::sync::mpsc;

        // Format messages into TinyLlama chat format
        let mut prompt = String::new();
        for message in messages {
            prompt.push_str(&format!("<|{}>\n{}</s>\n", message.role, message.content));
        }
        prompt.push_str("<|assistant|>\n");

        let llama = self.llama.clone();
        let config = self.config.clone();

        let (tx, mut rx) = mpsc::unbounded_channel::<Result<StreamResponse, LLMError>>();

        // Spawn generation task
        spawn_future(async move {
            let mut llama_lock = llama.lock().await;
            llama_lock.reset();

            let mut sampler = if config.temperature > 0.0 {
                crate::model::llama::generation::Sampler::TopP(
                    crate::model::llama::generation::TopP::new(config.top_p, config.seed),
                )
            } else {
                crate::model::llama::generation::Sampler::Argmax
            };

            let mut total_tokens = 0;
            let result = llama_lock.generate_stream(
                &prompt,
                config.max_tokens,
                config.temperature,
                &mut sampler,
                |_token_id, decoded_text| {
                    // Send each piece of decoded text through the channel as StreamResponse
                    if !decoded_text.is_empty() {
                        total_tokens += 1;
                        let stream_response = StreamResponse {
                            choices: vec![StreamChoice {
                                delta: StreamDelta {
                                    content: Some(decoded_text),
                                    tool_calls: None,
                                },
                            }],
                            usage: None,
                        };
                        let _ = tx.send(Ok(stream_response));
                    }
                    true // Continue generation
                },
            );

            match result {
                Ok(_) => {
                    // Send final response with usage information
                    let final_response = StreamResponse {
                        choices: vec![],
                        usage: Some(autoagents_llm::chat::Usage {
                            prompt_tokens: 0,
                            completion_tokens: total_tokens as u32,
                            total_tokens: total_tokens as u32,
                            completion_tokens_details: None,
                            prompt_tokens_details: None,
                        }),
                    };
                    let _ = tx.send(Ok(final_response));
                }
                Err(e) => {
                    let _ = tx.send(Err(LLMError::Generic(format!("Generation error: {:?}", e))));
                }
            }
        });

        // Convert the receiver into a stream
        let stream = stream::unfold(rx, |mut rx| async move {
            match rx.recv().await {
                Some(item) => Some((item, rx)),
                None => None,
            }
        });

        Ok(Box::pin(stream))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SimpleChatResponse {
    content: String,
    tokens_used: usize,
}

impl ChatResponse for SimpleChatResponse {
    fn text(&self) -> Option<String> {
        Some(self.content.clone())
    }

    fn tool_calls(&self) -> Option<Vec<autoagents_llm::ToolCall>> {
        None
    }

    fn usage(&self) -> Option<autoagents_llm::chat::Usage> {
        Some(autoagents_llm::chat::Usage {
            prompt_tokens: 0,
            completion_tokens: self.tokens_used as u32,
            total_tokens: self.tokens_used as u32,
            completion_tokens_details: None,
            prompt_tokens_details: None,
        })
    }
}

impl std::fmt::Display for SimpleChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.content)
    }
}

#[async_trait]
impl<B: Backend> EmbeddingProvider for TinyLlama<B> {
    async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::Generic(
            "Embeddings not implemented for TinyLlama".to_string(),
        ))
    }
}

impl<B: Backend> ModelsProvider for TinyLlama<B> {}

impl<B: Backend> LLMProvider for TinyLlama<B> {}

pub struct ModelConfig {
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub max_seq_len: usize,
    pub generation_config: GenerationConfig,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/tinyllama.mpk"),
            tokenizer_path: PathBuf::from("models/tokenizer.model"),
            max_seq_len: 512,
            generation_config: GenerationConfig::default(),
        }
    }
}

pub struct TinyLlamaBuilder<T> {
    config: ModelConfig,
    _phantom: PhantomData<T>,
}

impl TinyLlamaBuilder<TinyLlama<InferenceBackend>> {
    pub fn new() -> Self {
        Self {
            config: ModelConfig::default(),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.model_path = path.into();
        self
    }

    pub fn tokenizer_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.tokenizer_path = path.into();
        self
    }

    pub fn max_seq_len(mut self, len: usize) -> Self {
        self.config.max_seq_len = len;
        self
    }

    pub fn temperature(mut self, temp: f64) -> Self {
        self.config.generation_config.temperature = temp;
        self
    }

    pub fn max_tokens(mut self, tokens: usize) -> Self {
        self.config.generation_config.max_tokens = tokens;
        self
    }

    pub fn build(self) -> Result<Arc<TinyLlama<InferenceBackend>>, LLMError> {
        use crate::backend::burn_backend_types::INFERENCE_DEVICE;

        let device = INFERENCE_DEVICE;

        println!("using device {}", crate::backend::burn_backend_types::NAME);

        let model_path = self
            .config
            .model_path
            .to_str()
            .ok_or_else(|| LLMError::Generic("Invalid model path".to_string()))?;
        let tokenizer_path = self
            .config
            .tokenizer_path
            .to_str()
            .ok_or_else(|| LLMError::Generic("Invalid tokenizer path".to_string()))?;

        let llama = LlamaConfig::load_tiny_llama::<InferenceBackend>(
            model_path,
            tokenizer_path,
            self.config.max_seq_len,
            &device,
        )
        .map_err(|e| LLMError::Generic(format!("Failed to load model: {}", e)))?;

        Ok(Arc::new(TinyLlama {
            llama: Arc::new(tokio::sync::Mutex::new(llama)),
            config: self.config.generation_config,
            marker: PhantomData,
        }))
    }
}
