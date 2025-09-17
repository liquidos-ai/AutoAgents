use crate::backend::burn_backend_types::InferenceBackend;
use crate::model::llama::generation::{stream_sender, Sampler, TopP};
use crate::model::llama::tokenizer::Tokenizer;
use crate::model::llama::Llama;
use crate::utils::{receiver_into_stream, spawn_future, CustomMutex};
use autoagents_llm::chat::{
    ChatMessage, ChatProvider, ChatResponse, StreamResponse, StructuredOutputFormat, Tool,
};
use autoagents_llm::completion::{CompletionProvider, CompletionRequest, CompletionResponse};
use autoagents_llm::embedding::EmbeddingProvider;
use autoagents_llm::error::LLMError;
use autoagents_llm::models::ModelsProvider;
use autoagents_llm::{async_trait, LLMProvider};
use burn::prelude::Backend;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;

pub(crate) enum LLamaModel {
    TinyLLama,
    Llama3,
}

/// Llama model wrapper for LLM provider
pub struct LlamaChat<B: Backend, T: Tokenizer> {
    pub(crate) llama: Arc<CustomMutex<Llama<InferenceBackend, T>>>,
    pub(crate) config: GenerationConfig,
    pub(crate) marker: PhantomData<B>,
    pub(crate) model: LLamaModel,
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

impl<B: Backend, T: Tokenizer> LlamaChat<B, T> {
    fn prompt(&self, messages: &[ChatMessage]) -> Result<String, LLMError> {
        match self.model {
            LLamaModel::Llama3 => {
                let mut prompt: Vec<String> = vec![];
                for message in messages {
                    prompt.push(format!(
                        "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                        message.role.to_string().to_lowercase(),
                        message.content
                    ));
                }
                let mut prompt = prompt.join("");
                prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                Ok(prompt)
            }
            LLamaModel::TinyLLama => {
                let mut prompt: Vec<String> = vec![];
                for message in messages {
                    prompt.push(format!("<|{}|>\n{}</s>\n", message.role, message.content));
                }
                let mut prompt = prompt.join("\n");
                prompt.push_str("<|assistant|>\n");
                Ok(prompt)
            }
        }
    }
}

#[async_trait]
impl<B: Backend, T: Tokenizer> CompletionProvider for LlamaChat<B, T> {
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
            Sampler::TopP(TopP::new(self.config.top_p, self.config.seed))
        } else {
            Sampler::Argmax
        };

        let result = llama
            .generate(&req.prompt, max_tokens, temperature, &mut sampler, None)
            .await
            .map_err(|e| LLMError::Generic(format!("Generation error: {:?}", e)))?;

        Ok(CompletionResponse {
            text: result.result,
        })
    }
}

#[async_trait]
impl<B: Backend, T: Tokenizer> ChatProvider for LlamaChat<B, T> {
    async fn chat(
        &self,
        messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        // Format messages into chat format
        let prompt = self.prompt(messages)?;

        let mut llama = self.llama.lock().await;
        llama.reset();

        let mut sampler = if self.config.temperature > 0.0 {
            Sampler::TopP(TopP::new(self.config.top_p, self.config.seed))
        } else {
            Sampler::Argmax
        };

        let result = llama
            .generate(
                &prompt,
                self.config.max_tokens,
                self.config.temperature,
                &mut sampler,
                None,
            )
            .await
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
        // Format messages into Llama chat format
        let prompt = self.prompt(messages)?;

        let llama = self.llama.clone();
        let config = self.config.clone();

        let (tx, rx) = stream_sender::StreamSender::new();

        // Spawn generation task
        spawn_future(async move {
            let mut llama_lock = llama.lock().await;
            llama_lock.reset();

            let mut sampler = if config.temperature > 0.0 {
                Sampler::TopP(TopP::new(config.top_p, config.seed))
            } else {
                Sampler::Argmax
            };

            let total_tokens = 0;

            let result = llama_lock
                .generate(
                    &prompt,
                    config.max_tokens,
                    config.temperature,
                    &mut sampler,
                    Some(tx.clone()),
                )
                .await;

            match result {
                Ok(_) => {
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
                    tx.send(Ok(final_response)).await;
                }
                Err(e) => {
                    tx.send(Err(LLMError::Generic(format!("Generation error: {:?}", e))))
                        .await;
                }
            }
        });

        Ok(receiver_into_stream(rx))
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
impl<B: Backend, T: Tokenizer> EmbeddingProvider for LlamaChat<B, T> {
    async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::Generic(
            "Embeddings not implemented for TinyLlama".to_string(),
        ))
    }
}

impl<B: Backend, T: Tokenizer> ModelsProvider for LlamaChat<B, T> {}

impl<B: Backend, T: Tokenizer> LLMProvider for LlamaChat<B, T> {}
