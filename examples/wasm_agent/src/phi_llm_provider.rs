use crate::phi_provider::{PhiInitInput, PhiModel, PhiTokenOutput};
use async_trait::async_trait;
use autoagents_llm::chat::{
    ChatMessage, ChatProvider, ChatResponse, ChatRole, MessageType, StructuredOutputFormat, Tool,
};
use autoagents_llm::completion::{CompletionProvider, CompletionRequest, CompletionResponse};
use autoagents_llm::embedding::EmbeddingProvider;
use autoagents_llm::error::LLMError;
use autoagents_llm::models::ModelsProvider;
use autoagents_llm::LLMProvider;
use futures::stream::{Stream, StreamExt};
use std::any::Any;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use wasm_bindgen::JsValue;

/// Wrapper to make a stream Send for WASM compatibility
#[cfg(target_arch = "wasm32")]
struct WasmSendStream<T> {
    inner: Pin<Box<T>>,
}

#[cfg(target_arch = "wasm32")]
unsafe impl<T> Send for WasmSendStream<T> {}

#[cfg(target_arch = "wasm32")]
impl<T> WasmSendStream<T> {
    fn new(stream: Pin<Box<T>>) -> Self {
        Self { inner: stream }
    }
}

#[cfg(target_arch = "wasm32")]
impl<T> Stream for WasmSendStream<T>
where
    T: Stream,
{
    type Item = T::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

/// LLM provider wrapper for PhiModel that implements the AutoAgents LLM provider traits
pub struct PhiLLMProvider {
    pub model: Arc<Mutex<PhiModel>>,
}

impl PhiLLMProvider {
    pub fn new(model: PhiModel) -> Self {
        Self {
            model: Arc::new(Mutex::new(model)),
        }
    }
}

/// Response struct for Phi chat responses
#[derive(Debug, Clone)]
pub struct PhiChatResponse {
    text: String,
}

impl ChatResponse for PhiChatResponse {
    fn text(&self) -> Option<String> {
        Some(self.text.clone())
    }

    fn tool_calls(&self) -> Option<Vec<autoagents_llm::ToolCall>> {
        None // Phi model doesn't support tool calls
    }
}

impl std::fmt::Display for PhiChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text)
    }
}

#[async_trait]
impl ChatProvider for PhiLLMProvider {
    async fn chat(
        &self,
        messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        unimplemented!()
    }

    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            // For non-WASM targets, return an error since this example is WASM-only
            return Err(LLMError::Generic(
                "This PhiLLMProvider is designed for WASM only".to_string(),
            ));
        }

        #[cfg(target_arch = "wasm32")]
        {
            crate::console_log!("PhiLLMProvider::chat_stream called");

            let prompt = self.messages_to_prompt(messages);
            crate::console_log!("Generated prompt: {}", &prompt[..100.min(prompt.len())]);

            let model_arc = self.model.clone();

            // Create platform-specific async streams
            #[cfg(target_arch = "wasm32")]
            let stream = async_stream::stream! {
                crate::console_log!("Starting WASM async token generation stream");

                // Lock the model for token generation
                let mut model_guard = match model_arc.lock() {
                    Ok(guard) => guard,
                    Err(e) => {
                        let error_msg = format!("Failed to lock model: {}", e);
                        crate::console_log!("Error: {}", error_msg);
                        yield Err(LLMError::ProviderError(error_msg));
                        return;
                    }
                };

                // Initialize with the prompt
                let init_data = PhiInitInput {
                    prompt,
                    temp: 0.7,
                    top_p: 0.9,
                    repeat_penalty: 1.1,
                    repeat_last_n: 64,
                    seed: 42,
                };

                let init_data_js = match serde_wasm_bindgen::to_value(&init_data) {
                    Ok(js_val) => js_val,
                    Err(e) => {
                        let error_msg = format!("Failed to serialize init data: {}", e);
                        crate::console_log!("Error: {}", error_msg);
                        yield Err(LLMError::ProviderError(error_msg));
                        return;
                    }
                };

                let first_token_result = match model_guard.init_with_prompt(init_data_js) {
                    Ok(result) => result,
                    Err(e) => {
                        let error_msg = format!("Failed to initialize model: {:?}", e);
                        crate::console_log!("Error: {}", error_msg);
                        yield Err(LLMError::ProviderError(error_msg));
                        return;
                    }
                };

                let first_token: PhiTokenOutput = match serde_wasm_bindgen::from_value(first_token_result) {
                    Ok(token) => token,
                    Err(e) => {
                        let error_msg = format!("Failed to deserialize first token: {}", e);
                        crate::console_log!("Error: {}", error_msg);
                        yield Err(LLMError::ProviderError(error_msg));
                        return;
                    }
                };

                crate::console_log!("Yielding first token: '{}'", first_token.token);
                yield Ok(first_token.token);

                // Yield control to allow other tasks to run (WASM-specific)
                wasm_bindgen_futures::JsFuture::from(
                    js_sys::Promise::resolve(&wasm_bindgen::JsValue::from(0))
                ).await.ok();

                // Generate remaining tokens one by one with async yields
                for i in 0..256 {
                    let token_result = match model_guard.next_token() {
                        Ok(result) => result,
                        Err(e) => {
                            let error_msg = format!("Failed to generate token {}: {:?}", i + 1, e);
                            crate::console_log!("Error: {}", error_msg);
                            yield Err(LLMError::ProviderError(error_msg));
                            return;
                        }
                    };

                    let token: PhiTokenOutput = match serde_wasm_bindgen::from_value(token_result) {
                        Ok(token) => token,
                        Err(e) => {
                            let error_msg = format!("Failed to deserialize token {}: {}", i + 1, e);
                            crate::console_log!("Error: {}", error_msg);
                            yield Err(LLMError::ProviderError(error_msg));
                            return;
                        }
                    };

                    if token.token == "<|endoftext|>" {
                        crate::console_log!("End of text reached at token {}", i + 1);
                        break;
                    }

                    crate::console_log!("Yielding token {}: '{}'", i + 1, token.token);
                    yield Ok(token.token);

                    // Yield control to allow other tasks to run after each token (WASM-specific)
                    wasm_bindgen_futures::JsFuture::from(
                        js_sys::Promise::resolve(&wasm_bindgen::JsValue::from(0))
                    ).await.ok();
                }

                crate::console_log!("Token generation stream completed");
            };
            crate::console_log!("Successfully created async token stream");

            // Wrap the stream to make it Send for WASM
            let boxed_stream = Box::pin(stream);
            let wasm_stream = WasmSendStream::new(boxed_stream);
            Ok(Box::pin(wasm_stream))
        }
    }
}

impl PhiLLMProvider {
    fn messages_to_prompt(&self, messages: &[ChatMessage]) -> String {
        let mut conversation = String::new();

        for message in messages {
            match message.role {
                ChatRole::User => {
                    conversation.push_str(&format!("Alice: {}  \n", message.content));
                }
                ChatRole::Assistant => {
                    conversation.push_str(&format!("Bob: {}  \n", message.content));
                }
                ChatRole::System => {
                    // For system messages, we can include them as context before the conversation
                    conversation = format!("System: {}  \n{}", message.content, conversation);
                }
                _ => {} // Ignore other roles for simplicity
            }
        }

        // Add the Bob prompt at the end for the model to continue
        conversation.push_str("Bob:");
        conversation
    }
}

#[async_trait]
impl CompletionProvider for PhiLLMProvider {
    async fn complete(
        &self,
        req: &CompletionRequest,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        // Convert completion request to a single message
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: req.prompt.clone(),
        }];

        let response = self.chat(&messages, None, None).await?;

        Ok(CompletionResponse {
            text: response.text().unwrap_or_default(),
        })
    }
}

#[async_trait]
impl EmbeddingProvider for PhiLLMProvider {
    async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        // Phi model doesn't support embeddings
        Err(LLMError::Generic(
            "Embeddings not supported by Phi model".to_string(),
        ))
    }
}

#[async_trait]
impl ModelsProvider for PhiLLMProvider {}

impl LLMProvider for PhiLLMProvider {}
