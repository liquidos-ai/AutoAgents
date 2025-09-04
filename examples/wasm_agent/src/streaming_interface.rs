use crate::llama::Model;
use crate::llama_provider::LlamaProvider;
use crate::{console_log, MathAgent, MathAgentOutput};
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::{ReActAgentOutput, ReActExecutor};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{ActorAgent, BaseAgent, DirectAgent};
use autoagents::core::agent::{AgentBuilder, AgentExecutor};
use autoagents::core::protocol::Event;
use futures::channel::mpsc::Receiver;
use futures::StreamExt;
use serde_json;
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{spawn_local, JsFuture};
use web_sys;

#[wasm_bindgen]
pub struct TokenStreamer {
    agent: Arc<BaseAgent<MathAgent, DirectAgent>>,
}

#[wasm_bindgen]
impl TokenStreamer {
    #[wasm_bindgen(constructor)]
    pub fn new(
        weights: Vec<u8>,
        tokenizer: Vec<u8>,
        config: Vec<u8>,
        quantized: bool,
    ) -> Result<TokenStreamer, JsValue> {
        console_log!("Creating token streamer...");
        let model = Model::load(weights, tokenizer, config, quantized)
            .map_err(|e| JsValue::from_str(&format!("Failed to load model: {:?}", e)))?;

        let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));
        let llama_provider = LlamaProvider::new(
            model,
            Some(0.3),  // Lower temperature for more focused responses
            Some(0.9),  // Slightly lower top_p
            Some(1.05), // Lower repeat penalty
            Some(64),
            Some(12345),
        );

        let agent = Arc::new(
            AgentBuilder::<_, DirectAgent>::new(MathAgent {})
                .llm(Arc::new(llama_provider))
                .memory(sliding_window_memory)
                .stream(true)
                .build()
                .unwrap(),
        );

        Ok(TokenStreamer { agent })
    }

    #[wasm_bindgen]
    pub async fn stream_tokens(
        &self,
        prompt: String,
        callback: &js_sys::Function,
    ) -> Result<(), JsValue> {
        console_log!("Starting agent token streaming for prompt: {}", prompt);

        match self
            .agent
            .clone()
            .run_stream(Task::new(prompt))
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to start stream: {:?}", e)))
        {
            Ok(mut stream) => {
                console_log!("Agent stream started successfully");
                console_log!("About to start consuming stream...");

                // Process tokens one by one with proper yielding
                while let Some(result) = stream.next().await {
                    match result {
                        Ok(task_result) => {
                            console_log!("Received task result: {:?}", task_result);
                        }
                        Err(e) => {
                            console_log!("Stream error: {:?}", e);
                            return Err(JsValue::from_str(&format!("Stream error: {:?}", e)));
                        }
                    }
                }

                console_log!("Agent stream completed");
                Ok(())
            }
            Err(e) => {
                console_log!("Failed to start agent stream: {:?}", e);
                Err(e)
            }
        }
    }
}
