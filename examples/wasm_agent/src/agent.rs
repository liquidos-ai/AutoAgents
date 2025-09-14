use crate::phi_llm_provider::PhiLLMProvider;
use crate::phi_provider::PhiModel;
use autoagents_core::agent::memory::SlidingWindowMemory;
use autoagents_core::agent::task::Task;
use autoagents_core::agent::{
    AgentBuilder, AgentDeriveT, AgentExecutor, AgentOutputT, BaseAgent, Context, DirectAgent,
    ExecutorConfig,
};
use autoagents_core::error::Error;
use autoagents_derive::{agent, AgentHooks};
use autoagents_llm::chat::{ChatMessage, ChatProvider, ChatRole, MessageType};
use autoagents_llm::LLMProvider;
use futures::{Stream, StreamExt};
use std::pin::Pin;
// Removed unused tool imports
use autoagents::async_trait;
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents_burn::model::llama::TinyLlamaBuilder;
use autoagents_core::agent::prebuilt::executor::{BasicAgent, ReActAgent};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

/// A simple chat agent that uses the Phi model for text generation
#[agent(
    name = "tiny_llama_chat_agent",
    description = "A conversational chat agent powered by Phi-1.5 model"
)]
#[derive(Default, Clone, AgentHooks)]
pub struct TInyLLamaChatAgent {}

/// Agent wrapper that provides a simple interface for chat interactions
#[wasm_bindgen]
pub struct TInyLLamaChatWrapper {
    agent: BaseAgent<BasicAgent<TInyLLamaChatAgent>, DirectAgent>,
}

impl TInyLLamaChatWrapper {
    /// Create a new Phi agent with the given model (non-WASM)
    pub async fn new() -> Result<TInyLLamaChatWrapper, JsError> {
        let llm = TinyLlamaBuilder::new()
            .with_model_bytes(vec![])
            .max_seq_len(512)
            .temperature(0.7)
            .max_tokens(256)
            .build()
            .expect("Failed to build LLM");

        let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

        let agent_handle =
            AgentBuilder::<_, DirectAgent>::new(BasicAgent::new(TInyLLamaChatAgent::default()))
                .llm(llm)
                .memory(sliding_window_memory)
                .build()
                .await
                .map_err(|e| JsError::new(&e.to_string()))?;

        println!("Finished Model Loading!");

        Ok(TInyLLamaChatWrapper {
            agent: agent_handle.agent,
        })
    }
}

#[wasm_bindgen]
impl TInyLLamaChatWrapper {
    /// Get a streaming response as individual tokens using LLM provider streaming
    pub async fn get_response_stream(
        &self,
        message: &str,
        callback: &js_sys::Function,
    ) -> Result<(), JsError> {
        use crate::console_log;
        use autoagents_llm::chat::{ChatMessage, ChatRole, MessageType};

        console_log!("Starting get_response_stream with message: {}", message);

        // Use the agent's streaming interface instead of direct LLM access
        let task = Task::new(message);

        console_log!("Starting agent streaming response...");

        // Use the agent's run_stream method to get proper agent execution
        let stream = self
            .agent
            .run_stream(task)
            .await
            .map_err(|e| JsError::new(&format!("Failed to start agent stream: {}", e)))?;

        console_log!("Stream created successfully, processing agent responses...");

        // Stream agent responses and call the callback for each token
        let mut stream_pin = stream;
        while let Some(result) = StreamExt::next(&mut stream_pin).await {
            match result {
                Ok(agent_output) => {
                    // The agent output is a String, but we want to stream it as individual tokens
                    // For now, we'll send the whole response, but this could be enhanced to split into tokens
                    let output_str = agent_output.to_string();
                    console_log!("Received agent output: '{}'", output_str);

                    // Send the output via callback
                    let output_js = JsValue::from_str(&output_str);
                    callback
                        .call1(&JsValue::NULL, &output_js)
                        .map_err(|e| JsError::new(&format!("Callback failed: {:?}", e)))?;

                    // Yield to JavaScript event loop to allow UI updates
                    let promise = js_sys::Promise::resolve(&JsValue::from(0));
                    let js_future = wasm_bindgen_futures::JsFuture::from(promise);
                    js_future.await.ok();
                }
                Err(e) => {
                    console_log!("Stream error: {:?}", e);
                    return Err(JsError::new(&format!("Stream error: {}", e)));
                }
            }
        }

        console_log!("Token generation completed successfully");
        Ok(())
    }
}
