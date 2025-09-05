use crate::phi_llm_provider::PhiLLMProvider;
use crate::phi_provider::PhiModel;
use autoagents_core::agent::memory::SlidingWindowMemory;
use autoagents_core::agent::task::Task;
use autoagents_core::agent::{
    AgentBuilder, AgentDeriveT, AgentExecutor, AgentOutputT, BaseAgent, Context, DirectAgent,
    ExecutorConfig,
};
use autoagents_core::error::Error;
use autoagents_derive::agent;
use autoagents_llm::chat::{ChatMessage, ChatProvider, ChatRole, MessageType};
use autoagents_llm::LLMProvider;
use futures::{Stream, StreamExt};
use std::pin::Pin;
// Removed unused tool imports
use async_trait::async_trait;
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

/// A simple chat agent that uses the Phi model for text generation
#[agent(
    name = "phi_chat_agent",
    description = "A conversational chat agent powered by Phi-1.5 model"
)]
#[derive(Default, Clone)]
pub struct PhiChatAgent {}

#[async_trait]
impl AgentExecutor for PhiChatAgent {
    type Output = String;
    type Error = Error;

    fn config(&self) -> ExecutorConfig {
        ExecutorConfig { max_turns: 10 }
    }

    async fn execute(
        &self,
        task: &Task,
        context: Arc<Context>,
    ) -> Result<Self::Output, Self::Error> {
        unimplemented!()
    }

    async fn execute_stream(
        &self,
        task: &Task,
        context: Arc<Context>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Self::Output, Self::Error>> + Send>>, Self::Error>
    {
        let mut messages = vec![ChatMessage {
            role: ChatRole::System,
            message_type: MessageType::Text,
            content: context.config().description.clone(),
        }];

        let chat_msg = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: task.prompt.clone(),
        };
        messages.push(chat_msg);

        let llm_stream = context
            .llm()
            .chat_stream(&messages, None, context.config().output_schema.clone())
            .await
            .map_err(|e| Error::LLMError(e))?;

        // Convert LLMError to Error in the stream
        let mapped_stream = llm_stream.map(|result| result.map_err(|e| Error::LLMError(e)));

        Ok(Box::pin(mapped_stream))
    }
}

/// Agent wrapper that provides a simple interface for chat interactions
#[wasm_bindgen]
pub struct PhiAgentWrapper {
    agent: BaseAgent<PhiChatAgent, DirectAgent>,
}

impl PhiAgentWrapper {
    /// Create a new Phi agent with the given model (non-WASM)
    pub fn new(phi_llm_provider: PhiLLMProvider) -> Result<PhiAgentWrapper, JsError> {
        let llm: Arc<dyn LLMProvider> = Arc::new(phi_llm_provider);
        let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

        let agent = AgentBuilder::<_, DirectAgent>::new(PhiChatAgent::default())
            .llm(llm)
            .memory(sliding_window_memory)
            .build()
            .map_err(|e| JsError::new(&e.to_string()))?;

        Ok(PhiAgentWrapper { agent })
    }
}

#[wasm_bindgen]
impl PhiAgentWrapper {
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

    /// Create a new Phi agent from a PhiModel (WASM-friendly factory function)
    pub fn from_phi_model(phi_model: PhiModel) -> Result<PhiAgentWrapper, JsError> {
        let llm_provider = PhiLLMProvider::new(phi_model);
        Self::new(llm_provider)
    }
}
