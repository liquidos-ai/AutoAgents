use crate::phi::Model;
use crate::phi_provider::PhiProvider;
use crate::{console_log, MathAgent, MathAgentOutput};
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::{ReActAgentOutput, ReActExecutor};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::RunnableAgentImpl;
use autoagents::core::agent::{AgentBuilder, AgentExecutor, RunnableAgent};
use autoagents::core::protocol::{Event, TaskResult};
use futures::channel::mpsc::Receiver;
use futures::StreamExt;
use serde_json;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct TokenStreamer {
    agent: Arc<RunnableAgentImpl<MathAgent>>,
    rx: Receiver<Event>,
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
        let phi_provider = PhiProvider::new(model, Some(0.7), Some(0.9), Some(1.0), Some(42), None);

        let (agent, rx) = AgentBuilder::new(MathAgent {})
            .with_llm(Arc::new(phi_provider))
            .with_memory(sliding_window_memory)
            .stream(true)
            .build_runnable()
            .map_err(|e| JsValue::from_str(&format!("Failed to build agent: {:?}", e)))?;

        Ok(TokenStreamer { agent, rx })
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

                while let Some(result) = stream.next().await {
                    console_log!("Got strem");
                    match result {
                        Ok(task_result) => {
                            console_log!("Received task result: {:?}", task_result);

                            match task_result {
                                TaskResult::Value(val) => {
                                    // Try to parse as ReActAgentOutput for streaming response
                                    match serde_json::from_value::<ReActAgentOutput>(val.clone()) {
                                        Ok(agent_out) => {
                                            if !agent_out.done {
                                                // Streaming response - send the token
                                                console_log!(
                                                    "Streaming response: {}",
                                                    agent_out.response
                                                );
                                                let js_token =
                                                    JsValue::from_str(&agent_out.response);
                                                if let Err(e) =
                                                    callback.call1(&JsValue::NULL, &js_token)
                                                {
                                                    console_log!("Callback error: {:?}", e);
                                                    return Err(JsValue::from_str(
                                                        "Callback failed",
                                                    ));
                                                }
                                            } else {
                                                // Final response - end streaming
                                                console_log!(
                                                    "Final response: {}",
                                                    agent_out.response
                                                );
                                                let js_response =
                                                    JsValue::from_str(&agent_out.response);
                                                if let Err(e) =
                                                    callback.call1(&JsValue::NULL, &js_response)
                                                {
                                                    console_log!("Callback error: {:?}", e);
                                                    return Err(JsValue::from_str(
                                                        "Callback failed",
                                                    ));
                                                }
                                                break; // End streaming
                                            }
                                        }
                                        Err(_) => {
                                            // Try to parse as MathAgentOutput (final structured output)
                                            match serde_json::from_value::<MathAgentOutput>(
                                                val.clone(),
                                            ) {
                                                Ok(math_out) => {
                                                    console_log!(
                                                        "Math output - Value: {}, Explanation: {}",
                                                        math_out.value,
                                                        math_out.explanation
                                                    );
                                                    let final_response = format!(
                                                        "Value: {}\nExplanation: {}",
                                                        math_out.value, math_out.explanation
                                                    );
                                                    let js_response =
                                                        JsValue::from_str(&final_response);
                                                    if let Err(e) =
                                                        callback.call1(&JsValue::NULL, &js_response)
                                                    {
                                                        console_log!("Callback error: {:?}", e);
                                                        return Err(JsValue::from_str(
                                                            "Callback failed",
                                                        ));
                                                    }
                                                    break; // End streaming
                                                }
                                                Err(e) => {
                                                    console_log!("Failed to parse as any known output type: {:?}, raw value: {:?}", e, val);
                                                    // Send raw value as fallback
                                                    let response_str = val.to_string();
                                                    let js_response =
                                                        JsValue::from_str(&response_str);
                                                    if let Err(e) =
                                                        callback.call1(&JsValue::NULL, &js_response)
                                                    {
                                                        console_log!("Callback error: {:?}", e);
                                                        return Err(JsValue::from_str(
                                                            "Callback failed",
                                                        ));
                                                    }
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                                TaskResult::Failure(error) => {
                                    console_log!("Task failed: {}", error);
                                    return Err(JsValue::from_str(&format!(
                                        "Task failed: {}",
                                        error
                                    )));
                                }
                                _ => {
                                    console_log!("Unexpected task result type: {:?}", task_result);
                                    continue;
                                }
                            }
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
