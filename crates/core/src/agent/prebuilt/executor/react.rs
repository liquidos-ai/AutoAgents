use crate::agent::executor::AgentExecutor;
use crate::agent::memory::MemoryProvider;
use crate::agent::task::Task;
use crate::agent::{Context, ExecutorConfig, TurnResult};
use crate::protocol::{Event, StreamingTurnResult, SubmissionId};
use crate::tool::{ToolCallResult, ToolT};
use async_trait::async_trait;
use autoagents_llm::chat::{ChatMessage, ChatRole, MessageType, Tool};
use autoagents_llm::{FunctionCall, ToolCall};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::pin::Pin;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::mpsc::error::SendError;
use tokio::sync::{mpsc, RwLock};

// Output of the ReAct-style agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReActAgentOutput {
    pub response: String,
    pub tool_calls: Vec<ToolCallResult>,
}

impl From<ReActAgentOutput> for Value {
    fn from(output: ReActAgentOutput) -> Self {
        serde_json::to_value(output).unwrap_or(Value::Null)
    }
}

impl ReActAgentOutput {
    /// Extract the agent output from the ReAct response
    pub fn extract_agent_output<T>(val: Value) -> Result<T, ReActExecutorError>
    where
        T: for<'de> serde::Deserialize<'de>,
    {
        let react_output: Self = serde_json::from_value(val)
            .map_err(|e| ReActExecutorError::AgentOutputError(e.to_string()))?;
        serde_json::from_str(&react_output.response)
            .map_err(|e| ReActExecutorError::AgentOutputError(e.to_string()))
    }
}

#[derive(Error, Debug)]
pub enum ReActExecutorError {
    #[error("LLM error: {0}")]
    LLMError(String),

    #[error("Maximum turns exceeded: {max_turns}")]
    MaxTurnsExceeded { max_turns: usize },

    #[error("Other error: {0}")]
    Other(String),

    #[error("Event error: {0}")]
    EventError(#[from] SendError<Event>),

    #[error("Extracting Agent Output Error: {0}")]
    AgentOutputError(String),
}

#[async_trait]
pub trait ReActExecutor: Send + Sync + Clone + 'static {
    async fn process_tool_calls(
        &self,
        tools: &[Box<dyn ToolT>],
        tool_calls: Vec<ToolCall>,
        tx_event: mpsc::Sender<Event>,
        _memory: Option<Arc<RwLock<Box<dyn MemoryProvider>>>>,
    ) -> Vec<ToolCallResult> {
        let mut results = Vec::new();

        for call in &tool_calls {
            let tool_name = call.function.name.clone();
            let tool_args = call.function.arguments.clone();

            let result = match tools.iter().find(|t| t.name() == tool_name) {
                Some(tool) => {
                    let _ = tx_event
                        .send(Event::ToolCallRequested {
                            id: call.id.clone(),
                            tool_name: tool_name.clone(),
                            arguments: tool_args.clone(),
                        })
                        .await;

                    match serde_json::from_str::<Value>(&tool_args) {
                        Ok(parsed_args) => match tool.run(parsed_args) {
                            Ok(output) => ToolCallResult {
                                tool_name: tool_name.clone(),
                                success: true,
                                arguments: serde_json::from_str(&tool_args).unwrap_or(Value::Null),
                                result: output,
                            },
                            Err(e) => ToolCallResult {
                                tool_name: tool_name.clone(),
                                success: false,
                                arguments: serde_json::from_str(&tool_args).unwrap_or(Value::Null),
                                result: serde_json::json!({"error": e.to_string()}),
                            },
                        },
                        Err(e) => ToolCallResult {
                            tool_name: tool_name.clone(),
                            success: false,
                            arguments: Value::Null,
                            result: serde_json::json!({"error": format!("Failed to parse arguments: {}", e)}),
                        },
                    }
                }
                None => ToolCallResult {
                    tool_name: tool_name.clone(),
                    success: false,
                    arguments: serde_json::from_str(&tool_args).unwrap_or(Value::Null),
                    result: serde_json::json!({"error": format!("Tool '{}' not found", tool_name)}),
                },
            };

            if result.success {
                let _ = tx_event
                    .send(Event::ToolCallCompleted {
                        id: call.id.clone(),
                        tool_name: tool_name.clone(),
                        result: result.result.clone(),
                    })
                    .await;
            } else {
                let _ = tx_event
                    .send(Event::ToolCallFailed {
                        id: call.id.clone(),
                        tool_name: tool_name.clone(),
                        error: result.result.to_string(),
                    })
                    .await;
            }

            results.push(result);
        }

        results
    }

    async fn process_turn(
        &self,
        context: &Context,
        tools: &[Box<dyn ToolT>],
    ) -> Result<TurnResult<ReActAgentOutput>, ReActExecutorError> {
        let llm = context.llm();
        let agent_config = context.config();
        let messages = context.messages();
        let memory = context.memory();
        let tx_event = context.tx();

        let response = if !tools.is_empty() {
            let tools_serialized: Vec<Tool> = tools.iter().map(Tool::from).collect();
            llm.chat_with_tools(
                messages,
                Some(&tools_serialized),
                agent_config.output_schema.clone(),
            )
            .await
            .map_err(|e| ReActExecutorError::LLMError(e.to_string()))?
        } else {
            llm.chat(messages, agent_config.output_schema.clone())
                .await
                .map_err(|e| ReActExecutorError::LLMError(e.to_string()))?
        };

        let response_text = response.text().unwrap_or_default();
        if let Some(tool_calls) = response.tool_calls() {
            let tool_results = self
                .process_tool_calls(tools, tool_calls.clone(), tx_event.clone(), memory.clone())
                .await;

            // Store tool calls and results in memory
            if let Some(mem) = &memory {
                let mut mem = mem.write().await;

                // Record that assistant is calling tools
                let _ = mem
                    .remember(&ChatMessage {
                        role: ChatRole::Assistant,
                        message_type: MessageType::ToolUse(tool_calls.clone()),
                        content: response_text.clone(),
                    })
                    .await;

                // Create ToolCall objects with the results
                let mut result_tool_calls = Vec::new();
                for (tool_call, result) in tool_calls.iter().zip(&tool_results) {
                    let result_content = if result.success {
                        match &result.result {
                            serde_json::Value::String(s) => s.clone(),
                            other => serde_json::to_string(other).unwrap_or_default(),
                        }
                    } else {
                        serde_json::json!({"error": format!("{:?}", result.result)}).to_string()
                    };

                    result_tool_calls.push(ToolCall {
                        id: tool_call.id.clone(),
                        call_type: tool_call.call_type.clone(),
                        function: FunctionCall {
                            name: tool_call.function.name.clone(),
                            arguments: result_content,
                        },
                    });
                }

                // Store tool results
                let _ = mem
                    .remember(&ChatMessage {
                        role: ChatRole::Tool,
                        message_type: MessageType::ToolResult(result_tool_calls),
                        content: String::new(),
                    })
                    .await;
            }

            let state = context.state();
            let mut guard = state.write().await;
            for result in &tool_results {
                guard.record_tool_call(result.clone());
            }

            Ok(TurnResult::Continue(Some(ReActAgentOutput {
                response: response_text,
                tool_calls: tool_results,
            })))
        } else {
            // Record the final response in memory
            if !response_text.is_empty() {
                if let Some(mem) = &memory {
                    let mut mem = mem.write().await;
                    let _ = mem
                        .remember(&ChatMessage {
                            role: ChatRole::Assistant,
                            message_type: MessageType::Text,
                            content: response_text.clone(),
                        })
                        .await;
                }
            }

            Ok(TurnResult::Complete(ReActAgentOutput {
                response: response_text,
                tool_calls: vec![],
            }))
        }
    }

    /// Process a streaming turn with tool support using a hybrid approach
    async fn process_streaming_turn_hybrid(
        &self,
        context: &Context,
        tools: &[Box<dyn ToolT>],
        accumulated_tool_calls: &[ToolCallResult],
        tx: &mpsc::Sender<Result<ReActAgentOutput, ReActExecutorError>>,
        submission_id: SubmissionId,
    ) -> Result<StreamingTurnResult, ReActExecutorError> {
        let messages = self.prepare_messages(context).await;
        let tools_serialized: Vec<Tool> = tools.iter().map(Tool::from).collect();
        let tools_for_streaming = if !tools.is_empty() {
            Some(&tools_serialized[..])
        } else {
            None
        };

        // First, stream the response for real-time updates
        let mut stream = context
            .llm()
            .chat_stream_with_tools(&messages, tools_for_streaming)
            .await
            .map_err(|e| ReActExecutorError::LLMError(e.to_string()))?;

        let mut response_text = String::new();

        // Collect streaming chunks
        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    response_text.push_str(&chunk);

                    // Send streaming update
                    let _ = context
                        .tx()
                        .send(Event::StreamChunk {
                            sub_id: submission_id,
                            chunk: chunk.clone(),
                        })
                        .await;

                    // Send intermediate result
                    let _ = tx
                        .send(Ok(ReActAgentOutput {
                            response: response_text.clone(),
                            tool_calls: accumulated_tool_calls.to_vec(),
                        }))
                        .await;
                }
                Err(e) => {
                    return Err(ReActExecutorError::LLMError(e.to_string()));
                }
            }
        }

        // After streaming, make a non-streaming call to check for tool calls
        // This is necessary because streaming doesn't return tool call information
        if !tools.is_empty() {
            // Add the streamed response to messages for context
            let mut check_messages = messages.clone();
            check_messages.push(ChatMessage {
                role: ChatRole::Assistant,
                message_type: MessageType::Text,
                content: response_text.clone(),
            });

            // Make a quick non-streaming call to detect tool calls
            let tool_check_response = context
                .llm()
                .chat_with_tools(&messages, Some(&tools_serialized), None)
                .await
                .map_err(|e| ReActExecutorError::LLMError(e.to_string()))?;

            if let Some(tool_calls) = tool_check_response.tool_calls() {
                // Emit streaming tool call events
                for tool_call in &tool_calls {
                    let _ = context
                        .tx()
                        .send(Event::StreamToolCall {
                            sub_id: submission_id,
                            tool_call: serde_json::to_value(tool_call)
                                .unwrap_or(serde_json::Value::Null),
                        })
                        .await;
                }

                // Process tool calls
                let tool_results = self
                    .process_tool_calls(
                        tools,
                        tool_calls.clone(),
                        context.tx().clone(),
                        context.memory(),
                    )
                    .await;

                // Update memory with tool calls and results
                self.update_memory_with_tools(
                    context.memory(),
                    &tool_calls,
                    &tool_results,
                    &response_text,
                )
                .await;

                // Update state
                let state = context.state();
                let mut guard = state.write().await;
                for result in &tool_results {
                    guard.record_tool_call(result.clone());
                }

                return Ok(StreamingTurnResult::ToolCallsProcessed(tool_results));
            }
        }

        // No tool calls detected, record the response
        if !response_text.is_empty() {
            if let Some(mem) = context.memory() {
                let mut mem = mem.write().await;
                let _ = mem
                    .remember(&ChatMessage {
                        role: ChatRole::Assistant,
                        message_type: MessageType::Text,
                        content: response_text.clone(),
                    })
                    .await;
            }
        }

        Ok(StreamingTurnResult::Complete(response_text))
    }

    /// Prepare messages for the current turn
    async fn prepare_messages(&self, context: &Context) -> Vec<ChatMessage> {
        let mut messages = vec![ChatMessage {
            role: ChatRole::System,
            message_type: MessageType::Text,
            content: context.config().description.clone(),
        }];

        if let Some(memory) = context.memory() {
            if let Ok(recalled) = memory.read().await.recall("", None).await {
                messages.extend(recalled);
            }
        }

        messages
    }

    /// Update memory with tool calls and results
    async fn update_memory_with_tools(
        &self,
        memory: Option<Arc<RwLock<Box<dyn MemoryProvider>>>>,
        tool_calls: &[ToolCall],
        tool_results: &[ToolCallResult],
        response_text: &str,
    ) {
        if let Some(mem) = memory {
            let mut mem = mem.write().await;

            // Record assistant calling tools
            let _ = mem
                .remember(&ChatMessage {
                    role: ChatRole::Assistant,
                    message_type: MessageType::ToolUse(tool_calls.to_vec()),
                    content: response_text.to_string(),
                })
                .await;

            // Create tool result messages
            let mut result_tool_calls = Vec::new();
            for (tool_call, result) in tool_calls.iter().zip(tool_results) {
                let result_content = if result.success {
                    match &result.result {
                        serde_json::Value::String(s) => s.clone(),
                        other => serde_json::to_string(other).unwrap_or_default(),
                    }
                } else {
                    serde_json::json!({"error": format!("{:?}", result.result)}).to_string()
                };

                result_tool_calls.push(ToolCall {
                    id: tool_call.id.clone(),
                    call_type: tool_call.call_type.clone(),
                    function: FunctionCall {
                        name: tool_call.function.name.clone(),
                        arguments: result_content,
                    },
                });
            }

            // Store tool results
            let _ = mem
                .remember(&ChatMessage {
                    role: ChatRole::Tool,
                    message_type: MessageType::ToolResult(result_tool_calls),
                    content: String::new(),
                })
                .await;
        }
    }
}

#[async_trait]
impl<T: ReActExecutor> AgentExecutor for T {
    type Output = ReActAgentOutput;
    type Error = ReActExecutorError;

    fn config(&self) -> ExecutorConfig {
        ExecutorConfig { max_turns: 10 }
    }

    async fn execute(
        &self,
        task: &Task,
        context: Arc<Context>,
    ) -> Result<Self::Output, Self::Error> {
        let task = task.clone();
        let max_turns = self.config().max_turns;
        let mut accumulated_tool_calls = Vec::new();
        let mut final_response = String::new();

        let llm = context.llm();
        let mut memory = context.memory();
        let tools = context.tools();
        let agent_config = context.config();
        let tx_event = context.tx();

        if let Some(memory) = &mut memory {
            let mut mem = memory.write().await;
            let chat_msg = ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: task.prompt.clone(),
            };
            let _ = mem.remember(&chat_msg).await;
        }

        // Record the task in state
        let state = context.state();
        let mut state = state.write().await;
        state.record_task(task.clone());

        tx_event
            .send(Event::TaskStarted {
                sub_id: task.submission_id,
                actor_id: agent_config.id,
                task_description: task.prompt,
            })
            .await
            .map_err(ReActExecutorError::EventError)?;

        for _ in 0..max_turns {
            let mut messages = vec![ChatMessage {
                role: ChatRole::System,
                message_type: MessageType::Text,
                content: agent_config.description.clone(),
            }];

            if let Some(memory) = &memory {
                if let Ok(recalled) = memory.read().await.recall("", None).await {
                    messages.extend(recalled);
                }
            }

            let turn_context = Context::new(llm.clone(), tx_event.clone())
                .with_memory(memory.clone())
                .with_config(agent_config.clone())
                .with_messages(messages)
                // .with_state(state.clone())
                .with_stream(context.stream());

            match self.process_turn(&turn_context, tools).await? {
                TurnResult::Complete(result) => {
                    if !accumulated_tool_calls.is_empty() {
                        return Ok(ReActAgentOutput {
                            response: result.response,
                            tool_calls: accumulated_tool_calls,
                        });
                    }
                    return Ok(result);
                }
                TurnResult::Continue(Some(partial_result)) => {
                    accumulated_tool_calls.extend(partial_result.tool_calls);
                    if !partial_result.response.is_empty() {
                        final_response = partial_result.response;
                    }
                    continue;
                }
                TurnResult::Continue(None) => {
                    continue;
                }
            }
        }

        if !final_response.is_empty() || !accumulated_tool_calls.is_empty() {
            Ok(ReActAgentOutput {
                response: final_response,
                tool_calls: accumulated_tool_calls,
            })
        } else {
            Err(ReActExecutorError::MaxTurnsExceeded { max_turns })
        }
    }

    async fn execute_stream(
        &self,
        task: &Task,
        context: Arc<Context>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Self::Output, Self::Error>> + Send>>, Self::Error>
    {
        let submission_id = task.submission_id;
        let task_prompt = task.prompt.clone();
        let max_turns = self.config().max_turns;

        // Initialize memory with the task
        if let Some(mem) = &context.memory() {
            let mut mem = mem.write().await;
            let _ = mem
                .remember(&ChatMessage {
                    role: ChatRole::User,
                    message_type: MessageType::Text,
                    content: task_prompt.clone(),
                })
                .await;
        }

        // Record task in state
        let state = context.state();
        let mut state = state.write().await;
        state.record_task(task.clone());

        // Send task started event
        let _ = context
            .tx()
            .send(Event::TaskStarted {
                sub_id: submission_id,
                actor_id: context.config().id,
                task_description: task_prompt.clone(),
            })
            .await;

        // Create channel for streaming results
        let (tx, rx) = mpsc::channel::<Result<ReActAgentOutput, ReActExecutorError>>(100);

        // Clone necessary components for the async task
        let executor = self.clone();
        let context_clone = context.clone();

        // Spawn the streaming task
        tokio::spawn(async move {
            let mut accumulated_tool_calls = Vec::new();
            let mut final_response = String::new();
            let tools = context_clone.tools();

            for turn in 0..max_turns {
                // Send turn started event
                let _ = context_clone
                    .tx()
                    .send(Event::TurnStarted {
                        turn_number: turn,
                        max_turns,
                    })
                    .await;

                // Build context for this turn
                let turn_context = Context::new(context_clone.llm(), context_clone.tx().clone())
                    .with_memory(context_clone.memory())
                    .with_config(context_clone.config().clone())
                    // .with_tools(tools.to_vec())
                    // .with_state(context_clone.state())
                    .with_stream(true);

                // Process streaming turn with hybrid approach
                match executor
                    .process_streaming_turn_hybrid(
                        &turn_context,
                        tools,
                        &accumulated_tool_calls,
                        &tx,
                        submission_id,
                    )
                    .await
                {
                    Ok(StreamingTurnResult::Complete(response)) => {
                        final_response = response;

                        // Send turn completed event
                        let _ = context_clone
                            .tx()
                            .send(Event::TurnCompleted {
                                turn_number: turn,
                                final_turn: true,
                            })
                            .await;

                        break;
                    }
                    Ok(StreamingTurnResult::ToolCallsProcessed(tool_results)) => {
                        // Accumulate tool results
                        accumulated_tool_calls.extend(tool_results);

                        // Send updated result with tool calls
                        let _ = tx
                            .send(Ok(ReActAgentOutput {
                                response: String::new(),
                                tool_calls: accumulated_tool_calls.clone(),
                            }))
                            .await;

                        // Send turn completed event
                        let _ = context_clone
                            .tx()
                            .send(Event::TurnCompleted {
                                turn_number: turn,
                                final_turn: false,
                            })
                            .await;

                        // Continue to next turn for final response after tool calls
                        continue;
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        return;
                    }
                }
            }

            // Send stream complete event
            let _ = context_clone
                .tx()
                .send(Event::StreamComplete {
                    sub_id: submission_id,
                })
                .await;

            // Send final result
            let _ = tx
                .send(Ok(ReActAgentOutput {
                    response: final_response,
                    tool_calls: accumulated_tool_calls,
                }))
                .await;
        });

        // Return the stream
        Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestAgentOutput {
        value: i32,
        message: String,
    }

    #[test]
    fn test_extract_agent_output_success() {
        let agent_output = TestAgentOutput {
            value: 42,
            message: "Hello, world!".to_string(),
        };

        let react_output = ReActAgentOutput {
            response: serde_json::to_string(&agent_output).unwrap(),
            tool_calls: vec![],
        };

        let react_value = serde_json::to_value(react_output).unwrap();
        let extracted: TestAgentOutput =
            ReActAgentOutput::extract_agent_output(react_value).unwrap();
        assert_eq!(extracted, agent_output);
    }
}
