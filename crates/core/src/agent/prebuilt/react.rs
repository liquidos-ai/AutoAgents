use crate::agent::base::AgentConfig;
use crate::agent::executor::{AgentExecutor, ExecutorConfig, TurnResult};
use crate::agent::runnable::AgentState;
use crate::agent::ActorTask;
use crate::memory::MemoryProvider;
use crate::protocol::Event;
use crate::runtime::Task;
use crate::tool::{ToolCallResult, ToolT};
use async_trait::async_trait;
use autoagents_llm::chat::{ChatMessage, ChatRole, MessageType, Tool};
use autoagents_llm::{LLMProvider, ToolCall};
use log::{debug, error};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::mpsc::error::SendError;
use tokio::sync::{mpsc, RwLock};

/// Output of the ReAct-style agent
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
    /// This parses the response string as JSON and deserializes it to the target type
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

    #[error("Tool execution error: {0}")]
    ToolError(String),

    #[error("Maximum turns exceeded: {max_turns}")]
    MaxTurnsExceeded { max_turns: usize },

    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Other error: {0}")]
    Other(String),

    #[error("Event error: {0}")]
    EventError(#[from] SendError<Event>),

    #[error("Extracting Agent Output Error: {0}")]
    AgentOutputError(String),
}

#[async_trait]
pub trait ReActExecutor: Send + Sync + 'static {
    async fn process_tool_calls(
        &self,
        tools: &[Box<dyn ToolT>],
        tool_calls: Vec<autoagents_llm::ToolCall>,
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

    #[allow(clippy::too_many_arguments)]
    async fn process_turn(
        &self,
        llm: Arc<dyn LLMProvider>,
        messages: &[ChatMessage],
        memory: Option<Arc<RwLock<Box<dyn MemoryProvider>>>>,
        tools: &[Box<dyn ToolT>],
        agent_config: &AgentConfig,
        state: Arc<RwLock<AgentState>>,
        tx_event: mpsc::Sender<Event>,
    ) -> Result<TurnResult<ReActAgentOutput>, ReActExecutorError> {
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

                // Create ToolCall objects with the results for ToolResult message type
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

                    // Create a new ToolCall with the result in the arguments field
                    result_tool_calls.push(ToolCall {
                        id: tool_call.id.clone(),
                        call_type: tool_call.call_type.clone(),
                        function: autoagents_llm::FunctionCall {
                            name: tool_call.function.name.clone(),
                            arguments: result_content,
                        },
                    });
                }

                // Store tool results using ToolResult message type with Tool role
                let _ = mem
                    .remember(&ChatMessage {
                        role: ChatRole::Tool,
                        message_type: MessageType::ToolResult(result_tool_calls),
                        content: String::new(),
                    })
                    .await;
            }

            {
                let mut guard = state.write().await;
                for result in &tool_results {
                    guard.record_tool_call(result.clone());
                }
            }

            // Continue to let the LLM generate a response based on tool results
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
        llm: Arc<dyn LLMProvider>,
        mut memory: Option<Arc<RwLock<Box<dyn MemoryProvider>>>>,
        tools: Vec<Box<dyn ToolT>>,
        agent_config: &AgentConfig,
        task: Box<dyn ActorTask>,
        state: Arc<RwLock<AgentState>>,
        tx_event: mpsc::Sender<Event>,
        _stream: bool,
    ) -> Result<Self::Output, Self::Error> {
        debug!("Starting ReAct Executor");
        let max_turns = self.config().max_turns;
        let mut accumulated_tool_calls = Vec::new();
        let mut final_response = String::new();

        // Downcast the Box<dyn ActorTask> to Task
        let task = task
            .as_any()
            .downcast_ref::<Task>()
            .expect("Expected Task type")
            .clone();

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
        {
            let mut state = state.write().await;
            state.record_task(task.clone());
        }

        tx_event
            .send(Event::TaskStarted {
                sub_id: task.submission_id,
                agent_id: agent_config.id,
                task_description: task.prompt,
            })
            .await?;

        for turn in 0..max_turns {
            //Prepare messages with memory
            let mut messages = vec![ChatMessage {
                role: ChatRole::System,
                message_type: MessageType::Text,
                content: agent_config.description.clone(),
            }];
            if let Some(memory) = &memory {
                // Fetch All previous messsages and extend
                messages.extend(
                    memory
                        .read()
                        .await
                        .recall("", None)
                        .await
                        .unwrap_or_default(),
                );
            }

            tx_event
                .send(Event::TurnStarted {
                    turn_number: turn,
                    max_turns,
                })
                .await?;
            match self
                .process_turn(
                    llm.clone(),
                    &messages,
                    memory.clone(),
                    &tools,
                    agent_config,
                    state.clone(),
                    tx_event.clone(),
                )
                .await?
            {
                TurnResult::Complete(result) => {
                    // If we have accumulated tool calls, merge them with the final result
                    if !accumulated_tool_calls.is_empty() {
                        tx_event
                            .send(Event::TurnCompleted {
                                turn_number: turn,
                                final_turn: true,
                            })
                            .await?;
                        return Ok(ReActAgentOutput {
                            response: result.response,
                            tool_calls: accumulated_tool_calls,
                        });
                    }
                    tx_event
                        .send(Event::TurnCompleted {
                            turn_number: turn,
                            final_turn: true,
                        })
                        .await?;
                    return Ok(result);
                }
                TurnResult::Continue(Some(partial_result)) => {
                    // Accumulate tool calls and continue for final response
                    accumulated_tool_calls.extend(partial_result.tool_calls);
                    if !partial_result.response.is_empty() {
                        final_response = partial_result.response;
                    }
                    tx_event
                        .send(Event::TurnCompleted {
                            turn_number: turn,
                            final_turn: false,
                        })
                        .await?;
                    continue;
                }
                TurnResult::Continue(None) => {
                    tx_event
                        .send(Event::TurnCompleted {
                            turn_number: turn,
                            final_turn: false,
                        })
                        .await?;
                    continue;
                }
            }
        }

        // If we've exhausted turns but have results, return what we have
        if !final_response.is_empty() || !accumulated_tool_calls.is_empty() {
            Ok(ReActAgentOutput {
                response: final_response,
                tool_calls: accumulated_tool_calls,
            })
        } else {
            Err(ReActExecutorError::MaxTurnsExceeded { max_turns })
        }
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
