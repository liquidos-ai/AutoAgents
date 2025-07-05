use crate::agent::base::AgentConfig;
use crate::agent::executor::{AgentExecutor, ExecutorConfig, TurnResult};
use crate::agent::runnable::AgentState;
use crate::memory::MemoryProvider;
use crate::protocol::Event;
use crate::session::Task;
use crate::tool::ToolCallResult;
use async_trait::async_trait;
use autoagents_llm::chat::{ChatMessage, ChatRole, MessageType, Tool};
use autoagents_llm::{LLMProvider, ToolT};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::{mpsc, RwLock};

/// The default agent output structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentOutput {
    /// The agent's response
    pub response: String,
    /// Any tool calls made during execution
    pub tool_calls: Vec<ToolCallResult>,
}

impl From<AgentOutput> for Value {
    fn from(output: AgentOutput) -> Self {
        serde_json::to_value(output).unwrap_or(Value::Null)
    }
}

/// Error types for the default executor
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
}

/// A ReAct-style executor that handles tool calls and conversation flow
#[derive(Debug, Default)]
pub struct ReActExecutor {
    config: ExecutorConfig,
}

impl ReActExecutor {
    pub fn with_config(mut self, config: ExecutorConfig) -> Self {
        self.config = config;
        self
    }

    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.config.max_turns = max_turns;
        self
    }

    /// Process tool calls from the LLM response
    async fn process_tool_calls(
        &self,
        tools: &[Arc<Box<dyn ToolT>>],
        tool_calls: Vec<autoagents_llm::ToolCall>,
        tx_event: mpsc::Sender<Event>,
    ) -> Vec<ToolCallResult> {
        let mut results = Vec::new();

        for tc in tool_calls {
            let tool_name = tc.function.name.clone();
            let tool_args = tc.function.arguments.clone();

            // Find the matching tool
            if let Some(tool) = tools.iter().find(|t| t.name() == tool_name) {
                let _ = tx_event
                    .send(Event::ToolCallRequested {
                        id: tc.id.clone(),
                        tool_name: tool_name.clone(),
                        arguments: tool_args.clone(),
                    })
                    .await;

                // Parse the arguments and execute the tool
                let result = match serde_json::from_str::<serde_json::Value>(&tool_args) {
                    Ok(parsed_args) => match tool.run(parsed_args) {
                        Ok(output) => ToolCallResult {
                            tool_name: tool_name.clone(),
                            success: true,
                            arguments: serde_json::from_str(&tool_args)
                                .unwrap_or(serde_json::Value::Null),
                            result: output,
                        },
                        Err(e) => ToolCallResult {
                            tool_name: tool_name.clone(),
                            success: false,
                            arguments: serde_json::from_str(&tool_args)
                                .unwrap_or(serde_json::Value::Null),
                            result: serde_json::json!({
                                "error": e.to_string()
                            }),
                        },
                    },
                    Err(e) => ToolCallResult {
                        tool_name: tool_name.clone(),
                        success: false,
                        arguments: serde_json::Value::Null,
                        result: serde_json::json!({
                            "error": format!("Failed to parse arguments: {}", e)
                        }),
                    },
                };

                let _ = tx_event
                    .send(Event::ToolCallCompleted {
                        id: tc.id.clone(),
                        tool_name: tool_name.clone(),
                        result: result.result.clone(),
                    })
                    .await;

                results.push(result);
            } else {
                // Tool not found
                let result = ToolCallResult {
                    tool_name: tool_name.clone(),
                    success: false,
                    arguments: serde_json::from_str(&tool_args).unwrap_or(serde_json::Value::Null),
                    result: serde_json::json!({
                        "error": format!("Tool '{}' not found", tool_name)
                    }),
                };
                results.push(result);
            }
        }

        results
    }
}

#[async_trait]
impl AgentExecutor for ReActExecutor {
    type Output = AgentOutput;
    type Error = ReActExecutorError;

    fn config(&self) -> &ExecutorConfig {
        &self.config
    }

    async fn process_turn(
        &self,
        llm: Arc<dyn LLMProvider>,
        memory: Option<Arc<RwLock<Box<dyn MemoryProvider>>>>,
        tools: &[Arc<Box<dyn ToolT>>],
        agent_config: &AgentConfig,
        task: &Task,
        state: Arc<RwLock<AgentState>>,
        turn_number: usize,
        tx_event: mpsc::Sender<Event>,
    ) -> Result<TurnResult<Self::Output>, Self::Error> {
        // Build conversation history
        let mut messages = vec![ChatMessage {
            role: ChatRole::Assistant,
            message_type: MessageType::Text,
            content: agent_config.description.clone(),
        }];

        // Add conversation history from state
        if let Some(memory) = memory {
            let history = memory.read().await.recall("", None).await.unwrap();
            messages.extend(history);
        }

        // Add the current task if this is the first turn
        if turn_number == 0 {
            messages.push(ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: task.prompt.clone(),
            });
        }

        // Make LLM call with tools
        let response = if !tools.is_empty() {
            let tools: Vec<Tool> = tools.iter().map(|arc_tool| arc_tool.into()).collect();
            llm.chat_with_tools(&messages, Some(&tools))
                .await
                .map_err(|e| ReActExecutorError::LLMError(e.to_string()))?
        } else {
            llm.chat(&messages)
                .await
                .map_err(|e| ReActExecutorError::LLMError(e.to_string()))?
        };

        let response_text = response.text().clone().unwrap_or_default();

        // Check if there are tool calls to process
        if let Some(tool_calls) = response.tool_calls() {
            // Process tool calls
            let tool_results = self
                .process_tool_calls(tools, tool_calls.clone(), tx_event.clone())
                .await;

            // Record tool calls in state
            {
                let mut state_guard = state.write().await;
                for result in &tool_results {
                    state_guard.record_tool_call(result.clone());
                }
            }

            // Continue to next turn to let the LLM see the tool results
            Ok(TurnResult::Continue(Some(AgentOutput {
                response: response_text,
                tool_calls: tool_results,
            })))
        } else {
            // No tool calls, this is the final response
            Ok(TurnResult::Complete(AgentOutput {
                response: response_text,
                tool_calls: Vec::new(),
            }))
        }
    }

    fn max_turns_error(&self) -> Self::Error {
        ReActExecutorError::MaxTurnsExceeded {
            max_turns: self.config.max_turns,
        }
    }
}
