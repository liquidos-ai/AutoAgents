use crate::agent::base::{AgentDeriveT, BaseAgent};
use crate::agent::executor::{AgentExecutor, TurnResult};
use crate::agent::runnable::AgentState;
use crate::session::Task;
use crate::tool::ToolCallResult;
use async_trait::async_trait;
use autoagents_llm::chat::{ChatMessage, ChatRole, MessageType, Tool};
use autoagents_llm::{LLMProvider, ToolCall, ToolT};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

/// A simple executor that processes user prompts and handles tool calls
#[derive(Clone)]
pub struct SimpleExecutor {
    /// System prompt for the executor
    system_prompt: String,
    /// Maximum number of turns
    max_turns: usize,
    /// Tools available to this executor
    tools: Vec<Arc<Box<dyn ToolT>>>,
}

/// Output type for the simple executor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleOutput {
    /// The final response from the agent
    pub response: String,
    /// Tool calls that were made
    pub tool_calls: Vec<ToolCall>,
}

/// Error type for the simple executor
#[derive(Debug, thiserror::Error)]
pub enum SimpleError {
    #[error("LLM error: {0}")]
    LLMError(String),

    #[error("Session error: {0}")]
    SessionError(String),

    #[error("Tool execution error: {0}")]
    ToolError(String),

    #[error("Maximum turns exceeded")]
    MaxTurnsExceeded,
}

impl SimpleExecutor {
    /// Create a new simple executor
    pub fn new(system_prompt: String, tools: Vec<Arc<Box<dyn ToolT>>>) -> Self {
        Self {
            system_prompt,
            max_turns: 10,
            tools,
        }
    }

    /// Set the maximum number of turns
    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// Process tool calls from the LLM response
    async fn process_tool_calls(
        &self,
        tool_calls: Vec<ToolCall>,
    ) -> Result<Option<ToolCallResult>, SimpleError> {
        // // Process each tool call
        if let Some(tool) = tool_calls.first() {
            let arguments = tool.function.arguments.clone();
            for tool_self in self.tools.clone() {
                if tool.function.name == tool_self.name() {
                    let parsed_args = match serde_json::from_str(&arguments) {
                        Ok(args) => args,
                        Err(e) => {
                            // Return error as ToolCallResult with error details
                            return Ok(Some(ToolCallResult {
                                tool_name: tool.function.name.clone(),
                                arguments: arguments.clone().into(),
                                result: serde_json::json!({
                                    "error": format!("Invalid tool arguments: {}", e),
                                    "details": e.to_string()
                                }),
                            }));
                        }
                    };
                    let result = tool_self.run(parsed_args);
                    // let result = llm.call_tool(tool_name, arguments.clone());
                    return Ok(Some(ToolCallResult {
                        tool_name: tool.function.name.clone(),
                        arguments: arguments.clone().into(),
                        result,
                    }));
                }
            }
        }

        Ok(None)
    }
}

#[async_trait]
impl AgentExecutor for SimpleExecutor {
    type Output = Value;
    type Error = SimpleError;

    async fn execute(
        &self,
        llm: Arc<dyn LLMProvider>,
        task: Task,
        state: Arc<Mutex<AgentState>>,
    ) -> Result<Self::Output, Self::Error> {
        // Initialize conversation with system prompt and task
        let mut previous_chat_history = state.lock().await.get_history().messages;
        let messages = vec![
            ChatMessage {
                role: ChatRole::Assistant,
                message_type: MessageType::Text,
                content: self.system_prompt.clone(),
            },
            ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: task.prompt,
            },
        ];
        previous_chat_history.extend(messages);

        let mut turn_count = 0;

        // Process turns until we get a final response or hit the limit
        loop {
            turn_count += 1;
            if turn_count > self.max_turns {
                return Err(SimpleError::MaxTurnsExceeded);
            }

            match self
                .process_turn(llm.clone(), &mut previous_chat_history, state.clone())
                .await?
            {
                TurnResult::Complete(output) => return Ok(output),
                TurnResult::Continue => continue,
                TurnResult::Error(e) => {
                    // Add error to conversation and continue
                    previous_chat_history.push(ChatMessage {
                        role: ChatRole::Assistant,
                        message_type: MessageType::Text,
                        content: format!("Error occurred: {}", e),
                    });
                }
            }
        }
    }

    async fn process_turn(
        &self,
        llm: Arc<dyn LLMProvider>,
        messages: &mut Vec<ChatMessage>,
        state: Arc<Mutex<AgentState>>,
    ) -> Result<TurnResult<Self::Output>, Self::Error> {
        let has_tools = !self.tools.is_empty();
        let response = if has_tools {
            let tools = self.tools.iter().map(Tool::from).collect::<Vec<_>>();
            llm.chat_with_tools(messages.as_slice(), Some(tools.as_slice()))
                .await
        } else {
            // Call the LLM
            llm.chat(messages.as_slice()).await
        };

        let response_mapped = response.map_err(|e| SimpleError::LLMError(e.to_string()))?;

        // Check if the response contains tool calls
        let final_response = if response_mapped.tool_calls().is_none() {
            // No tool calls, this is the final response
            response_mapped.text().clone().unwrap()
        } else {
            let tool_calls = response_mapped.tool_calls().unwrap();
            let result = self.process_tool_calls(tool_calls).await.unwrap();
            let restult_string = result.clone().unwrap().result.to_string();
            messages.push(ChatMessage {
                role: ChatRole::Assistant,
                message_type: MessageType::Text,
                content: restult_string,
            });
            state.lock().await.record_tool_call(result.unwrap());
            // Continue the conversation after tool calls
            return Ok(TurnResult::Continue);
        };
        // Record the final message
        state.lock().await.record_conversation(ChatMessage {
            role: ChatRole::Assistant,
            message_type: MessageType::Text,
            content: final_response.clone(),
        });

        Ok(TurnResult::Complete(final_response.into()))
    }

    fn max_turns(&self) -> usize {
        self.max_turns
    }
}

/// Builder for creating Simple agents
pub struct SimpleAgentBuilder {
    name: String,
    executor: SimpleExecutor,
    prompt: String,
    tools: Vec<Arc<Box<dyn ToolT>>>,
    llm: Option<Arc<dyn LLMProvider>>,
}

impl SimpleAgentBuilder {
    pub fn new(name: impl Into<String>, system_prompt: String) -> Self {
        Self {
            name: name.into(),
            prompt: String::new(),
            executor: SimpleExecutor::new(system_prompt, vec![]),
            tools: Vec::new(),
            llm: None,
        }
    }

    pub fn from_agent<T: AgentDeriveT>(agent: T) -> Self {
        let tools: Vec<Arc<Box<dyn ToolT>>> = agent
            .tools()
            .into_iter()
            .map(|tool| {
                let boxed: Box<dyn ToolT> = tool;
                let arc: Arc<Box<dyn ToolT>> = Arc::new(boxed);
                arc
            })
            .collect();
        Self {
            name: agent.name().into(),
            prompt: agent.prompt().into(),
            executor: SimpleExecutor::new(agent.prompt().into(), tools.clone()),
            tools,
            llm: None,
        }
    }

    pub fn with_llm(mut self, llm: Arc<dyn LLMProvider>) -> Self {
        self.llm = Some(llm);
        self
    }

    pub fn build(self) -> Result<BaseAgent<SimpleExecutor>, SimpleError> {
        let llm = self
            .llm
            .ok_or_else(|| SimpleError::LLMError("LLm is not set".into()))?;
        Ok(BaseAgent::new(
            self.name,
            self.prompt,
            self.executor,
            self.tools,
            llm,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_executor_creation() {
        let executor = SimpleExecutor::new("You are a helpful assistant".to_string(), vec![])
            .with_max_turns(5);

        assert_eq!(executor.max_turns(), 5);
    }
}
