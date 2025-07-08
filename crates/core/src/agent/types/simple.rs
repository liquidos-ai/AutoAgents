use crate::agent::base::{AgentConfig, AgentDeriveT, BaseAgent};
use crate::agent::error::AgentBuildError;
use crate::agent::executor::{AgentExecutor, ExecutorConfig, TurnResult};
use crate::agent::runnable::AgentState;
use crate::memory::MemoryProvider;
use crate::protocol::Event;
use crate::session::Task;
use crate::tool::ToolCallResult;
use async_trait::async_trait;
use autoagents_llm::chat::{ChatMessage, Tool};
use autoagents_llm::error::LLMError;
use autoagents_llm::{LLMProvider, ToolCall, ToolT};

use crate::error::Error;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

// --- SimpleAgent --- //

/// A simple agent that uses the SimpleExecutor.
#[derive(Clone)]
pub struct SimpleAgent {
    name: String,
    description: String,
    executor: SimpleExecutor,
}

#[async_trait]
impl AgentDeriveT for SimpleAgent {
    type Output = Value;

    fn name(&self) -> String {
        self.name.clone()
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn tools(&self) -> Vec<Box<dyn ToolT>> {
        self.executor.tools.clone()
    }
}

#[async_trait]
impl AgentExecutor for SimpleAgent {
    type Output = Value;
    type Error = SimpleError;

    fn config(&self) -> ExecutorConfig {
        self.executor.config()
    }

    async fn execute(
        &self,
        llm: Arc<dyn LLMProvider>,
        memory: Option<Arc<RwLock<Box<dyn MemoryProvider>>>>,
        tools: Vec<Box<dyn ToolT>>,
        agent_config: &AgentConfig,
        task: Task,
        state: Arc<RwLock<AgentState>>,
        tx_event: mpsc::Sender<Event>,
    ) -> Result<Self::Output, Self::Error> {
        self.executor
            .execute(llm, memory, tools, agent_config, task, state, tx_event)
            .await
    }


}

// --- SimpleExecutor --- //

/// A simple executor that processes user prompts and handles tool calls
pub struct SimpleExecutor {
    system_prompt: String,
    max_turns: usize,
    tools: Vec<Box<dyn ToolT>>,
}

impl Clone for SimpleExecutor {
    fn clone(&self) -> Self {
        Self {
            system_prompt: self.system_prompt.clone(),
            max_turns: self.max_turns,
            tools: self.tools.clone(),
        }
    }
}

impl SimpleExecutor {
    pub fn new(system_prompt: String, tools: Vec<Box<dyn ToolT>>) -> Self {
        Self {
            system_prompt,
            max_turns: 10,
            tools,
        }
    }

    async fn process_tool_calls(
        tools: &[Box<dyn ToolT>],
        tool_calls: Vec<ToolCall>,
    ) -> Result<Vec<ToolCallResult>, SimpleError> {
        let mut results = Vec::new();
        for call in tool_calls {
            let tool = tools
                .iter()
                .find(|t| t.name() == call.function.name)
                .ok_or_else(|| {
                    SimpleError::ToolError(format!("Tool '{}' not found", call.function.name))
                })?;

            let args: Value = serde_json::from_str(&call.function.arguments)
                .map_err(|e| SimpleError::ToolError(format!("Invalid JSON arguments: {}", e)))?;

            let result = tool.run(args);
            let success = result.is_ok();
            let result_value = match result {
                Ok(value) => value,
                Err(e) => serde_json::json!({ "error": e.to_string() }),
            };
            results.push(ToolCallResult {
                tool_name: call.function.name.clone(),
                arguments: serde_json::from_str(&call.function.arguments).unwrap_or(Value::Null),
                success,
                result: result_value,
            });
        }
        Ok(results)
    }
}

#[async_trait]
impl AgentExecutor for SimpleExecutor {
    type Output = Value;
    type Error = SimpleError;

    fn config(&self) -> ExecutorConfig {
        ExecutorConfig {
            max_turns: self.max_turns,
        }
    }

    async fn execute(
        &self,
        llm: Arc<dyn LLMProvider>,
        memory: Option<Arc<RwLock<Box<dyn MemoryProvider>>>>,
        tools: Vec<Box<dyn ToolT>>,
        _agent_config: &AgentConfig,
        task: Task,
        state: Arc<RwLock<AgentState>>,
        tx_event: mpsc::Sender<Event>,
    ) -> Result<Self::Output, Self::Error> {
        let mut messages: Vec<ChatMessage> = vec![];
        let system_prompt = self.system_prompt.clone();
        messages.push(ChatMessage::system().content(system_prompt).build());
        if let Some(mem) = memory {
            let history = mem.read().await.recall("", None).await?;
            messages.extend(history);
        }
        messages.push(ChatMessage::user().content(&task.prompt).build());
        // This line is now handled by the runnable agent's stream method
        // let _stream = llm.chat_stream(messages.as_slice(), None).await?;

        let mut turn_count = 0;
        loop {
            if turn_count >= self.config().max_turns {
                return Err(SimpleError::MaxTurnsExceeded);
            }
            turn_count += 1;

            let turn_result = self
                .process_turn(llm.clone(), &mut messages, &tools, state.clone(), tx_event.clone())
                .await?;

            match turn_result {
                TurnResult::Complete(output) => return Ok(output),
                TurnResult::Continue(_) => continue,
                TurnResult::Error(e) => {
                    return Err(SimpleError::SessionError(e));
                }
                TurnResult::Fatal(e) => {
                    return Err(SimpleError::SessionError(e.to_string()));
                }
            }
        }
    }


}

impl SimpleExecutor {
    async fn process_turn(
        &self,
        llm: Arc<dyn LLMProvider>,
        messages: &mut Vec<ChatMessage>,
        tools: &[Box<dyn ToolT>],
        _state: Arc<RwLock<AgentState>>,
        _tx_event: mpsc::Sender<Event>,
    ) -> Result<TurnResult<Value>, SimpleError> {
        let llm_tools = tools.iter().map(Tool::from).collect::<Vec<_>>();
        let response = llm
            .chat_with_tools(messages.as_slice(), Some(&llm_tools))
            .await
            .map_err(|e| SimpleError::LLMError(e.to_string()))?;

        let mut builder = ChatMessage::assistant().content(response.text().unwrap_or_default());
        if let Some(tool_calls) = response.tool_calls() {
            builder = builder.tool_use(tool_calls);
        }
        messages.push(builder.build());

        if let Some(tool_calls) = response.tool_calls() {
            let tool_results = Self::process_tool_calls(tools, tool_calls).await?;
            for result in tool_results {
                let content = serde_json::to_string(&result.result).unwrap_or_default();
                messages.push(
                    ChatMessage::tool()
                        .tool_call_id(&result.tool_name) // Assuming tool_name is the ID for now
                        .content(content)
                        .build(),
                );
            }
            Ok(TurnResult::Continue(None))
        } else {
            let text = response.text().unwrap_or_default();
            Ok(TurnResult::Complete(Value::String(text)))
        }
    }
}

// --- SimpleAgentBuilder --- //

/// Builder for creating Simple agents
#[derive(Default)]
pub struct SimpleAgentBuilder {
    name: String,
    description: String,
    system_prompt: String,
    tools: Vec<Box<dyn ToolT>>,
    llm: Option<Arc<dyn LLMProvider>>,
}

impl SimpleAgentBuilder {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        system_prompt: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            system_prompt: system_prompt.into(),
            tools: Vec::new(),
            llm: None,
        }
    }

    pub fn with_tool(mut self, tool: Box<dyn ToolT>) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn with_llm(mut self, llm: Arc<dyn LLMProvider>) -> Self {
        self.llm = Some(llm);
        self
    }

    pub fn build(self) -> Result<BaseAgent<SimpleAgent>, AgentBuildError> {
        let llm = self.llm.clone().ok_or_else(|| AgentBuildError::BuildFailure("Missing LLM provider".to_string()))?;

        let agent = SimpleAgent {
            name: self.name,
            description: self.description,
            executor: SimpleExecutor::new(self.system_prompt, self.tools),
        };

        Ok(BaseAgent::new(agent, llm, None))
    }
}

// --- Error --- //

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

impl From<LLMError> for SimpleError {
    fn from(e: LLMError) -> Self {
        SimpleError::LLMError(e.to_string())
    }
}

impl From<SimpleError> for Error {
    fn from(e: SimpleError) -> Self {
        Error::RunnableAgentError(e.into())
    }
}


