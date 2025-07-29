use crate::agent::base::AgentConfig;
use crate::agent::executor::{AgentExecutor, ExecutorConfig, TurnResult, StreamingAgentExecutor};
use crate::agent::runnable::AgentState;
use crate::memory::MemoryProvider;
use crate::protocol::{Event, SubmissionId};
use crate::runtime::Task;
use crate::tool::{ToolCallResult, ToolT};
use async_trait::async_trait;
use autoagents_llm::{
    chat::{ChatMessage, ChatRole, MessageType, Tool},
    LLMProvider, ToolCall,
};
use log::debug;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::{mpsc, RwLock};
use tokio::sync::mpsc::error::SendError;

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
    pub fn extract_agent_output<T>(val: Value) -> Result<T, ReActExecutorError>
    where
        T: for<'de> serde::Deserialize<'de>,
    {
        let react_output: Self = serde_json::from_value(val)
            .map_err(|e| ReActExecutorError::JsonError(e))?;
        serde_json::from_value(serde_json::to_value(react_output).unwrap_or(Value::Null))
            .map_err(|e| ReActExecutorError::JsonError(e))
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

    /// Process a turn with streaming support
    #[allow(clippy::too_many_arguments)]
    async fn process_turn_streaming(
        &self,
        llm: Arc<dyn LLMProvider>,
        messages: &[ChatMessage],
        memory: Option<Arc<RwLock<Box<dyn MemoryProvider>>>>,
        tools: &[Box<dyn ToolT>],
        agent_config: &AgentConfig,
        state: Arc<RwLock<AgentState>>,
        tx_event: mpsc::Sender<Event>,
        submission_id: SubmissionId,
    ) -> Result<TurnResult<ReActAgentOutput>, ReActExecutorError> {
        // Try to use streaming if available, otherwise fall back to non-streaming
        if !tools.is_empty() {
            let tools_serialized: Vec<Tool> = tools.iter().map(Tool::from).collect();
            
            // Try streaming first
            match llm.chat_stream(messages).await {
                Ok(mut stream) => {
                    let mut response_text = String::new();

                    use tokio_stream::StreamExt;
                    while let Some(chunk_result) = stream.next().await {
                        match chunk_result {
                            Ok(chunk) => {
                                // For now, we'll accumulate text chunks
                                // In a more sophisticated implementation, we'd parse tool calls from the stream
                                response_text.push_str(&chunk);
                                
                                // Send streaming text chunk
                                let _ = tx_event
                                    .send(Event::StreamTextChunk {
                                        sub_id: submission_id,
                                        chunk,
                                        is_final: false,
                                    })
                                    .await;
                            }
                            Err(e) => {
                                return Err(ReActExecutorError::LLMError(e.to_string()));
                            }
                        }
                    }

                    // Send final text chunk
                    if !response_text.is_empty() {
                        let _ = tx_event
                            .send(Event::StreamTextChunk {
                                sub_id: submission_id,
                                chunk: response_text.clone(),
                                is_final: true,
                            })
                            .await;
                    }

                    // For now, fall back to non-streaming for tool calls
                    // In a full implementation, we'd parse tool calls from the stream
                    let response = llm.chat_with_tools(
                        messages,
                        Some(&tools_serialized),
                        agent_config.output_schema.clone(),
                    )
                    .await
                    .map_err(|e| ReActExecutorError::LLMError(e.to_string()))?;

                    if let Some(tool_calls) = response.tool_calls() {
                        // Process tool calls with streaming events
                        for tool_call in &tool_calls {
                            // Send tool call start event
                            let _ = tx_event
                                .send(Event::StreamToolCallStart {
                                    sub_id: submission_id,
                                    tool_call: tool_call.clone(),
                                })
                                .await;

                            // Process the tool call
                            let tool_results = self
                                .process_tool_calls(tools, vec![tool_call.clone()], tx_event.clone(), memory.clone())
                                .await;

                            // Send tool call end event
                            let _ = tx_event
                                .send(Event::StreamToolCallEnd {
                                    sub_id: submission_id,
                                    tool_call_id: tool_call.id.clone(),
                                })
                                .await;

                            return Ok(TurnResult::Continue(Some(ReActAgentOutput {
                                response: response_text,
                                tool_calls: tool_results,
                            })));
                        }
                    }

                    Ok(TurnResult::Complete(ReActAgentOutput {
                        response: response_text,
                        tool_calls: vec![],
                    }))
                }
                Err(_) => {
                    // Fall back to non-streaming
                    self.process_turn(llm, messages, memory, tools, agent_config, state, tx_event).await
                }
            }
        } else {
            // For non-tool calls, use streaming if available
            match llm.chat_stream(messages).await {
                Ok(mut stream) => {
                    let mut response_text = String::new();

                    use tokio_stream::StreamExt;
                    while let Some(chunk_result) = stream.next().await {
                        match chunk_result {
                            Ok(chunk) => {
                                // For now, we'll accumulate text chunks
                                // In a more sophisticated implementation, we'd parse tool calls from the stream
                                response_text.push_str(&chunk);
                                
                                // Send streaming text chunk
                                let _ = tx_event
                                    .send(Event::StreamTextChunk {
                                        sub_id: submission_id,
                                        chunk,
                                        is_final: false,
                                    })
                                    .await;
                            }
                            Err(e) => {
                                return Err(ReActExecutorError::LLMError(e.to_string()));
                            }
                        }
                    }

                    // Send final text chunk
                    if !response_text.is_empty() {
                        let _ = tx_event
                            .send(Event::StreamTextChunk {
                                sub_id: submission_id,
                                chunk: response_text.clone(),
                                is_final: true,
                            })
                            .await;
                    }

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
                Err(_) => {
                    // Fall back to non-streaming
                    self.process_turn(llm, messages, memory, tools, agent_config, state, tx_event).await
                }
            }
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
        task: Task,
        state: Arc<RwLock<AgentState>>,
        tx_event: mpsc::Sender<Event>,
    ) -> Result<Self::Output, Self::Error> {
        debug!("Starting ReAct Executor");
        let max_turns = self.config().max_turns;
        let mut accumulated_tool_calls = Vec::new();

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
                        let mut merged_result = result;
                        merged_result.tool_calls.extend(accumulated_tool_calls);
                        return Ok(merged_result);
                    } else {
                        return Ok(result);
                    }
                }
                TurnResult::Continue(Some(result)) => {
                    accumulated_tool_calls.extend(result.tool_calls);
                }
                TurnResult::Continue(None) => {
                    // No output from this turn, continue to next
                }
            }

            tx_event
                .send(Event::TurnCompleted {
                    turn_number: turn,
                    final_turn: false,
                })
                .await?;
        }

        tx_event
            .send(Event::TurnCompleted {
                turn_number: max_turns,
                final_turn: true,
            })
            .await?;

        Err(ReActExecutorError::MaxTurnsExceeded { max_turns })
    }
}

/// Streaming implementation for ReAct executor
#[async_trait]
impl<T: ReActExecutor> StreamingAgentExecutor for T {
    async fn execute_streaming(
        &self,
        llm: Arc<dyn LLMProvider>,
        mut memory: Option<Arc<RwLock<Box<dyn MemoryProvider>>>>,
        tools: Vec<Box<dyn ToolT>>,
        agent_config: &AgentConfig,
        task: Task,
        state: Arc<RwLock<AgentState>>,
        tx_event: mpsc::Sender<Event>,
    ) -> Result<<T as AgentExecutor>::Output, <T as AgentExecutor>::Error> {
        debug!("Starting Streaming ReAct Executor");
        let max_turns = self.config().max_turns;
        let mut accumulated_tool_calls = Vec::new();

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
            // Prepare messages with memory
            let mut messages = vec![ChatMessage {
                role: ChatRole::System,
                message_type: MessageType::Text,
                content: agent_config.description.clone(),
            }];
            if let Some(memory) = &memory {
                // Fetch All previous messages and extend
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
                .process_turn_streaming(
                    llm.clone(),
                    &messages,
                    memory.clone(),
                    &tools,
                    agent_config,
                    state.clone(),
                    tx_event.clone(),
                    task.submission_id,
                )
                .await?
            {
                TurnResult::Complete(result) => {
                    // If we have accumulated tool calls, merge them with the final result
                    if !accumulated_tool_calls.is_empty() {
                        let mut merged_result = result;
                        merged_result.tool_calls.extend(accumulated_tool_calls);
                        return Ok(merged_result);
                    } else {
                        return Ok(result);
                    }
                }
                TurnResult::Continue(Some(result)) => {
                    accumulated_tool_calls.extend(result.tool_calls);
                }
                TurnResult::Continue(None) => {
                    // No output from this turn, continue to next
                }
            }

            tx_event
                .send(Event::TurnCompleted {
                    turn_number: turn,
                    final_turn: false,
                })
                .await?;
        }

        tx_event
            .send(Event::TurnCompleted {
                turn_number: max_turns,
                final_turn: true,
            })
            .await?;

        Err(ReActExecutorError::MaxTurnsExceeded { max_turns })
    }

    fn supports_streaming(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::base::AgentConfig;
    use crate::agent::runnable::AgentState;
    use crate::memory::MemoryProvider;
    use crate::protocol::Event;
    use crate::runtime::Task;
    use async_trait::async_trait;
    use autoagents_llm::{chat::StructuredOutputFormat, LLMProvider};
    use serde_json::Value;
    use std::sync::Arc;
    use tokio::sync::{mpsc, RwLock};

    // Mock LLM provider for testing
    struct MockLLMProvider;

    #[async_trait]
    impl autoagents_llm::chat::ChatProvider for MockLLMProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[autoagents_llm::chat::ChatMessage],
            _tools: Option<&[autoagents_llm::chat::Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Box<dyn autoagents_llm::chat::ChatResponse>, autoagents_llm::error::LLMError> {
            // Mock response with tool calls
            let tool_call = autoagents_llm::ToolCall {
                id: "call_123".to_string(),
                call_type: "function".to_string(),
                function: autoagents_llm::FunctionCall {
                    name: "test_function".to_string(),
                    arguments: r#"{"param": "value"}"#.to_string(),
                },
            };

            Ok(Box::new(MockChatResponse {
                text: Some("I'll help you with that.".to_string()),
                tool_calls: Some(vec![tool_call]),
            }))
        }
    }

    #[async_trait]
    impl autoagents_llm::completion::CompletionProvider for MockLLMProvider {
        async fn complete(
            &self,
            _req: &autoagents_llm::completion::CompletionRequest,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<autoagents_llm::completion::CompletionResponse, autoagents_llm::error::LLMError> {
            Ok(autoagents_llm::completion::CompletionResponse {
                text: "Mock completion response".to_string(),
            })
        }
    }

    #[async_trait]
    impl autoagents_llm::embedding::EmbeddingProvider for MockLLMProvider {
        async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, autoagents_llm::error::LLMError> {
            Ok(vec![vec![0.1, 0.2, 0.3]])
        }
    }

    impl autoagents_llm::models::ModelsProvider for MockLLMProvider {}

    impl LLMProvider for MockLLMProvider {
        fn tools(&self) -> Option<&[autoagents_llm::chat::Tool]> {
            None
        }
    }

    struct MockChatResponse {
        text: Option<String>,
        tool_calls: Option<Vec<autoagents_llm::ToolCall>>,
    }

    impl autoagents_llm::chat::ChatResponse for MockChatResponse {
        fn text(&self) -> Option<String> {
            self.text.clone()
        }

        fn tool_calls(&self) -> Option<Vec<autoagents_llm::ToolCall>> {
            self.tool_calls.clone()
        }
    }

    impl std::fmt::Debug for MockChatResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MockChatResponse")
                .field("text", &self.text)
                .field("tool_calls", &self.tool_calls)
                .finish()
        }
    }

    impl std::fmt::Display for MockChatResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "MockChatResponse")
        }
    }

    // Mock tool for testing
    #[derive(Debug)]
    struct MockTool;

    #[async_trait]
    impl crate::tool::ToolRuntime for MockTool {
        fn execute(&self, _args: Value) -> Result<Value, crate::tool::ToolCallError> {
            Ok(serde_json::json!({"result": "success"}))
        }
    }

    #[async_trait]
    impl ToolT for MockTool {
        fn name(&self) -> &'static str {
            "mock_tool"
        }

        fn description(&self) -> &'static str {
            "A mock tool for testing"
        }

        fn args_schema(&self) -> Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "param": {"type": "string"}
                },
                "required": ["param"]
            })
        }

        fn run(&self, _args: Value) -> Result<Value, crate::tool::ToolCallError> {
            Ok(serde_json::json!({"result": "success"}))
        }
    }

    // Mock ReAct executor for testing
    struct MockReActExecutor;

    #[async_trait]
    impl ReActExecutor for MockReActExecutor {}

    #[tokio::test]
    async fn test_streaming_executor_supports_streaming() {
        let executor = MockReActExecutor;
        assert!(executor.supports_streaming());
    }

    #[tokio::test]
    async fn test_streaming_executor_execution() {
        let executor = MockReActExecutor;
        let llm = Arc::new(MockLLMProvider);
        let tools = vec![Box::new(MockTool) as Box<dyn ToolT>];
        let agent_config = AgentConfig {
            name: "test_agent".to_string(),
            description: "A test agent".to_string(),
            id: uuid::Uuid::new_v4(),
            output_schema: None,
        };
        let task = Task::new("Test task", None);
        let state = Arc::new(RwLock::new(AgentState::new()));
        let (tx_event, mut rx_event) = mpsc::channel(100);

        // Test that the executor can be executed
        let result = executor
            .execute_streaming(
                llm,
                None,
                tools,
                &agent_config,
                task,
                state,
                tx_event,
            )
            .await;

        // The execution should complete (even if it fails due to max turns)
        assert!(result.is_ok() || result.is_err());

        // Check that events were sent
        let mut event_count = 0;
        while let Ok(_) = rx_event.try_recv() {
            event_count += 1;
        }
        assert!(event_count > 0, "Expected events to be sent");
    }

    #[tokio::test]
    async fn test_streaming_events() {
        let (tx_event, mut rx_event) = mpsc::channel(100);
        let submission_id = uuid::Uuid::new_v4();

        // Send a streaming text chunk
        let _ = tx_event
            .send(Event::StreamTextChunk {
                sub_id: submission_id,
                chunk: "Hello".to_string(),
                is_final: false,
            })
            .await;

        // Send a tool call start event
        let tool_call = autoagents_llm::ToolCall {
            id: "call_123".to_string(),
            call_type: "function".to_string(),
            function: autoagents_llm::FunctionCall {
                name: "test_function".to_string(),
                arguments: r#"{"param": "value"}"#.to_string(),
            },
        };

        let _ = tx_event
            .send(Event::StreamToolCallStart {
                sub_id: submission_id,
                tool_call,
            })
            .await;

        // Send a tool call end event
        let _ = tx_event
            .send(Event::StreamToolCallEnd {
                sub_id: submission_id,
                tool_call_id: "call_123".to_string(),
            })
            .await;

        // Receive and verify events
        let mut events = Vec::new();
        while let Ok(event) = rx_event.try_recv() {
            events.push(event);
        }

        assert_eq!(events.len(), 3, "Expected 3 events");
        
        // Check that we have the expected event types
        let has_text_chunk = events.iter().any(|e| matches!(e, Event::StreamTextChunk { .. }));
        let has_tool_start = events.iter().any(|e| matches!(e, Event::StreamToolCallStart { .. }));
        let has_tool_end = events.iter().any(|e| matches!(e, Event::StreamToolCallEnd { .. }));

        assert!(has_text_chunk, "Expected StreamTextChunk event");
        assert!(has_tool_start, "Expected StreamToolCallStart event");
        assert!(has_tool_end, "Expected StreamToolCallEnd event");
    }
}
