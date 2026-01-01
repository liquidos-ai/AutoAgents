use crate::agent::executor::AgentExecutor;
use crate::agent::task::Task;
use crate::agent::{AgentDeriveT, Context, ExecutorConfig, TurnResult};
use crate::protocol::{Event, StreamingTurnResult, SubmissionId};
use crate::tool::{to_llm_tool, ToolCallResult, ToolT};
use async_trait::async_trait;
use autoagents_llm::chat::{ChatMessage, ChatRole, MessageType, Tool};
use autoagents_llm::error::LLMError;
use autoagents_llm::ToolCall;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::Arc;
use thiserror::Error;

#[cfg(not(target_arch = "wasm32"))]
pub use tokio::sync::mpsc::error::SendError;

#[cfg(target_arch = "wasm32")]
pub use futures::lock::Mutex;
#[cfg(target_arch = "wasm32")]
use futures::SinkExt;
#[cfg(target_arch = "wasm32")]
type SendError = futures::channel::mpsc::SendError;

use crate::agent::executor::event_helper::EventHelper;
use crate::agent::executor::memory_helper::MemoryHelper;
use crate::agent::executor::tool_processor::ToolProcessor;
use crate::agent::hooks::{AgentHooks, HookOutcome};
use crate::channel::{channel, Sender};
use crate::utils::{receiver_into_stream, spawn_future};

/// Output of the ReAct-style agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReActAgentOutput {
    pub response: String,
    pub tool_calls: Vec<ToolCallResult>,
    pub done: bool,
}

impl From<ReActAgentOutput> for Value {
    fn from(output: ReActAgentOutput) -> Self {
        serde_json::to_value(output).unwrap_or(Value::Null)
    }
}
impl From<ReActAgentOutput> for String {
    fn from(output: ReActAgentOutput) -> Self {
        output.response
    }
}

impl ReActAgentOutput {
    /// Try to parse the response string as structured JSON of type `T`.
    /// Returns `serde_json::Error` if parsing fails.
    pub fn try_parse<T: for<'de> serde::Deserialize<'de>>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_str::<T>(&self.response)
    }

    /// Parse the response string as structured JSON of type `T`, or map the raw
    /// text into `T` using the provided fallback function if parsing fails.
    /// This is useful in examples to avoid repeating parsing boilerplate.
    pub fn parse_or_map<T, F>(&self, fallback: F) -> T
    where
        T: for<'de> serde::Deserialize<'de>,
        F: FnOnce(&str) -> T,
    {
        self.try_parse::<T>()
            .unwrap_or_else(|_| fallback(&self.response))
    }
}

impl ReActAgentOutput {
    /// Extract the agent output from the ReAct response
    #[allow(clippy::result_large_err)]
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

    #[cfg(not(target_arch = "wasm32"))]
    #[error("Event error: {0}")]
    EventError(#[from] SendError<Event>),

    #[cfg(target_arch = "wasm32")]
    #[error("Event error: {0}")]
    EventError(#[from] SendError),

    #[error("Extracting Agent Output Error: {0}")]
    AgentOutputError(String),
}

/// Wrapper type for the multi-turn ReAct executor with tool calling support.
///
/// Use `ReActAgent<T>` when your agent needs to perform tool calls, manage
/// multiple turns, and optionally stream content and tool-call deltas.
#[derive(Debug)]
pub struct ReActAgent<T: AgentDeriveT> {
    inner: Arc<T>,
}

impl<T: AgentDeriveT> Clone for ReActAgent<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T: AgentDeriveT> ReActAgent<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

impl<T: AgentDeriveT> Deref for ReActAgent<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Implement AgentDeriveT for the wrapper by delegating to the inner type
#[async_trait]
impl<T: AgentDeriveT> AgentDeriveT for ReActAgent<T> {
    type Output = <T as AgentDeriveT>::Output;

    fn description(&self) -> &'static str {
        self.inner.description()
    }

    fn output_schema(&self) -> Option<Value> {
        self.inner.output_schema()
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn tools(&self) -> Vec<Box<dyn ToolT>> {
        self.inner.tools()
    }
}

#[async_trait]
impl<T> AgentHooks for ReActAgent<T>
where
    T: AgentDeriveT + AgentHooks + Send + Sync + 'static,
{
    async fn on_agent_create(&self) {
        self.inner.on_agent_create().await
    }

    async fn on_run_start(&self, task: &Task, ctx: &Context) -> HookOutcome {
        self.inner.on_run_start(task, ctx).await
    }

    async fn on_run_complete(&self, task: &Task, result: &Self::Output, ctx: &Context) {
        self.inner.on_run_complete(task, result, ctx).await
    }

    async fn on_turn_start(&self, turn_index: usize, ctx: &Context) {
        self.inner.on_turn_start(turn_index, ctx).await
    }

    async fn on_turn_complete(&self, turn_index: usize, ctx: &Context) {
        self.inner.on_turn_complete(turn_index, ctx).await
    }

    async fn on_tool_call(&self, tool_call: &ToolCall, ctx: &Context) -> HookOutcome {
        self.inner.on_tool_call(tool_call, ctx).await
    }

    async fn on_tool_start(&self, tool_call: &ToolCall, ctx: &Context) {
        self.inner.on_tool_start(tool_call, ctx).await
    }

    async fn on_tool_result(&self, tool_call: &ToolCall, result: &ToolCallResult, ctx: &Context) {
        self.inner.on_tool_result(tool_call, result, ctx).await
    }

    async fn on_tool_error(&self, tool_call: &ToolCall, err: Value, ctx: &Context) {
        self.inner.on_tool_error(tool_call, err, ctx).await
    }
    async fn on_agent_shutdown(&self) {
        self.inner.on_agent_shutdown().await
    }
}

impl<T: AgentDeriveT + AgentHooks> ReActAgent<T> {
    /// Process a single turn with the LLM
    async fn process_turn(
        &self,
        context: &Context,
        tools: &[Box<dyn ToolT>],
    ) -> Result<TurnResult<ReActAgentOutput>, ReActExecutorError> {
        let messages = self.prepare_messages(context).await;
        let response = self.get_llm_response(context, &messages, tools).await?;
        let response_text = response.text().unwrap_or_default();

        if let Some(tool_calls) = response.tool_calls() {
            self.handle_tool_calls(context, tools, tool_calls.clone(), response_text)
                .await
        } else {
            self.handle_text_response(context, response_text).await
        }
    }

    /// Get LLM response for the given messages and tools
    async fn get_llm_response(
        &self,
        context: &Context,
        messages: &[ChatMessage],
        tools: &[Box<dyn ToolT>],
    ) -> Result<Box<dyn autoagents_llm::chat::ChatResponse>, ReActExecutorError> {
        let llm = context.llm();
        let agent_config = context.config();
        let tools_serialized: Vec<Tool> = tools.iter().map(to_llm_tool).collect();

        if tools.is_empty() {
            llm.chat(messages, agent_config.output_schema.clone())
                .await
                .map_err(|e| ReActExecutorError::LLMError(e.to_string()))
        } else {
            llm.chat_with_tools(
                messages,
                Some(&tools_serialized),
                agent_config.output_schema.clone(),
            )
            .await
            .map_err(|e| ReActExecutorError::LLMError(e.to_string()))
        }
    }

    /// Handle tool calls and return the result
    async fn handle_tool_calls(
        &self,
        context: &Context,
        tools: &[Box<dyn ToolT>],
        tool_calls: Vec<ToolCall>,
        response_text: String,
    ) -> Result<TurnResult<ReActAgentOutput>, ReActExecutorError> {
        let tx_event = context.tx().ok();

        // Process tool calls
        let mut tool_results = Vec::new();
        for call in &tool_calls {
            if let Some(result) = ToolProcessor::process_single_tool_call_with_hooks(
                self, context, tools, call, &tx_event,
            )
            .await
            {
                tool_results.push(result);
            }
        }

        // Store in memory
        MemoryHelper::store_tool_interaction(
            &context.memory(),
            &tool_calls,
            &tool_results,
            &response_text,
        )
        .await;

        // Update state - use try_lock to avoid deadlock
        {
            let state = context.state();
            #[cfg(not(target_arch = "wasm32"))]
            if let Ok(mut guard) = state.try_lock() {
                for result in &tool_results {
                    guard.record_tool_call(result.clone());
                }
            };
            #[cfg(target_arch = "wasm32")]
            if let Some(mut guard) = state.try_lock() {
                for result in &tool_results {
                    guard.record_tool_call(result.clone());
                }
            };
        }

        Ok(TurnResult::Continue(Some(ReActAgentOutput {
            response: response_text,
            done: true,
            tool_calls: tool_results,
        })))
    }

    /// Handle text-only response
    async fn handle_text_response(
        &self,
        context: &Context,
        response_text: String,
    ) -> Result<TurnResult<ReActAgentOutput>, ReActExecutorError> {
        if !response_text.is_empty() {
            MemoryHelper::store_assistant_response(&context.memory(), response_text.clone()).await;
        }

        Ok(TurnResult::Complete(ReActAgentOutput {
            response: response_text,
            done: true,
            tool_calls: vec![],
        }))
    }

    /// Prepare messages for the current turn
    async fn prepare_messages(&self, context: &Context) -> Vec<ChatMessage> {
        let mut messages = vec![ChatMessage {
            role: ChatRole::System,
            message_type: MessageType::Text,
            content: context.config().description.clone(),
        }];

        let recalled = MemoryHelper::recall_messages(&context.memory()).await;
        messages.extend(recalled);

        messages
    }

    /// Process a streaming turn with tool support
    async fn process_streaming_turn(
        &self,
        context: &Context,
        tools: &[Box<dyn ToolT>],
        tx: &mut Sender<Result<ReActAgentOutput, ReActExecutorError>>,
        submission_id: SubmissionId,
    ) -> Result<StreamingTurnResult, ReActExecutorError> {
        let messages = self.prepare_messages(context).await;
        let mut stream = self.get_llm_stream(context, &messages, tools).await?;

        let mut response_text = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut tool_call_ids: HashSet<String> = HashSet::new();

        // Process stream chunks
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| ReActExecutorError::LLMError(e.to_string()))?;

            if let Some(choice) = chunk.choices.first() {
                // Handle content
                if let Some(content) = &choice.delta.content {
                    response_text.push_str(content);
                    let _ = tx
                        .send(Ok(ReActAgentOutput {
                            response: content.to_string(),
                            tool_calls: vec![],
                            done: false,
                        }))
                        .await;
                }

                // Handle tool calls
                if let Some(chunk_tool_calls) = &choice.delta.tool_calls {
                    for tool_call in chunk_tool_calls {
                        if tool_call_ids.insert(tool_call.id.clone()) {
                            tool_calls.push(tool_call.clone());

                            let tx_event = context.tx().ok();
                            EventHelper::send_stream_tool_call(
                                &tx_event,
                                submission_id,
                                serde_json::to_value(tool_call).unwrap_or(Value::Null),
                            )
                            .await;
                        }
                    }
                }

                // Send stream chunk event
                let tx_event = context.tx().ok();
                EventHelper::send_stream_chunk(&tx_event, submission_id, choice.clone()).await;
            }
        }

        // Process collected tool calls if any
        if tool_calls.is_empty() {
            if !response_text.is_empty() {
                MemoryHelper::store_assistant_response(&context.memory(), response_text.clone())
                    .await;
            }
            return Ok(StreamingTurnResult::Complete(response_text));
        }

        let tx_event = context.tx().ok();
        let tool_results =
            ToolProcessor::process_tool_calls(tools, tool_calls.clone(), tx_event).await;

        MemoryHelper::store_tool_interaction(
            &context.memory(),
            &tool_calls,
            &tool_results,
            &response_text,
        )
        .await;

        let state = context.state();
        let mut guard = state.lock().await;
        for result in &tool_results {
            guard.record_tool_call(result.clone());
        }

        Ok(StreamingTurnResult::ToolCallsProcessed(tool_results))
    }

    /// Get streaming LLM response
    async fn get_llm_stream(
        &self,
        context: &Context,
        messages: &[ChatMessage],
        tools: &[Box<dyn ToolT>],
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<autoagents_llm::chat::StreamResponse, LLMError>> + Send>>,
        ReActExecutorError,
    > {
        let llm = context.llm();
        let agent_config = context.config();
        let tools_serialized: Vec<Tool> = tools.iter().map(to_llm_tool).collect();

        llm.chat_stream_struct(
            messages,
            if tools.is_empty() {
                None
            } else {
                Some(&tools_serialized)
            },
            agent_config.output_schema.clone(),
        )
        .await
        .map_err(|e| ReActExecutorError::LLMError(e.to_string()))
    }
}

/// Implementation of AgentExecutor for the ReActExecutorWrapper
#[async_trait]
impl<T: AgentDeriveT + AgentHooks> AgentExecutor for ReActAgent<T> {
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
        // Initialize task
        MemoryHelper::store_user_message(
            &context.memory(),
            task.prompt.clone(),
            task.image.clone(),
        )
        .await;

        // Record task in state - use try_lock to avoid deadlock
        {
            let state = context.state();
            #[cfg(not(target_arch = "wasm32"))]
            if let Ok(mut guard) = state.try_lock() {
                guard.record_task(task.clone());
            };
            #[cfg(target_arch = "wasm32")]
            if let Some(mut guard) = state.try_lock() {
                guard.record_task(task.clone());
            };
        }

        // Send task started event
        let tx_event = context.tx().ok();
        EventHelper::send_task_started(
            &tx_event,
            task.submission_id,
            context.config().id,
            task.prompt.clone(),
            context.config().name.clone(),
        )
        .await;

        // Execute turns
        let max_turns = self.config().max_turns;
        let mut accumulated_tool_calls = Vec::new();
        let mut final_response = String::new();

        for turn_num in 0..max_turns {
            let tools = context.tools();
            EventHelper::send_turn_started(&tx_event, turn_num, max_turns).await;

            //Run Hook
            self.on_turn_start(turn_num, &context).await;

            match self.process_turn(&context, tools).await? {
                TurnResult::Complete(result) => {
                    if !accumulated_tool_calls.is_empty() {
                        return Ok(ReActAgentOutput {
                            response: result.response,
                            done: true,
                            tool_calls: accumulated_tool_calls,
                        });
                    }
                    EventHelper::send_turn_completed(&tx_event, turn_num, false).await;
                    //Run Hook
                    self.on_turn_complete(turn_num, &context).await;
                    return Ok(result);
                }
                TurnResult::Continue(Some(partial_result)) => {
                    accumulated_tool_calls.extend(partial_result.tool_calls);
                    if !partial_result.response.is_empty() {
                        final_response = partial_result.response;
                    }
                }
                TurnResult::Continue(None) => continue,
            }
        }

        if !final_response.is_empty() || !accumulated_tool_calls.is_empty() {
            EventHelper::send_task_completed(
                &tx_event,
                task.submission_id,
                context.config().id,
                final_response.clone(),
                context.config().name.clone(),
            )
            .await;
            Ok(ReActAgentOutput {
                response: final_response,
                done: true,
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
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<ReActAgentOutput, Self::Error>> + Send>>,
        Self::Error,
    > {
        // Initialize task
        MemoryHelper::store_user_message(
            &context.memory(),
            task.prompt.clone(),
            task.image.clone(),
        )
        .await;

        // Record task in state - use try_lock to avoid deadlock
        {
            let state = context.state();
            #[cfg(not(target_arch = "wasm32"))]
            if let Ok(mut guard) = state.try_lock() {
                guard.record_task(task.clone());
            };
            #[cfg(target_arch = "wasm32")]
            if let Some(mut guard) = state.try_lock() {
                guard.record_task(task.clone());
            };
        }

        // Send task started event
        let tx_event = context.tx().ok();
        EventHelper::send_task_started(
            &tx_event,
            task.submission_id,
            context.config().id,
            task.prompt.clone(),
            context.config().name.clone(),
        )
        .await;

        // Create channel for streaming
        let (mut tx, rx) = channel::<Result<ReActAgentOutput, ReActExecutorError>>(100);

        // Clone necessary components
        let executor = self.clone();
        let context_clone = context.clone();
        let submission_id = task.submission_id;
        let max_turns = executor.config().max_turns;

        // Spawn streaming task
        spawn_future(async move {
            let mut accumulated_tool_calls = Vec::new();
            let mut final_response = String::new();
            let tools = context_clone.tools();

            for turn in 0..max_turns {
                // Send turn events
                let tx_event = context_clone.tx().ok();
                EventHelper::send_turn_started(&tx_event, turn, max_turns).await;

                // Process streaming turn
                match executor
                    .process_streaming_turn(&context_clone, tools, &mut tx, submission_id)
                    .await
                {
                    Ok(StreamingTurnResult::Complete(response)) => {
                        final_response = response;
                        EventHelper::send_turn_completed(&tx_event, turn, true).await;
                        break;
                    }
                    Ok(StreamingTurnResult::ToolCallsProcessed(tool_results)) => {
                        accumulated_tool_calls.extend(tool_results);

                        let _ = tx
                            .send(Ok(ReActAgentOutput {
                                response: String::new(),
                                done: false,
                                tool_calls: accumulated_tool_calls.clone(),
                            }))
                            .await;

                        EventHelper::send_turn_completed(&tx_event, turn, false).await;
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        return;
                    }
                }
            }

            // Send final result
            let tx_event = context_clone.tx().ok();
            EventHelper::send_stream_complete(&tx_event, submission_id).await;

            let _ = tx
                .send(Ok(ReActAgentOutput {
                    response: final_response,
                    done: true,
                    tool_calls: accumulated_tool_calls,
                }))
                .await;
        });

        Ok(receiver_into_stream(rx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            done: true,
            tool_calls: vec![],
        };

        let react_value = serde_json::to_value(react_output).unwrap();
        let extracted: TestAgentOutput =
            ReActAgentOutput::extract_agent_output(react_value).unwrap();
        assert_eq!(extracted, agent_output);
    }
}
