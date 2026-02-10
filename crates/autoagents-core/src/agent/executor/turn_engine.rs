use crate::agent::Context;
use crate::agent::executor::event_helper::EventHelper;
use crate::agent::executor::memory_policy::{MemoryAdapter, MemoryPolicy};
use crate::agent::executor::tool_processor::ToolProcessor;
use crate::agent::hooks::AgentHooks;
use crate::agent::task::Task;
use crate::channel::{Sender, channel};
use crate::tool::{ToolCallResult, ToolT, to_llm_tool};
use crate::utils::{receiver_into_stream, spawn_future};
use autoagents_llm::ToolCall;
use autoagents_llm::chat::{ChatMessage, ChatRole, MessageType, StreamChunk, StreamResponse, Tool};
use autoagents_llm::error::LLMError;
use autoagents_protocol::{Event, SubmissionId};
use futures::{Stream, StreamExt};
use serde_json::Value;
use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;
use thiserror::Error;

#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::mpsc;

#[cfg(target_arch = "wasm32")]
use futures::channel::mpsc;

/// Defines if tools are enabled for a given execution plan.
#[derive(Debug, Clone, Copy)]
pub enum ToolMode {
    Enabled,
    Disabled,
}

/// Defines which streaming primitive to use.
#[derive(Debug, Clone, Copy)]
pub enum StreamMode {
    Structured,
    Tool,
}

/// Configuration for the shared executor engine.
#[derive(Debug, Clone)]
pub struct TurnEngineConfig {
    pub max_turns: usize,
    pub tool_mode: ToolMode,
    pub stream_mode: StreamMode,
    pub memory_policy: MemoryPolicy,
}

impl TurnEngineConfig {
    pub fn basic(max_turns: usize) -> Self {
        Self {
            max_turns,
            tool_mode: ToolMode::Disabled,
            stream_mode: StreamMode::Structured,
            memory_policy: MemoryPolicy::basic(),
        }
    }

    pub fn react(max_turns: usize) -> Self {
        Self {
            max_turns,
            tool_mode: ToolMode::Enabled,
            stream_mode: StreamMode::Tool,
            memory_policy: MemoryPolicy::react(),
        }
    }
}

/// Normalized output emitted by the engine for a single turn.
#[derive(Debug, Clone)]
pub struct TurnEngineOutput {
    pub response: String,
    pub tool_calls: Vec<ToolCallResult>,
}

/// Streaming deltas emitted per turn.
#[derive(Debug)]
pub enum TurnDelta {
    Text(String),
    ToolResults(Vec<ToolCallResult>),
    Done(crate::agent::executor::TurnResult<TurnEngineOutput>),
}

#[derive(Error, Debug)]
pub enum TurnEngineError {
    #[error("LLM error: {0}")]
    LLMError(String),

    #[error("Run aborted by hook")]
    Aborted,

    #[error("Other error: {0}")]
    Other(String),
}

/// Per-run state for the turn engine.
#[derive(Clone)]
pub struct TurnState {
    memory: MemoryAdapter,
    stored_user: bool,
}

impl TurnState {
    pub fn new(context: &Context, policy: MemoryPolicy) -> Self {
        Self {
            memory: MemoryAdapter::new(context.memory(), policy),
            stored_user: false,
        }
    }

    pub fn memory(&self) -> &MemoryAdapter {
        &self.memory
    }

    pub fn stored_user(&self) -> bool {
        self.stored_user
    }

    fn mark_user_stored(&mut self) {
        self.stored_user = true;
    }
}

/// Shared turn engine that handles memory, tools, and events consistently.
#[derive(Debug, Clone)]
pub struct TurnEngine {
    config: TurnEngineConfig,
}

impl TurnEngine {
    pub fn new(config: TurnEngineConfig) -> Self {
        Self { config }
    }

    pub fn turn_state(&self, context: &Context) -> TurnState {
        TurnState::new(context, self.config.memory_policy.clone())
    }

    pub async fn run_turn<H: AgentHooks>(
        &self,
        hooks: &H,
        task: &Task,
        context: &Context,
        turn_state: &mut TurnState,
        turn_index: usize,
        max_turns: usize,
    ) -> Result<crate::agent::executor::TurnResult<TurnEngineOutput>, TurnEngineError> {
        let max_turns = normalize_max_turns(max_turns, self.config.max_turns);
        let tx_event = context.tx().ok();
        EventHelper::send_turn_started(
            &tx_event,
            task.submission_id,
            context.config().id,
            turn_index,
            max_turns,
        )
        .await;

        hooks.on_turn_start(turn_index, context).await;

        let include_user_prompt =
            should_include_user_prompt(turn_state.memory(), turn_state.stored_user());
        let messages = self
            .build_messages(context, task, turn_state.memory(), include_user_prompt)
            .await;

        if should_store_user(turn_state) {
            turn_state.memory.store_user(task).await;
            turn_state.mark_user_stored();
        }

        let tools = context.tools();
        let response = self.get_llm_response(context, &messages, tools).await?;
        let response_text = response.text().unwrap_or_default();

        let tool_calls = if matches!(self.config.tool_mode, ToolMode::Enabled) {
            response.tool_calls().unwrap_or_default()
        } else {
            Vec::new()
        };

        if !tool_calls.is_empty() {
            let tool_results = process_tool_calls_with_hooks(
                hooks,
                context,
                task.submission_id,
                tools,
                &tool_calls,
                &tx_event,
            )
            .await;

            turn_state
                .memory
                .store_tool_interaction(&tool_calls, &tool_results, &response_text)
                .await;
            record_tool_calls_state(context, &tool_results);

            EventHelper::send_turn_completed(
                &tx_event,
                task.submission_id,
                context.config().id,
                turn_index,
                false,
            )
            .await;
            hooks.on_turn_complete(turn_index, context).await;

            return Ok(crate::agent::executor::TurnResult::Continue(Some(
                TurnEngineOutput {
                    response: response_text,
                    tool_calls: tool_results,
                },
            )));
        }

        if !response_text.is_empty() {
            turn_state.memory.store_assistant(&response_text).await;
        }

        EventHelper::send_turn_completed(
            &tx_event,
            task.submission_id,
            context.config().id,
            turn_index,
            true,
        )
        .await;
        hooks.on_turn_complete(turn_index, context).await;

        Ok(crate::agent::executor::TurnResult::Complete(
            TurnEngineOutput {
                response: response_text,
                tool_calls: Vec::new(),
            },
        ))
    }

    pub async fn run_turn_stream<H>(
        &self,
        hooks: H,
        task: &Task,
        context: Arc<Context>,
        turn_state: &mut TurnState,
        turn_index: usize,
        max_turns: usize,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<TurnDelta, TurnEngineError>> + Send>>,
        TurnEngineError,
    >
    where
        H: AgentHooks + Clone + Send + Sync + 'static,
    {
        let max_turns = normalize_max_turns(max_turns, self.config.max_turns);
        let include_user_prompt =
            should_include_user_prompt(turn_state.memory(), turn_state.stored_user());
        let messages = self
            .build_messages(&context, task, turn_state.memory(), include_user_prompt)
            .await;

        if should_store_user(turn_state) {
            turn_state.memory.store_user(task).await;
            turn_state.mark_user_stored();
        }

        let (mut tx, rx) = channel::<Result<TurnDelta, TurnEngineError>>(100);
        let engine = self.clone();
        let context_clone = context.clone();
        let task = task.clone();
        let hooks = hooks.clone();
        let memory = turn_state.memory.clone();
        let messages = messages.clone();

        spawn_future(async move {
            let tx_event = context_clone.tx().ok();
            EventHelper::send_turn_started(
                &tx_event,
                task.submission_id,
                context_clone.config().id,
                turn_index,
                max_turns,
            )
            .await;
            hooks.on_turn_start(turn_index, &context_clone).await;

            let result = match engine.config.stream_mode {
                StreamMode::Structured => {
                    engine
                        .stream_structured(&context_clone, &task, &memory, &mut tx, &messages)
                        .await
                }
                StreamMode::Tool => {
                    engine
                        .stream_with_tools(
                            &hooks,
                            &context_clone,
                            &task,
                            context_clone.tools(),
                            &memory,
                            &mut tx,
                            &messages,
                        )
                        .await
                }
            };

            match result {
                Ok(turn_result) => {
                    let final_turn =
                        matches!(turn_result, crate::agent::executor::TurnResult::Complete(_));
                    EventHelper::send_turn_completed(
                        &tx_event,
                        task.submission_id,
                        context_clone.config().id,
                        turn_index,
                        final_turn,
                    )
                    .await;
                    hooks.on_turn_complete(turn_index, &context_clone).await;
                    let _ = tx.send(Ok(TurnDelta::Done(turn_result))).await;
                }
                Err(err) => {
                    let _ = tx.send(Err(err)).await;
                }
            }
        });

        Ok(receiver_into_stream(rx))
    }

    async fn stream_structured(
        &self,
        context: &Context,
        task: &Task,
        memory: &MemoryAdapter,
        tx: &mut Sender<Result<TurnDelta, TurnEngineError>>,
        messages: &[ChatMessage],
    ) -> Result<crate::agent::executor::TurnResult<TurnEngineOutput>, TurnEngineError> {
        let mut stream = self.get_structured_stream(context, messages).await?;
        let mut response_text = String::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| TurnEngineError::LLMError(e.to_string()))?;
            let content = chunk
                .choices
                .first()
                .and_then(|choice| choice.delta.content.as_ref())
                .map_or("", |value| value)
                .to_string();

            if content.is_empty() {
                continue;
            }

            response_text.push_str(&content);

            let _ = tx.send(Ok(TurnDelta::Text(content.clone()))).await;

            let tx_event = context.tx().ok();
            EventHelper::send_stream_chunk(
                &tx_event,
                task.submission_id,
                StreamChunk::Text(content),
            )
            .await;
        }

        if !response_text.is_empty() {
            memory.store_assistant(&response_text).await;
        }

        Ok(crate::agent::executor::TurnResult::Complete(
            TurnEngineOutput {
                response: response_text,
                tool_calls: Vec::new(),
            },
        ))
    }

    #[allow(clippy::too_many_arguments)]
    async fn stream_with_tools<H: AgentHooks>(
        &self,
        hooks: &H,
        context: &Context,
        task: &Task,
        tools: &[Box<dyn ToolT>],
        memory: &MemoryAdapter,
        tx: &mut Sender<Result<TurnDelta, TurnEngineError>>,
        messages: &[ChatMessage],
    ) -> Result<crate::agent::executor::TurnResult<TurnEngineOutput>, TurnEngineError> {
        let mut stream = self.get_tool_stream(context, messages, tools).await?;
        let mut response_text = String::new();
        let mut tool_calls = Vec::new();
        let mut tool_call_ids = HashSet::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| TurnEngineError::LLMError(e.to_string()))?;
            let chunk_clone = chunk.clone();

            match chunk {
                StreamChunk::Text(content) => {
                    response_text.push_str(&content);
                    let _ = tx.send(Ok(TurnDelta::Text(content.clone()))).await;
                }
                StreamChunk::ToolUseComplete { tool_call, .. } => {
                    if tool_call_ids.insert(tool_call.id.clone()) {
                        tool_calls.push(tool_call.clone());
                        let tx_event = context.tx().ok();
                        EventHelper::send_stream_tool_call(
                            &tx_event,
                            task.submission_id,
                            serde_json::to_value(tool_call).unwrap_or(Value::Null),
                        )
                        .await;
                    }
                }
                StreamChunk::Usage(_) => {}
                _ => {}
            }

            let tx_event = context.tx().ok();
            EventHelper::send_stream_chunk(&tx_event, task.submission_id, chunk_clone).await;
        }

        if tool_calls.is_empty() {
            if !response_text.is_empty() {
                memory.store_assistant(&response_text).await;
            }
            return Ok(crate::agent::executor::TurnResult::Complete(
                TurnEngineOutput {
                    response: response_text,
                    tool_calls: Vec::new(),
                },
            ));
        }

        let tx_event = context.tx().ok();
        let tool_results = process_tool_calls_with_hooks(
            hooks,
            context,
            task.submission_id,
            tools,
            &tool_calls,
            &tx_event,
        )
        .await;

        memory
            .store_tool_interaction(&tool_calls, &tool_results, &response_text)
            .await;
        record_tool_calls_state(context, &tool_results);

        let _ = tx
            .send(Ok(TurnDelta::ToolResults(tool_results.clone())))
            .await;

        Ok(crate::agent::executor::TurnResult::Continue(Some(
            TurnEngineOutput {
                response: response_text,
                tool_calls: tool_results,
            },
        )))
    }

    async fn get_llm_response(
        &self,
        context: &Context,
        messages: &[ChatMessage],
        tools: &[Box<dyn ToolT>],
    ) -> Result<Box<dyn autoagents_llm::chat::ChatResponse>, TurnEngineError> {
        let llm = context.llm();
        let output_schema = context.config().output_schema.clone();

        if matches!(self.config.tool_mode, ToolMode::Enabled) && !tools.is_empty() {
            let tools_serialized: Vec<Tool> = tools.iter().map(to_llm_tool).collect();
            llm.chat_with_tools(messages, Some(&tools_serialized), output_schema)
                .await
                .map_err(|e| TurnEngineError::LLMError(e.to_string()))
        } else {
            llm.chat(messages, output_schema)
                .await
                .map_err(|e| TurnEngineError::LLMError(e.to_string()))
        }
    }

    async fn get_structured_stream(
        &self,
        context: &Context,
        messages: &[ChatMessage],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>, TurnEngineError>
    {
        context
            .llm()
            .chat_stream_struct(messages, None, context.config().output_schema.clone())
            .await
            .map_err(|e| TurnEngineError::LLMError(e.to_string()))
    }

    async fn get_tool_stream(
        &self,
        context: &Context,
        messages: &[ChatMessage],
        tools: &[Box<dyn ToolT>],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, TurnEngineError>
    {
        let tools_serialized: Vec<Tool> = tools.iter().map(to_llm_tool).collect();
        context
            .llm()
            .chat_stream_with_tools(
                messages,
                if tools_serialized.is_empty() {
                    None
                } else {
                    Some(&tools_serialized)
                },
                context.config().output_schema.clone(),
            )
            .await
            .map_err(|e| TurnEngineError::LLMError(e.to_string()))
    }

    async fn build_messages(
        &self,
        context: &Context,
        task: &Task,
        memory: &MemoryAdapter,
        include_user_prompt: bool,
    ) -> Vec<ChatMessage> {
        let system_prompt = task
            .system_prompt
            .as_deref()
            .unwrap_or_else(|| &context.config().description);
        let mut messages = vec![ChatMessage {
            role: ChatRole::System,
            message_type: MessageType::Text,
            content: system_prompt.to_string(),
        }];

        let recalled = memory.recall_messages(task).await;
        messages.extend(recalled);

        if include_user_prompt {
            messages.push(user_message(task));
        }

        messages
    }
}

pub fn record_task_state(context: &Context, task: &Task) {
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

fn user_message(task: &Task) -> ChatMessage {
    if let Some((mime, image_data)) = &task.image {
        ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Image(((*mime).into(), image_data.clone())),
            content: task.prompt.clone(),
        }
    } else {
        ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: task.prompt.clone(),
        }
    }
}

fn should_include_user_prompt(memory: &MemoryAdapter, stored_user: bool) -> bool {
    if !memory.is_enabled() {
        return true;
    }
    if !memory.policy().recall {
        return true;
    }
    if !memory.policy().store_user {
        return true;
    }
    !stored_user
}

fn should_store_user(turn_state: &TurnState) -> bool {
    if !turn_state.memory.is_enabled() {
        return false;
    }
    if !turn_state.memory.policy().store_user {
        return false;
    }
    !turn_state.stored_user
}

fn normalize_max_turns(max_turns: usize, fallback: usize) -> usize {
    if max_turns == 0 {
        return fallback.max(1);
    }
    max_turns
}

fn record_tool_calls_state(context: &Context, tool_results: &[ToolCallResult]) {
    if tool_results.is_empty() {
        return;
    }
    let state = context.state();
    #[cfg(not(target_arch = "wasm32"))]
    if let Ok(mut guard) = state.try_lock() {
        for result in tool_results {
            guard.record_tool_call(result.clone());
        }
    };
    #[cfg(target_arch = "wasm32")]
    if let Some(mut guard) = state.try_lock() {
        for result in tool_results {
            guard.record_tool_call(result.clone());
        }
    };
}

async fn process_tool_calls_with_hooks<H: AgentHooks>(
    hooks: &H,
    context: &Context,
    submission_id: SubmissionId,
    tools: &[Box<dyn ToolT>],
    tool_calls: &[ToolCall],
    tx_event: &Option<mpsc::Sender<Event>>,
) -> Vec<ToolCallResult> {
    let mut results = Vec::new();
    for call in tool_calls {
        if let Some(result) = ToolProcessor::process_single_tool_call_with_hooks(
            hooks,
            context,
            submission_id,
            tools,
            call,
            tx_event,
        )
        .await
        {
            results.push(result);
        }
    }
    results
}
