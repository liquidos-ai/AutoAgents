use crate::agent::Context;
use crate::agent::executor::event_helper::EventHelper;
use crate::agent::executor::memory_policy::{MemoryAdapter, MemoryPolicy};
use crate::agent::executor::tool_processor::ToolProcessor;
use crate::agent::hooks::AgentHooks;
use crate::agent::task::Task;
use crate::channel::{Sender, channel};
use crate::tool::{ToolCallResult, ToolT, to_llm_tool};
use crate::utils::stream_from_producer;
use autoagents_llm::ToolCall;
use autoagents_llm::chat::{ChatMessage, ChatRole, MessageType, StreamChunk, StreamResponse};
use autoagents_llm::error::LLMError;
use autoagents_protocol::{Event, SubmissionId};
#[cfg(target_arch = "wasm32")]
use futures::SinkExt;
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
    pub reasoning_content: String,
    pub tool_calls: Vec<ToolCallResult>,
}

/// Streaming deltas emitted per turn.
#[derive(Debug)]
pub enum TurnDelta {
    Text(String),
    ReasoningContent(String),
    ToolResults(Vec<ToolCallResult>),
    Done(crate::agent::executor::TurnResult<TurnEngineOutput>),
}

#[derive(Error, Debug)]
pub enum TurnEngineError {
    #[error("LLM error: {0}")]
    LLMError(
        #[from]
        #[source]
        LLMError,
    ),

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
        let store_user = should_store_user(turn_state);

        let tools = context.tools();
        let response = self.get_llm_response(context, &messages, tools).await?;
        let response_text = response.text().unwrap_or_default();
        let reasoning_content = response.thinking().unwrap_or_default();
        if store_user {
            turn_state.memory.store_user(task).await;
            turn_state.mark_user_stored();
        }

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
                    reasoning_content,
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
                reasoning_content,
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
    ) -> Result<crate::utils::BoxRuntimeStream<Result<TurnDelta, TurnEngineError>>, TurnEngineError>
    where
        H: AgentHooks + Clone + Send + Sync + 'static,
    {
        let max_turns = normalize_max_turns(max_turns, self.config.max_turns);
        let include_user_prompt =
            should_include_user_prompt(turn_state.memory(), turn_state.stored_user());
        let messages = self
            .build_messages(&context, task, turn_state.memory(), include_user_prompt)
            .await;
        let store_user = should_store_user(turn_state);
        if store_user {
            turn_state.mark_user_stored();
        }

        let (mut tx, rx) = channel::<Result<TurnDelta, TurnEngineError>>(100);
        let engine = self.clone();
        let context_clone = context.clone();
        let task = task.clone();
        let hooks = hooks.clone();
        let memory = turn_state.memory.clone();
        let messages = messages.clone();

        let producer = async move {
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
                        .stream_structured(
                            &context_clone,
                            &task,
                            &memory,
                            &mut tx,
                            &messages,
                            store_user,
                        )
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
                            store_user,
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
        };

        Ok(stream_from_producer(rx, producer))
    }

    async fn stream_structured(
        &self,
        context: &Context,
        task: &Task,
        memory: &MemoryAdapter,
        tx: &mut Sender<Result<TurnDelta, TurnEngineError>>,
        messages: &[ChatMessage],
        store_user: bool,
    ) -> Result<crate::agent::executor::TurnResult<TurnEngineOutput>, TurnEngineError> {
        let mut stream = self.get_structured_stream(context, messages).await?;
        if store_user {
            memory.store_user(task).await;
        }
        let mut response_text = String::default();
        let mut reasoning_content = String::default();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(TurnEngineError::LLMError)?;
            let delta = chunk.choices.first().map(|choice| &choice.delta);
            let content = delta
                .and_then(|d| d.content.as_ref())
                .map(String::as_str)
                .unwrap_or("")
                .to_string();
            let reasoning = delta
                .and_then(|d| d.reasoning_content.as_ref())
                .map(String::as_str)
                .unwrap_or("")
                .to_string();

            let tx_event = context.tx().ok();
            if !content.is_empty() {
                response_text.push_str(&content);
                let _ = tx.send(Ok(TurnDelta::Text(content.clone()))).await;
                EventHelper::send_stream_chunk(
                    &tx_event,
                    task.submission_id,
                    StreamChunk::Text(content),
                )
                .await;
            }
            if !reasoning.is_empty() {
                reasoning_content.push_str(&reasoning);
                let _ = tx
                    .send(Ok(TurnDelta::ReasoningContent(reasoning.clone())))
                    .await;
                EventHelper::send_stream_chunk(
                    &tx_event,
                    task.submission_id,
                    StreamChunk::ReasoningContent(reasoning),
                )
                .await;
            }
        }

        if !response_text.is_empty() {
            memory.store_assistant(&response_text).await;
        }

        Ok(crate::agent::executor::TurnResult::Complete(
            TurnEngineOutput {
                response: response_text,
                reasoning_content,
                tool_calls: Vec::default(),
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
        store_user: bool,
    ) -> Result<crate::agent::executor::TurnResult<TurnEngineOutput>, TurnEngineError> {
        let mut stream = self.get_tool_stream(context, messages, tools).await?;
        if store_user {
            memory.store_user(task).await;
        }
        let mut response_text = String::default();
        let mut reasoning_content = String::default();
        let mut tool_calls = Vec::default();
        let mut tool_call_ids: HashSet<String> = HashSet::default();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(TurnEngineError::LLMError)?;
            let chunk_clone = chunk.clone();

            match chunk {
                StreamChunk::Text(content) => {
                    response_text.push_str(&content);
                    let _ = tx.send(Ok(TurnDelta::Text(content.clone()))).await;
                }
                StreamChunk::ReasoningContent(content) => {
                    reasoning_content.push_str(&content);
                    let _ = tx.send(Ok(TurnDelta::ReasoningContent(content))).await;
                }
                StreamChunk::ToolUseComplete { tool_call, .. }
                    if tool_call_ids.insert(tool_call.id.clone()) =>
                {
                    tool_calls.push(tool_call.clone());
                    let tx_event = context.tx().ok();
                    EventHelper::send_stream_tool_call(
                        &tx_event,
                        task.submission_id,
                        serde_json::to_value(tool_call).unwrap_or(Value::Null),
                    )
                    .await;
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
                    reasoning_content,
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
                reasoning_content,
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
            let cached = context.serialized_tools();
            let tools_serialized = if let Some(cached) = cached {
                cached
            } else {
                Arc::new(tools.iter().map(to_llm_tool).collect::<Vec<_>>())
            };
            llm.chat_with_tools(messages, Some(&tools_serialized), output_schema)
                .await
                .map_err(TurnEngineError::LLMError)
        } else {
            llm.chat(messages, output_schema)
                .await
                .map_err(TurnEngineError::LLMError)
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
            .map_err(TurnEngineError::LLMError)
    }

    async fn get_tool_stream(
        &self,
        context: &Context,
        messages: &[ChatMessage],
        tools: &[Box<dyn ToolT>],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, TurnEngineError>
    {
        let cached = context.serialized_tools();
        let tools_serialized = if let Some(cached) = cached {
            cached
        } else {
            Arc::new(tools.iter().map(to_llm_tool).collect::<Vec<_>>())
        };
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
            .map_err(TurnEngineError::LLMError)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::memory::{MemoryProvider, SlidingWindowMemory};
    use crate::agent::task::Task;
    use crate::agent::{AgentConfig, Context};
    use crate::tests::{ConfigurableLLMProvider, StaticChatResponse};
    use async_trait::async_trait;
    use autoagents_llm::LLMProvider;
    use autoagents_llm::ToolCall;
    use autoagents_llm::chat::{StreamChoice, StreamChunk, StreamDelta, StreamResponse};
    use autoagents_llm::error::GuardrailPhase;
    use autoagents_protocol::ActorID;
    use futures::StreamExt;

    #[derive(Debug)]
    struct LocalTool {
        name: String,
        output: serde_json::Value,
    }

    impl LocalTool {
        fn new(name: &str, output: serde_json::Value) -> Self {
            Self {
                name: name.to_string(),
                output,
            }
        }
    }

    impl crate::tool::ToolT for LocalTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "local tool"
        }

        fn args_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
    }

    #[async_trait]
    impl crate::tool::ToolRuntime for LocalTool {
        async fn execute(
            &self,
            _args: serde_json::Value,
        ) -> Result<serde_json::Value, crate::tool::ToolCallError> {
            Ok(self.output.clone())
        }
    }

    #[derive(Debug)]
    struct GuardrailRejectLLMProvider;

    fn guardrail_block_error() -> LLMError {
        LLMError::GuardrailBlocked {
            phase: GuardrailPhase::Input,
            guard: "prompt-injection".to_string().into(),
            rule_id: "prompt_injection_detected".to_string().into(),
            category: "prompt_injection".to_string().into(),
            severity: "high".to_string().into(),
            message: "detected suspicious instruction pattern: jailbreak"
                .to_string()
                .into(),
        }
    }

    #[async_trait]
    impl autoagents_llm::chat::ChatProvider for GuardrailRejectLLMProvider {
        async fn chat(
            &self,
            _messages: &[ChatMessage],
            _json_schema: Option<autoagents_llm::chat::StructuredOutputFormat>,
        ) -> Result<Box<dyn autoagents_llm::chat::ChatResponse>, LLMError> {
            Err(guardrail_block_error())
        }

        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[autoagents_llm::chat::Tool]>,
            _json_schema: Option<autoagents_llm::chat::StructuredOutputFormat>,
        ) -> Result<Box<dyn autoagents_llm::chat::ChatResponse>, LLMError> {
            Err(guardrail_block_error())
        }

        async fn chat_stream_struct(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[autoagents_llm::chat::Tool]>,
            _json_schema: Option<autoagents_llm::chat::StructuredOutputFormat>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>, LLMError>
        {
            Err(guardrail_block_error())
        }

        async fn chat_stream_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[autoagents_llm::chat::Tool]>,
            _json_schema: Option<autoagents_llm::chat::StructuredOutputFormat>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError>
        {
            Err(guardrail_block_error())
        }
    }

    #[async_trait]
    impl autoagents_llm::completion::CompletionProvider for GuardrailRejectLLMProvider {
        async fn complete(
            &self,
            _req: &autoagents_llm::completion::CompletionRequest,
            _json_schema: Option<autoagents_llm::chat::StructuredOutputFormat>,
        ) -> Result<autoagents_llm::completion::CompletionResponse, LLMError> {
            Ok(autoagents_llm::completion::CompletionResponse {
                text: String::default(),
            })
        }
    }

    #[async_trait]
    impl autoagents_llm::embedding::EmbeddingProvider for GuardrailRejectLLMProvider {
        async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
            Ok(Vec::new())
        }
    }

    #[async_trait]
    impl autoagents_llm::models::ModelsProvider for GuardrailRejectLLMProvider {}

    impl LLMProvider for GuardrailRejectLLMProvider {}

    fn context_with_memory(llm: Arc<dyn LLMProvider>) -> Context {
        let config = AgentConfig {
            id: ActorID::new_v4(),
            name: "memory_agent".to_string(),
            description: "desc".to_string(),
            output_schema: None,
        };
        let memory: Box<dyn MemoryProvider> = Box::new(SlidingWindowMemory::new(20));
        Context::new(llm, None)
            .with_config(config)
            .with_memory(Some(Arc::new(tokio::sync::Mutex::new(memory))))
    }

    async fn recalled_messages(context: &Context) -> Vec<ChatMessage> {
        let memory = context.memory().expect("memory should exist");
        memory
            .lock()
            .await
            .recall("", None)
            .await
            .expect("memory recall should succeed")
    }

    #[test]
    fn test_turn_engine_config_basic() {
        let config = TurnEngineConfig::basic(5);
        assert_eq!(config.max_turns, 5);
        assert!(matches!(config.tool_mode, ToolMode::Disabled));
        assert!(matches!(config.stream_mode, StreamMode::Structured));
        assert!(config.memory_policy.recall);
    }

    #[test]
    fn test_turn_engine_config_react() {
        let config = TurnEngineConfig::react(10);
        assert_eq!(config.max_turns, 10);
        assert!(matches!(config.tool_mode, ToolMode::Enabled));
        assert!(matches!(config.stream_mode, StreamMode::Tool));
        assert!(config.memory_policy.recall);
    }

    #[tokio::test]
    async fn test_run_turn_llm_error_does_not_store_user_message() {
        use crate::tests::MockAgentImpl;

        let llm: Arc<dyn LLMProvider> = Arc::new(GuardrailRejectLLMProvider);
        let context = context_with_memory(llm);
        let engine = TurnEngine::new(TurnEngineConfig::basic(1));
        let mut turn_state = engine.turn_state(&context);
        let task = Task::new("jailbreak");
        let hooks = MockAgentImpl::new("test", "test");

        let result = engine
            .run_turn(&hooks, &task, &context, &mut turn_state, 0, 1)
            .await;
        assert!(matches!(
            result,
            Err(TurnEngineError::LLMError(LLMError::GuardrailBlocked { .. }))
        ));

        let stored = recalled_messages(&context).await;
        assert!(stored.is_empty());
    }

    #[tokio::test]
    async fn test_run_turn_success_stores_user_once_in_memory() {
        use crate::tests::MockAgentImpl;

        let llm: Arc<dyn LLMProvider> = Arc::new(ConfigurableLLMProvider::default());
        let context = context_with_memory(llm);
        let engine = TurnEngine::new(TurnEngineConfig::basic(1));
        let mut turn_state = engine.turn_state(&context);
        let task = Task::new("hello");
        let hooks = MockAgentImpl::new("test", "test");

        let result = engine
            .run_turn(&hooks, &task, &context, &mut turn_state, 0, 1)
            .await;
        assert!(matches!(
            result,
            Ok(crate::agent::executor::TurnResult::Complete(_))
        ));

        let stored = recalled_messages(&context).await;
        let user_count = stored
            .iter()
            .filter(|m| m.role == ChatRole::User && m.content == "hello")
            .count();
        let assistant_count = stored
            .iter()
            .filter(|m| m.role == ChatRole::Assistant)
            .count();

        assert_eq!(user_count, 1);
        assert_eq!(assistant_count, 1);
    }

    #[test]
    fn test_normalize_max_turns_nonzero() {
        assert_eq!(normalize_max_turns(5, 10), 5);
    }

    #[test]
    fn test_normalize_max_turns_zero_uses_fallback() {
        assert_eq!(normalize_max_turns(0, 10), 10);
    }

    #[test]
    fn test_normalize_max_turns_zero_fallback_zero() {
        assert_eq!(normalize_max_turns(0, 0), 1);
    }

    #[test]
    fn test_should_include_user_prompt_no_memory() {
        let adapter = MemoryAdapter::new(None, MemoryPolicy::basic());
        assert!(should_include_user_prompt(&adapter, false));
    }

    #[test]
    fn test_should_include_user_prompt_recall_disabled() {
        let mut policy = MemoryPolicy::basic();
        policy.recall = false;
        let mem: Box<dyn crate::agent::memory::MemoryProvider> =
            Box::new(crate::agent::memory::SlidingWindowMemory::new(10));
        let adapter = MemoryAdapter::new(
            Some(std::sync::Arc::new(tokio::sync::Mutex::new(mem))),
            policy,
        );
        assert!(should_include_user_prompt(&adapter, false));
    }

    #[test]
    fn test_should_include_user_prompt_store_user_disabled() {
        let mut policy = MemoryPolicy::basic();
        policy.store_user = false;
        let mem: Box<dyn crate::agent::memory::MemoryProvider> =
            Box::new(crate::agent::memory::SlidingWindowMemory::new(10));
        let adapter = MemoryAdapter::new(
            Some(std::sync::Arc::new(tokio::sync::Mutex::new(mem))),
            policy,
        );
        assert!(should_include_user_prompt(&adapter, false));
    }

    #[test]
    fn test_should_include_user_prompt_already_stored() {
        let mem: Box<dyn crate::agent::memory::MemoryProvider> =
            Box::new(crate::agent::memory::SlidingWindowMemory::new(10));
        let adapter = MemoryAdapter::new(
            Some(std::sync::Arc::new(tokio::sync::Mutex::new(mem))),
            MemoryPolicy::basic(),
        );
        // stored_user = true => should not include
        assert!(!should_include_user_prompt(&adapter, true));
    }

    #[test]
    fn test_should_store_user_no_memory() {
        let state = TurnState {
            memory: MemoryAdapter::new(None, MemoryPolicy::basic()),
            stored_user: false,
        };
        assert!(!should_store_user(&state));
    }

    #[test]
    fn test_should_store_user_already_stored() {
        let mem: Box<dyn crate::agent::memory::MemoryProvider> =
            Box::new(crate::agent::memory::SlidingWindowMemory::new(10));
        let state = TurnState {
            memory: MemoryAdapter::new(
                Some(std::sync::Arc::new(tokio::sync::Mutex::new(mem))),
                MemoryPolicy::basic(),
            ),
            stored_user: true,
        };
        assert!(!should_store_user(&state));
    }

    #[test]
    fn test_user_message_text() {
        let task = Task::new("hello");
        let msg = user_message(&task);
        assert!(matches!(msg.role, ChatRole::User));
        assert!(matches!(msg.message_type, MessageType::Text));
        assert_eq!(msg.content, "hello");
    }

    #[test]
    fn test_user_message_image() {
        let mut task = Task::new("describe");
        task.image = Some((autoagents_protocol::ImageMime::PNG, vec![1, 2, 3]));
        let msg = user_message(&task);
        assert!(matches!(msg.role, ChatRole::User));
        assert!(matches!(msg.message_type, MessageType::Image(_)));
    }

    #[test]
    fn test_turn_state_new_and_mark_user_stored() {
        let config = AgentConfig {
            id: ActorID::new_v4(),
            name: "test".to_string(),
            description: "test".to_string(),
            output_schema: None,
        };
        let llm = std::sync::Arc::new(crate::tests::MockLLMProvider {});
        let context = Context::new(llm, None).with_config(config);

        let mut state = TurnState::new(&context, MemoryPolicy::basic());
        assert!(!state.stored_user());
        state.mark_user_stored();
        assert!(state.stored_user());
    }

    #[tokio::test]
    async fn test_build_messages_with_system_prompt() {
        let config = AgentConfig {
            id: ActorID::new_v4(),
            name: "test".to_string(),
            description: "default desc".to_string(),
            output_schema: None,
        };
        let llm = std::sync::Arc::new(crate::tests::MockLLMProvider {});
        let context = Context::new(llm, None).with_config(config);

        let engine = TurnEngine::new(TurnEngineConfig::basic(1));
        let adapter = MemoryAdapter::new(None, MemoryPolicy::basic());
        let mut task = Task::new("user input");
        task.system_prompt = Some("custom system".to_string());

        let messages = engine.build_messages(&context, &task, &adapter, true).await;
        // System + user
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "custom system");
        assert_eq!(messages[0].role, ChatRole::System);
        assert_eq!(messages[1].content, "user input");
    }

    #[tokio::test]
    async fn test_build_messages_without_user_prompt() {
        let config = AgentConfig {
            id: ActorID::new_v4(),
            name: "test".to_string(),
            description: "desc".to_string(),
            output_schema: None,
        };
        let llm = std::sync::Arc::new(crate::tests::MockLLMProvider {});
        let context = Context::new(llm, None).with_config(config);

        let engine = TurnEngine::new(TurnEngineConfig::basic(1));
        let adapter = MemoryAdapter::new(None, MemoryPolicy::basic());
        let task = Task::new("user input");

        let messages = engine
            .build_messages(&context, &task, &adapter, false)
            .await;
        // Only system prompt
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, ChatRole::System);
    }

    #[tokio::test]
    async fn test_run_turn_no_tools_single_turn() {
        use crate::tests::MockAgentImpl;
        let config = AgentConfig {
            id: ActorID::new_v4(),
            name: "test".to_string(),
            description: "test desc".to_string(),
            output_schema: None,
        };
        let llm = std::sync::Arc::new(crate::tests::MockLLMProvider {});
        let context = Context::new(llm, None).with_config(config);

        let engine = TurnEngine::new(TurnEngineConfig::basic(1));
        let mut turn_state = engine.turn_state(&context);
        let task = Task::new("test prompt");
        let hooks = MockAgentImpl::new("test", "test");

        let result = engine
            .run_turn(&hooks, &task, &context, &mut turn_state, 0, 1)
            .await;
        assert!(result.is_ok());
        let turn_result = result.unwrap();
        assert!(matches!(
            turn_result,
            crate::agent::executor::TurnResult::Complete(_)
        ));
        if let crate::agent::executor::TurnResult::Complete(output) = turn_result {
            assert_eq!(output.response, "Mock response");
        }
    }

    #[tokio::test]
    async fn test_run_turn_with_tool_calls_continues() {
        use crate::tests::MockAgentImpl;
        let tool_call = ToolCall {
            id: "call_1".to_string(),
            call_type: "function".to_string(),
            function: autoagents_llm::FunctionCall {
                name: "tool_a".to_string(),
                arguments: r#"{"value":1}"#.to_string(),
            },
        };

        let llm = Arc::new(ConfigurableLLMProvider {
            chat_response: StaticChatResponse {
                text: Some("Use tool".to_string()),
                tool_calls: Some(vec![tool_call.clone()]),
                usage: None,
                thinking: None,
            },
            ..ConfigurableLLMProvider::default()
        });

        let config = AgentConfig {
            id: ActorID::new_v4(),
            name: "tool_agent".to_string(),
            description: "desc".to_string(),
            output_schema: None,
        };
        let tool = LocalTool::new("tool_a", serde_json::json!({"ok": true}));
        let context = Context::new(llm, None)
            .with_config(config)
            .with_tools(vec![Box::new(tool)]);

        let engine = TurnEngine::new(TurnEngineConfig {
            max_turns: 2,
            tool_mode: ToolMode::Enabled,
            stream_mode: StreamMode::Structured,
            memory_policy: MemoryPolicy::basic(),
        });
        let mut turn_state = engine.turn_state(&context);
        let task = Task::new("prompt");
        let hooks = MockAgentImpl::new("test", "test");

        let result = engine
            .run_turn(&hooks, &task, &context, &mut turn_state, 0, 2)
            .await
            .unwrap();

        match result {
            crate::agent::executor::TurnResult::Continue(Some(output)) => {
                assert_eq!(output.response, "Use tool");
                assert_eq!(output.tool_calls.len(), 1);
                assert!(output.tool_calls[0].success);
            }
            _ => panic!("expected Continue(Some)"),
        }

        #[cfg(not(target_arch = "wasm32"))]
        if let Ok(state) = context.state().try_lock() {
            assert_eq!(state.tool_calls.len(), 1);
        }
    }

    #[tokio::test]
    async fn test_run_turn_tool_mode_disabled_ignores_tool_calls() {
        use crate::tests::MockAgentImpl;
        let tool_call = ToolCall {
            id: "call_1".to_string(),
            call_type: "function".to_string(),
            function: autoagents_llm::FunctionCall {
                name: "tool_a".to_string(),
                arguments: r#"{"value":1}"#.to_string(),
            },
        };

        let llm = Arc::new(ConfigurableLLMProvider {
            chat_response: StaticChatResponse {
                text: Some("No tools".to_string()),
                tool_calls: Some(vec![tool_call]),
                usage: None,
                thinking: None,
            },
            ..ConfigurableLLMProvider::default()
        });

        let config = AgentConfig {
            id: ActorID::new_v4(),
            name: "tool_agent".to_string(),
            description: "desc".to_string(),
            output_schema: None,
        };
        let context = Context::new(llm, None).with_config(config);

        let engine = TurnEngine::new(TurnEngineConfig {
            max_turns: 1,
            tool_mode: ToolMode::Disabled,
            stream_mode: StreamMode::Structured,
            memory_policy: MemoryPolicy::basic(),
        });
        let mut turn_state = engine.turn_state(&context);
        let task = Task::new("prompt");
        let hooks = MockAgentImpl::new("test", "test");

        let result = engine
            .run_turn(&hooks, &task, &context, &mut turn_state, 0, 1)
            .await
            .unwrap();

        match result {
            crate::agent::executor::TurnResult::Complete(output) => {
                assert_eq!(output.response, "No tools");
                assert!(output.tool_calls.is_empty());
            }
            _ => panic!("expected Complete"),
        }
    }

    #[tokio::test]
    async fn test_run_turn_propagates_reasoning_content() {
        use crate::tests::MockAgentImpl;

        let llm = Arc::new(ConfigurableLLMProvider {
            chat_response: StaticChatResponse {
                text: Some("answer".to_string()),
                tool_calls: None,
                usage: None,
                thinking: Some("reasoning".to_string()),
            },
            ..ConfigurableLLMProvider::default()
        });

        let config = AgentConfig {
            id: ActorID::new_v4(),
            name: "reasoning_agent".to_string(),
            description: "desc".to_string(),
            output_schema: None,
        };
        let context = Context::new(llm, None).with_config(config);
        let engine = TurnEngine::new(TurnEngineConfig::basic(1));
        let mut turn_state = engine.turn_state(&context);
        let task = Task::new("prompt");
        let hooks = MockAgentImpl::new("test", "test");

        let result = engine
            .run_turn(&hooks, &task, &context, &mut turn_state, 0, 1)
            .await
            .unwrap();

        match result {
            crate::agent::executor::TurnResult::Complete(output) => {
                assert_eq!(output.response, "answer");
                assert_eq!(output.reasoning_content, "reasoning");
            }
            _ => panic!("expected Complete"),
        }
    }

    #[tokio::test]
    async fn test_run_turn_stream_structured_aggregates_text() {
        use crate::tests::MockAgentImpl;
        let llm = Arc::new(ConfigurableLLMProvider {
            structured_stream: vec![
                StreamResponse {
                    choices: vec![StreamChoice {
                        delta: StreamDelta {
                            content: Some("Hello ".to_string()),
                            reasoning_content: None,
                            tool_calls: None,
                        },
                    }],
                    usage: None,
                },
                StreamResponse {
                    choices: vec![StreamChoice {
                        delta: StreamDelta {
                            content: Some("world".to_string()),
                            reasoning_content: None,
                            tool_calls: None,
                        },
                    }],
                    usage: None,
                },
            ],
            ..ConfigurableLLMProvider::default()
        });

        let config = AgentConfig {
            id: ActorID::new_v4(),
            name: "stream_agent".to_string(),
            description: "desc".to_string(),
            output_schema: None,
        };
        let context = Arc::new(Context::new(llm, None).with_config(config));
        let engine = TurnEngine::new(TurnEngineConfig {
            max_turns: 1,
            tool_mode: ToolMode::Disabled,
            stream_mode: StreamMode::Structured,
            memory_policy: MemoryPolicy::basic(),
        });
        let mut turn_state = engine.turn_state(&context);
        let task = Task::new("prompt");
        let hooks = MockAgentImpl::new("test", "test");

        let mut stream = engine
            .run_turn_stream(hooks, &task, context, &mut turn_state, 0, 1)
            .await
            .unwrap();

        let mut final_text = String::default();
        while let Some(delta) = stream.next().await {
            if let Ok(TurnDelta::Done(result)) = delta {
                final_text = match result {
                    crate::agent::executor::TurnResult::Complete(output) => output.response,
                    crate::agent::executor::TurnResult::Continue(Some(output)) => output.response,
                    crate::agent::executor::TurnResult::Continue(None) => String::default(),
                };
                break;
            }
        }

        assert_eq!(final_text, "Hello world");
    }

    #[tokio::test]
    async fn test_run_turn_stream_structured_emits_reasoning_content() {
        use crate::tests::MockAgentImpl;
        let llm = Arc::new(ConfigurableLLMProvider {
            structured_stream: vec![StreamResponse {
                choices: vec![StreamChoice {
                    delta: StreamDelta {
                        content: None,
                        reasoning_content: Some("think".to_string()),
                        tool_calls: None,
                    },
                }],
                usage: None,
            }],
            ..ConfigurableLLMProvider::default()
        });

        let config = AgentConfig {
            id: ActorID::new_v4(),
            name: "stream_reasoning_agent".to_string(),
            description: "desc".to_string(),
            output_schema: None,
        };
        let context = Arc::new(Context::new(llm, None).with_config(config));
        let engine = TurnEngine::new(TurnEngineConfig::basic(1));
        let mut turn_state = engine.turn_state(&context);
        let task = Task::new("prompt");
        let hooks = MockAgentImpl::new("test", "test");

        let mut stream = engine
            .run_turn_stream(hooks, &task, context, &mut turn_state, 0, 1)
            .await
            .unwrap();

        let mut saw_delta = false;
        let mut final_reasoning = String::default();
        while let Some(delta) = stream.next().await {
            match delta {
                Ok(TurnDelta::ReasoningContent(text)) => {
                    saw_delta = true;
                    assert_eq!(text, "think");
                }
                Ok(TurnDelta::Done(result)) => {
                    final_reasoning = match result {
                        crate::agent::executor::TurnResult::Complete(output) => {
                            output.reasoning_content
                        }
                        crate::agent::executor::TurnResult::Continue(Some(output)) => {
                            output.reasoning_content
                        }
                        crate::agent::executor::TurnResult::Continue(None) => String::default(),
                    };
                    break;
                }
                _ => {}
            }
        }

        assert!(saw_delta);
        assert_eq!(final_reasoning, "think");
    }

    #[tokio::test]
    async fn test_run_turn_stream_with_tools_executes_tools() {
        use crate::tests::MockAgentImpl;
        let tool_call = ToolCall {
            id: "call_1".to_string(),
            call_type: "function".to_string(),
            function: autoagents_llm::FunctionCall {
                name: "tool_a".to_string(),
                arguments: r#"{"value":1}"#.to_string(),
            },
        };

        let llm = Arc::new(ConfigurableLLMProvider {
            stream_chunks: vec![
                StreamChunk::Text("thinking".to_string()),
                StreamChunk::ToolUseComplete {
                    index: 0,
                    tool_call: tool_call.clone(),
                },
                StreamChunk::Done {
                    stop_reason: "tool_use".to_string(),
                },
            ],
            ..ConfigurableLLMProvider::default()
        });

        let config = AgentConfig {
            id: ActorID::new_v4(),
            name: "tool_stream_agent".to_string(),
            description: "desc".to_string(),
            output_schema: None,
        };
        let tool = LocalTool::new("tool_a", serde_json::json!({"ok": true}));
        let context = Arc::new(
            Context::new(llm, None)
                .with_config(config)
                .with_tools(vec![Box::new(tool)]),
        );
        let engine = TurnEngine::new(TurnEngineConfig {
            max_turns: 1,
            tool_mode: ToolMode::Enabled,
            stream_mode: StreamMode::Tool,
            memory_policy: MemoryPolicy::basic(),
        });
        let mut turn_state = engine.turn_state(&context);
        let task = Task::new("prompt");
        let hooks = MockAgentImpl::new("test", "test");

        let mut stream = engine
            .run_turn_stream(hooks, &task, context, &mut turn_state, 0, 1)
            .await
            .unwrap();

        let mut final_result = None;
        while let Some(delta) = stream.next().await {
            if let Ok(TurnDelta::Done(result)) = delta {
                final_result = Some(result);
                break;
            }
        }

        match final_result.expect("done") {
            crate::agent::executor::TurnResult::Continue(Some(output)) => {
                assert_eq!(output.tool_calls.len(), 1);
                assert!(output.tool_calls[0].success);
            }
            _ => panic!("expected Continue(Some)"),
        }
    }

    #[tokio::test]
    async fn test_run_turn_stream_llm_error_does_not_store_user_message() {
        use crate::tests::MockAgentImpl;

        let llm: Arc<dyn LLMProvider> = Arc::new(GuardrailRejectLLMProvider);
        let context = Arc::new(context_with_memory(llm));
        let engine = TurnEngine::new(TurnEngineConfig::basic(1));
        let mut turn_state = engine.turn_state(&context);
        let task = Task::new("jailbreak");
        let hooks = MockAgentImpl::new("test", "test");

        let mut stream = engine
            .run_turn_stream(hooks, &task, context.clone(), &mut turn_state, 0, 1)
            .await
            .expect("stream should initialize");

        let first = stream
            .next()
            .await
            .expect("stream should emit an error event");
        assert!(matches!(
            first,
            Err(TurnEngineError::LLMError(LLMError::GuardrailBlocked { .. }))
        ));

        let stored = recalled_messages(&context).await;
        assert!(stored.is_empty());
    }

    #[tokio::test]
    async fn test_run_turn_stream_success_stores_user_once_in_memory() {
        use crate::tests::MockAgentImpl;

        let llm: Arc<dyn LLMProvider> = Arc::new(ConfigurableLLMProvider {
            structured_stream: vec![StreamResponse {
                choices: vec![StreamChoice {
                    delta: StreamDelta {
                        content: Some("hello".to_string()),
                        reasoning_content: None,
                        tool_calls: None,
                    },
                }],
                usage: None,
            }],
            ..ConfigurableLLMProvider::default()
        });
        let context = Arc::new(context_with_memory(llm));
        let engine = TurnEngine::new(TurnEngineConfig::basic(1));
        let mut turn_state = engine.turn_state(&context);
        let task = Task::new("hello");
        let hooks = MockAgentImpl::new("test", "test");

        let mut stream = engine
            .run_turn_stream(hooks, &task, context.clone(), &mut turn_state, 0, 1)
            .await
            .expect("stream should initialize");

        while let Some(delta) = stream.next().await {
            if matches!(delta, Ok(TurnDelta::Done(_))) {
                break;
            }
        }

        let stored = recalled_messages(&context).await;
        let user_count = stored
            .iter()
            .filter(|m| m.role == ChatRole::User && m.content == "hello")
            .count();
        let assistant_count = stored
            .iter()
            .filter(|m| m.role == ChatRole::Assistant)
            .count();

        assert_eq!(user_count, 1);
        assert_eq!(assistant_count, 1);
    }
}
