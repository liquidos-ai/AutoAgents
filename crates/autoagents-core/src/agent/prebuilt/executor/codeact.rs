use crate::agent::executor::event_helper::EventHelper;
use crate::agent::executor::memory_policy::{MemoryAdapter, MemoryPolicy};
use crate::agent::executor::tool_processor::ToolProcessor;
use crate::agent::executor::turn_engine::record_task_state;
use crate::agent::task::Task;
use crate::agent::{AgentDeriveT, AgentExecutor, AgentHooks, Context, ExecutorConfig, HookOutcome};
use crate::channel::channel;
use crate::tool::{ToolCallResult, ToolT};
use crate::utils::stream_from_producer;
use async_trait::async_trait;
use autoagents_llm::ToolCall;
use autoagents_llm::chat::{ChatMessage, ChatRole, FunctionTool, MessageType, StreamChunk, Tool};
use autoagents_llm::error::LLMError;
use autoagents_protocol::{Event, SubmissionId};
#[cfg(target_arch = "wasm32")]
use futures::SinkExt;
use futures::Stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
use std::time::Duration;
use std::time::Instant;
use thiserror::Error;

#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::mpsc;

#[cfg(target_arch = "wasm32")]
use futures::channel::mpsc;

use deno_ast::swc::ast::{
    BlockStmtOrExpr, CallExpr, Callee, Decl, Expr, MemberExpr, MemberProp, MetaPropExpr,
    MetaPropKind, ModuleItem, Pat, Stmt,
};
use deno_ast::swc::ecma_visit::{Visit, VisitWith};
use deno_ast::{
    EmitOptions, MediaType, ModuleSpecifier, ParseParams, ParsedSource, ProgramRef, SourceRange,
    TranspileModuleOptions, TranspileOptions, parse_program,
};
#[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
use rquickjs::function::Func;
#[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
use rquickjs::{CatchResultExt, Context as JsContext, Promise, Runtime as JsRuntime};

const CODEACT_TOOL_NAME: &str = "execute_typescript";
const CODEACT_LANGUAGE: &str = "typescript";
const CODEACT_WRAPPER_PREFIX: &str = "const __codeact_main = async () => {\n";
const CODEACT_WRAPPER_SUFFIX: &str = "\n};\n";

/// Runtime limits applied to a single CodeAct TypeScript execution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodeActSandboxLimits {
    pub timeout_ms: u64,
    pub memory_limit_bytes: usize,
    pub max_source_bytes: usize,
    pub max_console_bytes: usize,
    pub max_tool_calls_per_execution: usize,
    pub max_concurrent_tool_calls: usize,
}

impl Default for CodeActSandboxLimits {
    fn default() -> Self {
        Self {
            timeout_ms: 10_000,
            memory_limit_bytes: 32 * 1024 * 1024,
            max_source_bytes: 64 * 1024,
            max_console_bytes: 32 * 1024,
            max_tool_calls_per_execution: 32,
            max_concurrent_tool_calls: 8,
        }
    }
}

/// Captured metadata for a single `execute_typescript` run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeActExecutionRecord {
    pub execution_id: String,
    pub source: String,
    pub console: Vec<String>,
    pub tool_calls: Vec<ToolCallResult>,
    pub result: Option<Value>,
    pub success: bool,
    pub error: Option<String>,
    pub duration_ms: u64,
}

/// Output of the CodeAct executor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeActAgentOutput {
    pub response: String,
    pub executions: Vec<CodeActExecutionRecord>,
    pub done: bool,
}

impl From<CodeActAgentOutput> for Value {
    fn from(output: CodeActAgentOutput) -> Self {
        serde_json::to_value(output).unwrap_or(Value::Null)
    }
}

impl From<CodeActAgentOutput> for String {
    fn from(output: CodeActAgentOutput) -> Self {
        output.response
    }
}

impl CodeActAgentOutput {
    pub fn try_parse<T: for<'de> serde::Deserialize<'de>>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_str::<T>(&self.response)
    }

    pub fn parse_or_map<T, F>(&self, fallback: F) -> T
    where
        T: for<'de> serde::Deserialize<'de>,
        F: FnOnce(&str) -> T,
    {
        self.try_parse::<T>()
            .unwrap_or_else(|_| fallback(&self.response))
    }

    #[allow(clippy::result_large_err)]
    pub fn extract_agent_output<T>(val: Value) -> Result<T, CodeActExecutorError>
    where
        T: for<'de> serde::Deserialize<'de>,
    {
        let output: Self = serde_json::from_value(val)
            .map_err(|e| CodeActExecutorError::AgentOutputError(e.to_string()))?;
        serde_json::from_str(&output.response)
            .map_err(|e| CodeActExecutorError::AgentOutputError(e.to_string()))
    }
}

#[derive(Error, Debug)]
pub enum CodeActExecutorError {
    #[error("LLM error: {0}")]
    LLMError(
        #[from]
        #[source]
        LLMError,
    ),

    #[error("Maximum turns exceeded: {max_turns}")]
    MaxTurnsExceeded { max_turns: usize },

    #[error("Sandbox error: {0}")]
    SandboxError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Transpile error: {0}")]
    TranspileError(String),

    #[error("Unsupported platform: {0}")]
    UnsupportedPlatform(String),

    #[error("Other error: {0}")]
    Other(String),

    #[error("Extracting Agent Output Error: {0}")]
    AgentOutputError(String),
}

#[derive(Debug)]
pub struct CodeActAgent<T: AgentDeriveT> {
    inner: Arc<T>,
    max_turns: usize,
    sandbox_limits: CodeActSandboxLimits,
}

impl<T: AgentDeriveT> Clone for CodeActAgent<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            max_turns: self.max_turns,
            sandbox_limits: self.sandbox_limits.clone(),
        }
    }
}

impl<T: AgentDeriveT> CodeActAgent<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner: Arc::new(inner),
            max_turns: 10,
            sandbox_limits: CodeActSandboxLimits::default(),
        }
    }

    pub fn with_max_turns(inner: T, max_turns: usize) -> Self {
        Self {
            inner: Arc::new(inner),
            max_turns: max_turns.max(1),
            sandbox_limits: CodeActSandboxLimits::default(),
        }
    }

    pub fn with_sandbox_limits(mut self, sandbox_limits: CodeActSandboxLimits) -> Self {
        self.sandbox_limits = sandbox_limits;
        self
    }
}

impl<T: AgentDeriveT> Deref for CodeActAgent<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[async_trait]
impl<T: AgentDeriveT> AgentDeriveT for CodeActAgent<T> {
    type Output = <T as AgentDeriveT>::Output;

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn output_schema(&self) -> Option<Value> {
        self.inner.output_schema()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn tools(&self) -> Vec<Box<dyn ToolT>> {
        self.inner.tools()
    }
}

#[async_trait]
impl<T> AgentHooks for CodeActAgent<T>
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

#[async_trait]
impl<T> AgentExecutor for CodeActAgent<T>
where
    T: AgentDeriveT + AgentHooks + Send + Sync + 'static,
{
    type Output = CodeActAgentOutput;
    type Error = CodeActExecutorError;

    fn config(&self) -> ExecutorConfig {
        ExecutorConfig {
            max_turns: self.max_turns,
        }
    }

    async fn execute(
        &self,
        task: &Task,
        context: Arc<Context>,
    ) -> Result<Self::Output, Self::Error> {
        #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
        {
            let _ = task;
            let _ = context;
            return Err(CodeActExecutorError::UnsupportedPlatform(
                "CodeAct is supported on native and WASI wasm targets; browser wasm is not supported"
                    .to_string(),
            ));
        }

        #[cfg(not(all(target_arch = "wasm32", not(target_os = "wasi"))))]
        {
            let engine = CodeActEngine::new(self.config().max_turns, self.sandbox_limits.clone());
            engine.execute(self.clone(), task, context).await
        }
    }

    async fn execute_stream(
        &self,
        task: &Task,
        context: Arc<Context>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Self::Output, Self::Error>> + Send>>, Self::Error>
    {
        #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
        {
            let _ = task;
            let _ = context;
            return Err(CodeActExecutorError::UnsupportedPlatform(
                "CodeAct is supported on native and WASI wasm targets; browser wasm is not supported"
                    .to_string(),
            ));
        }

        #[cfg(not(all(target_arch = "wasm32", not(target_os = "wasi"))))]
        {
            let engine = CodeActEngine::new(self.config().max_turns, self.sandbox_limits.clone());
            engine.execute_stream(self.clone(), task, context).await
        }
    }
}

#[derive(Debug, Clone)]
struct CodeActEngine {
    max_turns: usize,
    sandbox_limits: CodeActSandboxLimits,
}

#[derive(Clone)]
struct CodeActExecutionContext {
    context: Arc<Context>,
    submission_id: SubmissionId,
    tx_event: Option<mpsc::Sender<Event>>,
    tool_bindings: Vec<CodeActToolBinding>,
}

impl CodeActExecutionContext {
    fn new(
        context: Arc<Context>,
        submission_id: SubmissionId,
        tx_event: Option<mpsc::Sender<Event>>,
        tool_bindings: Vec<CodeActToolBinding>,
    ) -> Self {
        Self {
            context,
            submission_id,
            tx_event,
            tool_bindings,
        }
    }

    fn sandbox_request(
        &self,
        execution_id: String,
        source: String,
        limits: CodeActSandboxLimits,
    ) -> CodeActSandboxRequest {
        CodeActSandboxRequest {
            context: Arc::clone(&self.context),
            submission_id: self.submission_id,
            tx_event: self.tx_event.clone(),
            execution_id,
            source,
            limits,
            tool_bindings: self.tool_bindings.clone(),
        }
    }
}

#[derive(Clone)]
struct CodeActSandboxRequest {
    context: Arc<Context>,
    submission_id: SubmissionId,
    tx_event: Option<mpsc::Sender<Event>>,
    execution_id: String,
    source: String,
    limits: CodeActSandboxLimits,
    tool_bindings: Vec<CodeActToolBinding>,
}

impl CodeActEngine {
    fn new(max_turns: usize, sandbox_limits: CodeActSandboxLimits) -> Self {
        Self {
            max_turns: max_turns.max(1),
            sandbox_limits,
        }
    }

    async fn execute<H>(
        &self,
        hooks: H,
        task: &Task,
        context: Arc<Context>,
    ) -> Result<CodeActAgentOutput, CodeActExecutorError>
    where
        H: AgentHooks + Clone + Send + Sync + 'static,
    {
        record_task_state(&context, task);

        let tx_event = context.tx().ok();
        EventHelper::send_task_started(
            &tx_event,
            task.submission_id,
            context.config().id,
            context.config().name.clone(),
            task.prompt.clone(),
        )
        .await;

        let memory = MemoryAdapter::new(context.memory(), MemoryPolicy::codeact());
        let tool_bindings = build_tool_bindings(context.tools())?;
        let execution_context = CodeActExecutionContext::new(
            Arc::clone(&context),
            task.submission_id,
            tx_event.clone(),
            tool_bindings,
        );
        let execute_tool = codeact_execute_tool();
        let tools = Arc::new(vec![execute_tool]);
        let max_turns = self.max_turns;
        let mut stored_user = false;
        let mut final_response = String::default();
        let mut executions = Vec::new();

        for turn_index in 0..max_turns {
            EventHelper::send_turn_started(
                &tx_event,
                task.submission_id,
                context.config().id,
                turn_index,
                max_turns,
            )
            .await;
            hooks.on_turn_start(turn_index, &context).await;

            let messages = build_messages(
                &context,
                task,
                &memory,
                stored_user,
                &execution_context.tool_bindings,
            )
            .await;
            let should_store_user = should_store_user(&memory, stored_user);
            let response = context
                .llm()
                .chat_with_tools(
                    &messages,
                    Some(tools.as_slice()),
                    context.config().output_schema.clone(),
                )
                .await?;
            let response_text = response.text().unwrap_or_default();
            if should_store_user {
                memory.store_user(task).await;
                stored_user = true;
            }

            let tool_calls = response.tool_calls().unwrap_or_default();
            if tool_calls.is_empty() {
                if !response_text.is_empty() {
                    memory.store_assistant(&response_text).await;
                    final_response = response_text;
                }

                EventHelper::send_turn_completed(
                    &tx_event,
                    task.submission_id,
                    context.config().id,
                    turn_index,
                    true,
                )
                .await;
                hooks.on_turn_complete(turn_index, &context).await;

                return Ok(CodeActAgentOutput {
                    response: final_response,
                    executions,
                    done: true,
                });
            }

            if !response_text.is_empty() {
                final_response.clone_from(&response_text);
            }

            let tool_results = self
                .process_code_calls(
                    hooks.clone(),
                    &execution_context,
                    &tool_calls,
                    &mut executions,
                )
                .await;

            memory
                .store_tool_interaction(&tool_calls, &tool_results, &response_text)
                .await;

            EventHelper::send_turn_completed(
                &tx_event,
                task.submission_id,
                context.config().id,
                turn_index,
                false,
            )
            .await;
            hooks.on_turn_complete(turn_index, &context).await;
        }

        Err(CodeActExecutorError::MaxTurnsExceeded { max_turns })
    }

    async fn execute_stream<H>(
        &self,
        hooks: H,
        task: &Task,
        context: Arc<Context>,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<CodeActAgentOutput, CodeActExecutorError>> + Send>>,
        CodeActExecutorError,
    >
    where
        H: AgentHooks + Clone + Send + Sync + 'static,
    {
        record_task_state(&context, task);

        let tx_event = context.tx().ok();
        EventHelper::send_task_started(
            &tx_event,
            task.submission_id,
            context.config().id,
            context.config().name.clone(),
            task.prompt.clone(),
        )
        .await;

        let tool_bindings = build_tool_bindings(context.tools())?;
        let execution_context = CodeActExecutionContext::new(
            context.clone(),
            task.submission_id,
            tx_event.clone(),
            tool_bindings,
        );
        let execute_tool = codeact_execute_tool();
        let tools = Arc::new(vec![execute_tool]);
        let max_turns = self.max_turns;
        let memory = MemoryAdapter::new(context.memory(), MemoryPolicy::codeact());
        let task = task.clone();
        let context_clone = context.clone();
        let engine = self.clone();

        #[cfg_attr(not(target_arch = "wasm32"), allow(unused_mut))]
        let (mut tx, rx) = channel::<Result<CodeActAgentOutput, CodeActExecutorError>>(100);

        let producer = async move {
            let mut stored_user = false;
            let mut final_response = String::default();
            let mut executions = Vec::new();

            for turn_index in 0..max_turns {
                EventHelper::send_turn_started(
                    &tx_event,
                    task.submission_id,
                    context_clone.config().id,
                    turn_index,
                    max_turns,
                )
                .await;
                hooks.on_turn_start(turn_index, &context_clone).await;

                let messages = build_messages(
                    &context_clone,
                    &task,
                    &memory,
                    stored_user,
                    &execution_context.tool_bindings,
                )
                .await;
                let should_store_user = should_store_user(&memory, stored_user);
                let stream = context_clone
                    .llm()
                    .chat_stream_with_tools(
                        &messages,
                        Some(tools.as_slice()),
                        context_clone.config().output_schema.clone(),
                    )
                    .await;

                let mut response_text = String::default();
                let mut tool_calls = Vec::new();
                let mut seen_tool_ids = HashSet::new();

                let mut stream = match stream {
                    Ok(stream) => stream,
                    Err(err) => {
                        let _ = tx.send(Err(err.into())).await;
                        return;
                    }
                };

                if should_store_user {
                    memory.store_user(&task).await;
                    stored_user = true;
                }

                while let Some(chunk_result) = stream.next().await {
                    let chunk = match chunk_result {
                        Ok(chunk) => chunk,
                        Err(err) => {
                            let _ = tx.send(Err(err.into())).await;
                            return;
                        }
                    };

                    let chunk_clone = chunk.clone();
                    match chunk {
                        StreamChunk::Text(content) => {
                            response_text.push_str(&content);
                            let _ = tx
                                .send(Ok(CodeActAgentOutput {
                                    response: content,
                                    executions: Vec::new(),
                                    done: false,
                                }))
                                .await;
                        }
                        StreamChunk::ReasoningContent(_) => {}
                        StreamChunk::ToolUseComplete { tool_call, .. } => {
                            if seen_tool_ids.insert(tool_call.id.clone()) {
                                tool_calls.push(tool_call.clone());
                                let value = serde_json::to_value(tool_call).unwrap_or(Value::Null);
                                EventHelper::send_stream_tool_call(
                                    &tx_event,
                                    task.submission_id,
                                    value,
                                )
                                .await;
                            }
                        }
                        StreamChunk::Usage(_) => {}
                        StreamChunk::Done { .. }
                        | StreamChunk::ToolUseStart { .. }
                        | StreamChunk::ToolUseInputDelta { .. } => {}
                    }

                    EventHelper::send_stream_chunk(&tx_event, task.submission_id, chunk_clone)
                        .await;
                }

                if tool_calls.is_empty() {
                    if !response_text.is_empty() {
                        memory.store_assistant(&response_text).await;
                        final_response = response_text;
                    }

                    EventHelper::send_turn_completed(
                        &tx_event,
                        task.submission_id,
                        context_clone.config().id,
                        turn_index,
                        true,
                    )
                    .await;
                    hooks.on_turn_complete(turn_index, &context_clone).await;

                    let output = CodeActAgentOutput {
                        response: final_response.clone(),
                        executions: executions.clone(),
                        done: true,
                    };
                    let _ = tx.send(Ok(output.clone())).await;

                    EventHelper::send_stream_complete(&tx_event, task.submission_id).await;
                    let result = serde_json::to_string_pretty(&output)
                        .unwrap_or_else(|_| output.response.clone());
                    EventHelper::send_task_completed(
                        &tx_event,
                        task.submission_id,
                        context_clone.config().id,
                        context_clone.config().name.clone(),
                        result,
                    )
                    .await;
                    return;
                }

                if !response_text.is_empty() {
                    final_response.clone_from(&response_text);
                }

                let tool_results = engine
                    .process_code_calls(
                        hooks.clone(),
                        &execution_context,
                        &tool_calls,
                        &mut executions,
                    )
                    .await;

                memory
                    .store_tool_interaction(&tool_calls, &tool_results, &response_text)
                    .await;

                let _ = tx
                    .send(Ok(CodeActAgentOutput {
                        response: String::default(),
                        executions: executions.clone(),
                        done: false,
                    }))
                    .await;

                EventHelper::send_turn_completed(
                    &tx_event,
                    task.submission_id,
                    context_clone.config().id,
                    turn_index,
                    false,
                )
                .await;
                hooks.on_turn_complete(turn_index, &context_clone).await;
            }

            let _ = tx
                .send(Err(CodeActExecutorError::MaxTurnsExceeded { max_turns }))
                .await;
        };

        Ok(stream_from_producer(rx, producer))
    }

    async fn process_code_calls<H>(
        &self,
        hooks: H,
        execution_context: &CodeActExecutionContext,
        tool_calls: &[ToolCall],
        executions: &mut Vec<CodeActExecutionRecord>,
    ) -> Vec<ToolCallResult>
    where
        H: AgentHooks + Clone + Send + Sync + 'static,
    {
        let mut tool_results = Vec::with_capacity(tool_calls.len());

        for tool_call in tool_calls {
            let args = parse_execute_args(&tool_call.function.arguments);
            let record = match args {
                Ok(args) => {
                    let request = execution_context.sandbox_request(
                        tool_call.id.clone(),
                        args.code,
                        self.sandbox_limits.clone(),
                    );
                    execute_typescript_sandbox(hooks.clone(), request).await
                }
                Err(error) => failed_execution_record(
                    tool_call.id.clone(),
                    String::default(),
                    error,
                    Vec::new(),
                    Vec::new(),
                    0,
                ),
            };

            record_nested_tool_calls_state(&execution_context.context, &record.tool_calls);
            executions.push(record.clone());

            tool_results.push(ToolCallResult {
                tool_name: CODEACT_TOOL_NAME.to_string(),
                success: record.success,
                arguments: serde_json::from_str(&tool_call.function.arguments)
                    .unwrap_or(Value::Null),
                result: serde_json::to_value(&record).unwrap_or_else(
                    |_| json!({"success": false, "error": "failed to serialize execution record"}),
                ),
            });
        }

        tool_results
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ExecuteTypescriptArgs {
    code: String,
}

#[derive(Debug, Clone)]
struct CodeActToolBinding {
    original_name: String,
    js_name: String,
    args_type_alias: String,
    args_type: String,
    result_type_alias: String,
    result_type: String,
    description: String,
}

async fn build_messages(
    context: &Context,
    task: &Task,
    memory: &MemoryAdapter,
    stored_user: bool,
    tool_bindings: &[CodeActToolBinding],
) -> Vec<ChatMessage> {
    let system_prompt = task
        .system_prompt
        .as_deref()
        .unwrap_or_else(|| &context.config().description);
    let mut messages = vec![ChatMessage {
        role: ChatRole::System,
        message_type: MessageType::Text,
        content: build_codeact_system_prompt(system_prompt, tool_bindings),
    }];

    let recalled = memory.recall_messages(task).await;
    messages.extend(recalled);

    if should_include_user_prompt(memory, stored_user) {
        messages.push(user_message(task));
    }

    messages
}

fn build_codeact_system_prompt(base_prompt: &str, tool_bindings: &[CodeActToolBinding]) -> String {
    let declarations = tool_bindings
        .iter()
        .map(render_typescript_binding_declaration)
        .collect::<String>();

    format!(
        "{base_prompt}\n\n\
You are operating in CodeAct mode.\n\
- When you need to inspect files, compute, transform data, or compose tools, call the `{CODEACT_TOOL_NAME}` tool with TypeScript code.\n\
- Prefer solving the task in a single `{CODEACT_TOOL_NAME}` call when one script is sufficient.\n\
- Only use the provided `external_*` functions. Do not import modules, export symbols, access the network, spawn processes, or assume filesystem/process globals exist.\n\
- Each `{CODEACT_TOOL_NAME}` run executes in a fresh sandbox. Persist state explicitly through tool results or your final answer.\n\
- Code must return a JSON-serializable value. Use `console.log` only for debugging.\n\
- The script's top-level result is what the sandbox returns. End with `return ...;` or a single trailing expression such as `compute();`. Do not start async work and ignore its promise.\n\
- After you have enough information, stop calling tools and answer the user directly.\n\n\
Available TypeScript bindings:\n```ts\n{declarations}```"
    )
}

fn build_tool_bindings(
    tools: &[Box<dyn ToolT>],
) -> Result<Vec<CodeActToolBinding>, CodeActExecutorError> {
    let mut seen = HashSet::new();
    let mut bindings = Vec::with_capacity(tools.len());

    for tool in tools {
        let js_name = format!("external_{}", sanitize_identifier(tool.name()));
        if !seen.insert(js_name.clone()) {
            return Err(CodeActExecutorError::ValidationError(format!(
                "tool name collision after sanitization for '{}'",
                tool.name()
            )));
        }

        let args_type_alias = format!("{}Args", sanitize_identifier(tool.name()));
        let args_type = schema_to_typescript(&tool.args_schema());
        let result_type_alias = format!("{}Result", sanitize_identifier(tool.name()));
        let result_type = tool
            .output_schema()
            .as_ref()
            .map(schema_to_typescript)
            .unwrap_or_else(|| "unknown".to_string());
        bindings.push(CodeActToolBinding {
            original_name: tool.name().to_string(),
            js_name,
            args_type_alias,
            args_type,
            result_type_alias,
            result_type,
            description: tool.description().to_string(),
        });
    }

    Ok(bindings)
}

fn codeact_execute_tool() -> Tool {
    Tool {
        tool_type: "function".to_string(),
        function: FunctionTool {
            name: CODEACT_TOOL_NAME.to_string(),
            description: "Execute TypeScript in a fresh sandbox. Use the provided external_* bindings to call tools. The code must return a JSON-serializable value and must not use imports or host globals.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "TypeScript source code to execute. Return a JSON-serializable value from the code."
                    }
                },
                "required": ["code"],
                "additionalProperties": false
            }),
        },
    }
}

fn parse_execute_args(arguments: &str) -> Result<ExecuteTypescriptArgs, String> {
    serde_json::from_str::<ExecuteTypescriptArgs>(arguments)
        .map_err(|e| format!("invalid execute_typescript arguments: {e}"))
}

fn wrap_typescript_source(source: &str) -> String {
    format!("{CODEACT_WRAPPER_PREFIX}{source}{CODEACT_WRAPPER_SUFFIX}")
}

fn render_typescript_binding_declaration(binding: &CodeActToolBinding) -> String {
    format!(
        "/** {} */\ntype {} = {};\ntype {} = {};\ndeclare function {}(args: {}): Promise<{}>;\n\n",
        escape_typescript_doc_comment(&binding.description),
        binding.args_type_alias,
        binding.args_type,
        binding.result_type_alias,
        binding.result_type,
        binding.js_name,
        binding.args_type_alias,
        binding.result_type_alias,
    )
}

fn render_typescript_binding_implementation(binding: &CodeActToolBinding) -> String {
    format!(
        "/** {} */\ntype {} = {};\ntype {} = {};\nasync function {}(args: {}): Promise<{}> {{\n  const __reply = JSON.parse(await __codeact_invoke({:?}, JSON.stringify(args ?? {{}})));\n  if (!__reply.ok) {{\n    throw new Error(__reply.error ?? \"tool call failed\");\n  }}\n  return __reply.value;\n}}\n\n",
        escape_typescript_doc_comment(&binding.description),
        binding.args_type_alias,
        binding.args_type,
        binding.result_type_alias,
        binding.result_type,
        binding.js_name,
        binding.args_type_alias,
        binding.result_type_alias,
        binding.original_name,
    )
}

fn build_typescript_prelude(tool_bindings: &[CodeActToolBinding]) -> String {
    tool_bindings
        .iter()
        .map(render_typescript_binding_implementation)
        .collect()
}

fn build_typescript_program(source: &str, tool_bindings: &[CodeActToolBinding]) -> String {
    let prelude = build_typescript_prelude(tool_bindings);
    let wrapped_source = wrap_typescript_source(source);
    if prelude.is_empty() {
        wrapped_source
    } else {
        format!("{prelude}{wrapped_source}")
    }
}

fn parse_typescript_program(source: &str) -> Result<ParsedSource, CodeActExecutorError> {
    parse_program(ParseParams {
        specifier: ModuleSpecifier::parse("file:///codeact.ts")
            .map_err(|err| CodeActExecutorError::TranspileError(err.to_string()))?,
        media_type: MediaType::TypeScript,
        text: source.into(),
        capture_tokens: false,
        scope_analysis: false,
        maybe_syntax: None,
    })
    .map_err(|err| CodeActExecutorError::TranspileError(err.to_string()))
}

fn codeact_main_body_statements(parsed: &ParsedSource) -> Option<&[Stmt]> {
    match parsed.program_ref() {
        ProgramRef::Module(module) => module.body.iter().find_map(codeact_main_body_from_item),
        ProgramRef::Script(script) => script.body.iter().find_map(codeact_main_body_from_stmt),
    }
}

fn codeact_main_body_from_item(item: &ModuleItem) -> Option<&[Stmt]> {
    match item {
        ModuleItem::Stmt(stmt) => codeact_main_body_from_stmt(stmt),
        ModuleItem::ModuleDecl(_) => None,
    }
}

fn codeact_main_body_from_stmt(stmt: &Stmt) -> Option<&[Stmt]> {
    let Stmt::Decl(Decl::Var(var_decl)) = stmt else {
        return None;
    };

    var_decl.decls.iter().find_map(|declarator| {
        let Pat::Ident(binding) = &declarator.name else {
            return None;
        };
        if binding.id.sym.as_ref() != "__codeact_main" {
            return None;
        }

        let init = declarator.init.as_deref()?;
        let Expr::Arrow(arrow) = init else {
            return None;
        };
        let BlockStmtOrExpr::BlockStmt(block) = arrow.body.as_ref() else {
            return None;
        };
        Some(block.stmts.as_slice())
    })
}

fn trailing_expression_statement_range(parsed: &ParsedSource) -> Option<SourceRange> {
    let statements = codeact_main_body_statements(parsed)?;
    let last_statement = statements
        .iter()
        .rev()
        .find(|stmt| !matches!(stmt, Stmt::Empty(_)))?;

    match last_statement {
        Stmt::Expr(expr_stmt) => Some(SourceRange::unsafely_from_span(expr_stmt.span)),
        _ => None,
    }
}

fn normalize_wrapped_typescript_source(
    parsed: &ParsedSource,
) -> Result<Option<String>, CodeActExecutorError> {
    let Some(expr_range) = trailing_expression_statement_range(parsed) else {
        return Ok(None);
    };

    let text_info = parsed.text_info_lazy();
    let expression_text = text_info
        .range_text(&expr_range)
        .trim()
        .trim_end_matches(';')
        .trim();
    if expression_text.is_empty() {
        return Ok(None);
    }

    let replacement = format!("return await ({expression_text});");
    let byte_range = expr_range.as_byte_range(text_info.range().start);
    let source = parsed.text();
    let mut normalized = String::with_capacity(source.len() + replacement.len());
    normalized.push_str(&source[..byte_range.start]);
    normalized.push_str(&replacement);
    normalized.push_str(&source[byte_range.end..]);

    Ok(Some(normalized))
}

fn schema_to_typescript(schema: &Value) -> String {
    if let Some(one_of) = schema.get("oneOf").and_then(Value::as_array) {
        let parts: Vec<String> = one_of.iter().map(schema_to_typescript).collect();
        return format_union(parts);
    }
    if let Some(any_of) = schema.get("anyOf").and_then(Value::as_array) {
        let parts: Vec<String> = any_of.iter().map(schema_to_typescript).collect();
        return format_union(parts);
    }
    if let Some(all_of) = schema.get("allOf").and_then(Value::as_array) {
        let parts: Vec<String> = all_of.iter().map(schema_to_typescript).collect();
        return format_intersection(parts);
    }
    if let Some(enum_values) = schema.get("enum").and_then(Value::as_array) {
        let parts = enum_values
            .iter()
            .filter_map(|value| match value {
                Value::String(value) => Some(format!("{value:?}")),
                Value::Number(value) => Some(value.to_string()),
                Value::Bool(value) => Some(value.to_string()),
                _ => None,
            })
            .collect::<Vec<_>>();
        if !parts.is_empty() {
            return format_union(parts);
        }
    }

    match schema.get("type").and_then(Value::as_str) {
        Some("string") => "string".to_string(),
        Some("integer") | Some("number") => "number".to_string(),
        Some("boolean") => "boolean".to_string(),
        Some("null") => "null".to_string(),
        Some("array") => {
            let item_type = schema
                .get("items")
                .map(schema_to_typescript)
                .unwrap_or_else(|| "unknown".to_string());
            format!("Array<{item_type}>")
        }
        Some("object") => object_schema_to_typescript(schema),
        Some(_) | None => {
            if schema.get("properties").is_some() {
                object_schema_to_typescript(schema)
            } else {
                "unknown".to_string()
            }
        }
    }
}

fn object_schema_to_typescript(schema: &Value) -> String {
    let required = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect::<HashSet<_>>()
        })
        .unwrap_or_default();
    let mut fields = Vec::new();

    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        for (name, property_schema) in properties {
            let optional = if required.contains(name) { "" } else { "?" };
            let property_type = schema_to_typescript(property_schema);
            let property_name = format_typescript_property_key(name);
            fields.push(format!("{property_name}{optional}: {property_type};"));
        }
    }

    if fields.is_empty() {
        match schema.get("additionalProperties") {
            Some(Value::Bool(true)) => return "Record<string, unknown>".to_string(),
            Some(Value::Object(_)) => {
                let value_type = schema_to_typescript(&schema["additionalProperties"]);
                return format!("Record<string, {value_type}>");
            }
            _ => {}
        }
        return "{ [key: string]: unknown }".to_string();
    }

    format!("{{ {} }}", fields.join(" "))
}

fn format_union(parts: Vec<String>) -> String {
    let parts: Vec<String> = parts.into_iter().filter(|part| !part.is_empty()).collect();
    if parts.is_empty() {
        "unknown".to_string()
    } else {
        parts.join(" | ")
    }
}

fn format_intersection(parts: Vec<String>) -> String {
    let parts: Vec<String> = parts.into_iter().filter(|part| !part.is_empty()).collect();
    if parts.is_empty() {
        "unknown".to_string()
    } else {
        parts.join(" & ")
    }
}

fn sanitize_identifier(name: &str) -> String {
    let mut output = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            output.push(ch);
        } else {
            output.push('_');
        }
    }

    if output.is_empty() {
        output.push('_');
    }

    if output.chars().next().is_some_and(|ch| ch.is_ascii_digit()) {
        output.insert(0, '_');
    }

    output
}

fn escape_typescript_doc_comment(text: &str) -> String {
    text.replace("*/", "*\\/")
}

fn is_typescript_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_' || first == '$') {
        return false;
    }

    chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '$')
        && !matches!(
            name,
            "break"
                | "case"
                | "catch"
                | "class"
                | "const"
                | "continue"
                | "debugger"
                | "default"
                | "delete"
                | "do"
                | "else"
                | "enum"
                | "export"
                | "extends"
                | "false"
                | "finally"
                | "for"
                | "function"
                | "if"
                | "import"
                | "in"
                | "instanceof"
                | "new"
                | "null"
                | "return"
                | "super"
                | "switch"
                | "this"
                | "throw"
                | "true"
                | "try"
                | "typeof"
                | "var"
                | "void"
                | "while"
                | "with"
                | "yield"
                | "let"
                | "static"
                | "implements"
                | "interface"
                | "package"
                | "private"
                | "protected"
                | "public"
        )
}

fn format_typescript_property_key(name: &str) -> String {
    if is_typescript_identifier(name) {
        name.to_string()
    } else {
        serde_json::to_string(name).unwrap_or_else(|_| "\"\"".to_string())
    }
}

#[derive(Default)]
struct ForbiddenSyntaxVisitor {
    violations: Vec<&'static str>,
}

impl ForbiddenSyntaxVisitor {
    fn record(&mut self, label: &'static str) {
        if !self.violations.contains(&label) {
            self.violations.push(label);
        }
    }
}

impl Visit for ForbiddenSyntaxVisitor {
    fn visit_call_expr(&mut self, call_expr: &CallExpr) {
        if matches!(call_expr.callee, Callee::Import(..)) {
            self.record("dynamic import");
        }

        if let Callee::Expr(callee) = &call_expr.callee
            && let Expr::Ident(ident) = &**callee
            && ident.sym.as_ref() == "require"
        {
            self.record("require()");
        }

        call_expr.visit_children_with(self);
    }

    fn visit_meta_prop_expr(&mut self, meta_prop: &MetaPropExpr) {
        if matches!(meta_prop.kind, MetaPropKind::ImportMeta) {
            self.record("import.meta");
        }
        meta_prop.visit_children_with(self);
    }

    fn visit_member_expr(&mut self, member_expr: &MemberExpr) {
        if let Expr::Ident(object) = &*member_expr.obj
            && object.sym.as_ref() == "require"
            && matches!(member_expr.prop, MemberProp::Ident(..))
        {
            self.record("require.*");
        }

        member_expr.visit_children_with(self);
    }
}

fn validate_typescript_source(
    source: &str,
    tool_bindings: &[CodeActToolBinding],
    limits: &CodeActSandboxLimits,
) -> Result<ParsedSource, CodeActExecutorError> {
    if source.len() > limits.max_source_bytes {
        return Err(CodeActExecutorError::ValidationError(format!(
            "source exceeds max_source_bytes ({})",
            limits.max_source_bytes
        )));
    }

    let parsed = parse_typescript_program(&build_typescript_program(source, tool_bindings))?;
    let mut visitor = ForbiddenSyntaxVisitor::default();
    parsed.program_ref().visit_with(&mut visitor);

    if !visitor.violations.is_empty() {
        return Err(CodeActExecutorError::ValidationError(format!(
            "{} is not allowed in CodeAct TypeScript",
            visitor.violations.join(", ")
        )));
    }

    Ok(parsed)
}

fn transpile_typescript(
    source: &str,
    tool_bindings: &[CodeActToolBinding],
    limits: &CodeActSandboxLimits,
) -> Result<String, CodeActExecutorError> {
    let parsed = validate_typescript_source(source, tool_bindings, limits)?;
    let parsed = if let Some(normalized_source) = normalize_wrapped_typescript_source(&parsed)? {
        parse_typescript_program(&normalized_source)?
    } else {
        parsed
    };

    parsed
        .transpile(
            &TranspileOptions::default(),
            &TranspileModuleOptions::default(),
            &EmitOptions::default(),
        )
        .map(|emitted| emitted.into_source().text)
        .map_err(|err| CodeActExecutorError::TranspileError(err.to_string()))
}

async fn execute_typescript_sandbox<H>(
    hooks: H,
    request: CodeActSandboxRequest,
) -> CodeActExecutionRecord
where
    H: AgentHooks + Clone + Send + Sync + 'static,
{
    let started_at = Instant::now();
    EventHelper::send_code_execution_started(
        &request.tx_event,
        request.submission_id,
        request.context.config().id,
        request.execution_id.clone(),
        CODEACT_LANGUAGE.to_string(),
        request.source.clone(),
    )
    .await;

    let duration_ms = started_at.elapsed().as_millis() as u64;
    let record = run_typescript_sandbox(hooks, request.clone())
        .await
        .into_record(
            request.execution_id.clone(),
            request.source.clone(),
            duration_ms,
        );

    for message in &record.console {
        EventHelper::send_code_execution_console(
            &request.tx_event,
            request.submission_id,
            request.context.config().id,
            request.execution_id.clone(),
            message.clone(),
        )
        .await;
    }

    if record.success {
        EventHelper::send_code_execution_completed(
            &request.tx_event,
            request.submission_id,
            request.context.config().id,
            request.execution_id,
            record.result.clone().unwrap_or(Value::Null),
            duration_ms,
        )
        .await;
    } else {
        EventHelper::send_code_execution_failed(
            &request.tx_event,
            request.submission_id,
            request.context.config().id,
            request.execution_id,
            record
                .error
                .clone()
                .unwrap_or_else(|| "unknown sandbox failure".to_string()),
            duration_ms,
        )
        .await;
    }

    record
}

#[derive(Debug)]
struct SandboxOutcome {
    result: Option<Value>,
    error: Option<String>,
    console: Vec<String>,
    tool_calls: Vec<ToolCallResult>,
}

impl SandboxOutcome {
    fn success(result: Value, console: Vec<String>, tool_calls: Vec<ToolCallResult>) -> Self {
        Self {
            result: Some(result),
            error: None,
            console,
            tool_calls,
        }
    }

    fn failure(
        error: impl Into<String>,
        console: Vec<String>,
        tool_calls: Vec<ToolCallResult>,
    ) -> Self {
        Self {
            result: None,
            error: Some(error.into()),
            console,
            tool_calls,
        }
    }

    fn into_record(
        self,
        execution_id: String,
        source: String,
        duration_ms: u64,
    ) -> CodeActExecutionRecord {
        CodeActExecutionRecord {
            execution_id,
            source,
            console: self.console,
            tool_calls: self.tool_calls,
            result: self.result,
            success: self.error.is_none(),
            error: self.error,
            duration_ms,
        }
    }
}

#[derive(Clone, Debug)]
struct ConsoleCapture {
    lines: Arc<StdMutex<Vec<String>>>,
    bytes: Arc<AtomicUsize>,
    truncated: Arc<AtomicBool>,
    limit: usize,
}

impl ConsoleCapture {
    fn new(limit: usize) -> Self {
        Self {
            lines: Arc::new(StdMutex::new(Vec::new())),
            bytes: Arc::new(AtomicUsize::new(0)),
            truncated: Arc::new(AtomicBool::new(false)),
            limit,
        }
    }

    fn push(&self, level: String, message: String) -> String {
        let formatted = format!("[{level}] {message}");
        let mut accepted = None;
        let current = self.bytes.load(Ordering::Relaxed);
        if current < self.limit {
            let remaining = self.limit - current;
            let mut line = formatted;
            if line.len() > remaining {
                line.truncate(remaining);
            }
            self.bytes.fetch_add(line.len(), Ordering::Relaxed);
            accepted = Some(line);
        } else if !self.truncated.swap(true, Ordering::Relaxed) {
            accepted = Some("[log] console output truncated".to_string());
        }

        if let Some(line) = accepted
            && let Ok(mut guard) = self.lines.lock()
        {
            guard.push(line);
        }

        String::default()
    }

    fn snapshot(&self) -> Vec<String> {
        self.lines
            .lock()
            .map(|guard| guard.to_vec())
            .unwrap_or_default()
    }
}

#[derive(Debug)]
struct CodeActToolInvocationState {
    tool_results: StdMutex<Vec<ToolCallResult>>,
    tool_call_count: AtomicUsize,
    inflight_tool_calls: AtomicUsize,
    next_tool_id: AtomicUsize,
    max_tool_calls_per_execution: usize,
    max_concurrent_tool_calls: usize,
}

impl CodeActToolInvocationState {
    fn new(limits: &CodeActSandboxLimits) -> Self {
        Self {
            tool_results: StdMutex::new(Vec::new()),
            tool_call_count: AtomicUsize::new(0),
            inflight_tool_calls: AtomicUsize::new(0),
            next_tool_id: AtomicUsize::new(0),
            max_tool_calls_per_execution: limits.max_tool_calls_per_execution,
            max_concurrent_tool_calls: limits.max_concurrent_tool_calls,
        }
    }

    fn tool_results(&self) -> Vec<ToolCallResult> {
        self.tool_results
            .lock()
            .map(|guard| guard.to_vec())
            .unwrap_or_default()
    }
}

fn tool_limit_envelope(limit_name: &str, limit_value: usize) -> String {
    json!({
        "ok": false,
        "error": format!("execution exceeded {limit_name} ({limit_value})"),
    })
    .to_string()
}

#[derive(Clone)]
struct CodeActToolInvoker<H> {
    hooks: H,
    context: Arc<Context>,
    submission_id: SubmissionId,
    execution_id: String,
    shared: Arc<CodeActToolInvocationState>,
}

impl<H> CodeActToolInvoker<H>
where
    H: AgentHooks + Clone + Send + Sync + 'static,
{
    fn new(
        hooks: H,
        context: Arc<Context>,
        submission_id: SubmissionId,
        execution_id: String,
        shared: Arc<CodeActToolInvocationState>,
    ) -> Self {
        Self {
            hooks,
            context,
            submission_id,
            execution_id,
            shared,
        }
    }

    async fn invoke(&self, tool_name: String, args_json: String) -> String {
        let concurrent_calls = self
            .shared
            .inflight_tool_calls
            .fetch_add(1, Ordering::SeqCst)
            + 1;
        if concurrent_calls > self.shared.max_concurrent_tool_calls {
            self.shared
                .inflight_tool_calls
                .fetch_sub(1, Ordering::SeqCst);
            return tool_limit_envelope(
                "max_concurrent_tool_calls",
                self.shared.max_concurrent_tool_calls,
            );
        }

        let current_call = self.shared.tool_call_count.fetch_add(1, Ordering::SeqCst) + 1;
        if current_call > self.shared.max_tool_calls_per_execution {
            self.shared
                .inflight_tool_calls
                .fetch_sub(1, Ordering::SeqCst);
            return tool_limit_envelope(
                "max_tool_calls_per_execution",
                self.shared.max_tool_calls_per_execution,
            );
        }

        let tool_call = ToolCall {
            id: format!(
                "{}-tool-{}",
                self.execution_id,
                self.shared.next_tool_id.fetch_add(1, Ordering::SeqCst) + 1
            ),
            call_type: "function".to_string(),
            function: autoagents_llm::FunctionCall {
                name: tool_name,
                arguments: args_json,
            },
        };

        let tool_tx_event = self.context.tx().ok();
        let result = ToolProcessor::process_single_tool_call_with_hooks(
            &self.hooks,
            &self.context,
            self.submission_id,
            self.context.tools(),
            &tool_call,
            &tool_tx_event,
        )
        .await;

        let envelope = match result {
            Some(result) => {
                if let Ok(mut guard) = self.shared.tool_results.lock() {
                    guard.push(result.clone());
                }
                if result.success {
                    json!({"ok": true, "value": result.result})
                } else {
                    json!({
                        "ok": false,
                        "error": result.result
                            .get("error")
                            .and_then(Value::as_str)
                            .unwrap_or("tool execution failed")
                    })
                }
            }
            None => json!({
                "ok": false,
                "error": "tool call aborted by hook"
            }),
        };

        self.shared
            .inflight_tool_calls
            .fetch_sub(1, Ordering::SeqCst);
        envelope.to_string()
    }
}

fn build_sandbox_outcome(
    execution_result: Result<String, String>,
    console: Vec<String>,
    tool_calls: Vec<ToolCallResult>,
) -> SandboxOutcome {
    match execution_result {
        Ok(result_json) => match serde_json::from_str::<Value>(&result_json) {
            Ok(Value::Object(result_value)) => {
                if let Some(error) = result_value.get("error").and_then(Value::as_str) {
                    return SandboxOutcome::failure(error.to_string(), console, tool_calls);
                }

                let result = result_value.get("value").cloned().unwrap_or(Value::Null);
                SandboxOutcome::success(result, console, tool_calls)
            }
            Ok(_) => SandboxOutcome::failure(
                "sandbox returned a non-object payload".to_string(),
                console,
                tool_calls,
            ),
            Err(error) => SandboxOutcome::failure(error.to_string(), console, tool_calls),
        },
        Err(error) => SandboxOutcome::failure(error, console, tool_calls),
    }
}

#[cfg(not(target_arch = "wasm32"))]
async fn run_typescript_sandbox<H>(hooks: H, request: CodeActSandboxRequest) -> SandboxOutcome
where
    H: AgentHooks + Clone + Send + Sync + 'static,
{
    let transpiled =
        match transpile_typescript(&request.source, &request.tool_bindings, &request.limits) {
            Ok(transpiled) => transpiled,
            Err(error) => {
                return SandboxOutcome::failure(error.to_string(), Vec::new(), Vec::new());
            }
        };
    let script = build_runtime_script(&transpiled);

    let console_capture = ConsoleCapture::new(request.limits.max_console_bytes);
    let invocation_state = Arc::new(CodeActToolInvocationState::new(&request.limits));
    let tool_invoker = CodeActToolInvoker::new(
        hooks,
        Arc::clone(&request.context),
        request.submission_id,
        request.execution_id,
        Arc::clone(&invocation_state),
    );
    let handle = tokio::runtime::Handle::current();
    let runtime_console_capture = console_capture.clone();
    let execution_result = tokio::task::spawn_blocking(move || {
        let runtime = JsRuntime::new().map_err(|err| err.to_string())?;
        runtime.set_memory_limit(request.limits.memory_limit_bytes);
        runtime.set_max_stack_size(1024 * 1024);
        runtime.set_interrupt_handler(Some(Box::new({
            let deadline = Instant::now() + Duration::from_millis(request.limits.timeout_ms);
            move || Instant::now() >= deadline
        })));

        let js_context = JsContext::full(&runtime).map_err(|err| err.to_string())?;

        js_context.with(|ctx| -> Result<String, String> {
            let globals = ctx.globals();

            let console_capture = runtime_console_capture.clone();
            globals
                .set(
                    "__codeact_console",
                    Func::from(move |level: String, message: String| -> String {
                        console_capture.push(level, message)
                    }),
                )
                .map_err(|err| err.to_string())?;

            let tool_invoker = tool_invoker.clone();
            let runtime_handle = handle.clone();
            globals
                .set(
                    "__codeact_invoke",
                    Func::from(move |tool_name: String, args_json: String| -> String {
                        let tool_invoker = tool_invoker.clone();
                        runtime_handle
                            .block_on(async { tool_invoker.invoke(tool_name, args_json).await })
                    }),
                )
                .map_err(|err| err.to_string())?;

            let promise = ctx
                .eval::<Promise, _>(script.as_str())
                .catch(&ctx)
                .map_err(|err| err.to_string())?;
            promise
                .finish::<String>()
                .catch(&ctx)
                .map_err(|err| err.to_string())
        })
    })
    .await
    .map_err(|err| format!("sandbox execution join failed: {err}"))
    .and_then(|result| result);

    build_sandbox_outcome(
        execution_result,
        console_capture.snapshot(),
        invocation_state.tool_results(),
    )
}

#[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
async fn run_typescript_sandbox<H>(hooks: H, request: CodeActSandboxRequest) -> SandboxOutcome
where
    H: AgentHooks + Clone + Send + Sync + 'static,
{
    let transpiled =
        match transpile_typescript(&request.source, &request.tool_bindings, &request.limits) {
            Ok(transpiled) => transpiled,
            Err(error) => {
                return SandboxOutcome::failure(error.to_string(), Vec::new(), Vec::new());
            }
        };
    let script = build_runtime_script(&transpiled);

    let console_capture = ConsoleCapture::new(request.limits.max_console_bytes);
    let invocation_state = Arc::new(CodeActToolInvocationState::new(&request.limits));
    let tool_invoker = CodeActToolInvoker::new(
        hooks,
        Arc::clone(&request.context),
        request.submission_id,
        request.execution_id,
        Arc::clone(&invocation_state),
    );

    let runtime = match JsRuntime::new() {
        Ok(runtime) => runtime,
        Err(error) => {
            return SandboxOutcome::failure(error.to_string(), Vec::new(), Vec::new());
        }
    };
    runtime.set_memory_limit(request.limits.memory_limit_bytes);
    runtime.set_max_stack_size(1024 * 1024);
    runtime.set_interrupt_handler(Some(Box::new({
        let deadline = Instant::now() + Duration::from_millis(request.limits.timeout_ms);
        move || Instant::now() >= deadline
    })));

    let js_context = match JsContext::full(&runtime) {
        Ok(js_context) => js_context,
        Err(error) => return SandboxOutcome::failure(error.to_string(), Vec::new(), Vec::new()),
    };

    let execution_result = js_context.with(|ctx| -> Result<String, String> {
        let globals = ctx.globals();

        let console_capture = console_capture.clone();
        globals
            .set(
                "__codeact_console",
                Func::from(move |level: String, message: String| -> String {
                    console_capture.push(level, message)
                }),
            )
            .map_err(|err| err.to_string())?;

        let tool_invoker = tool_invoker.clone();
        globals
            .set(
                "__codeact_invoke",
                Func::from(move |tool_name: String, args_json: String| -> String {
                    let tool_invoker = tool_invoker.clone();
                    futures::executor::block_on(async move {
                        tool_invoker.invoke(tool_name, args_json).await
                    })
                }),
            )
            .map_err(|err| err.to_string())?;

        let promise = ctx
            .eval::<Promise, _>(script.as_str())
            .catch(&ctx)
            .map_err(|err| err.to_string())?;
        promise
            .finish::<String>()
            .catch(&ctx)
            .map_err(|err| err.to_string())
    });

    build_sandbox_outcome(
        execution_result,
        console_capture.snapshot(),
        invocation_state.tool_results(),
    )
}

#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
async fn run_typescript_sandbox<H>(_hooks: H, _request: CodeActSandboxRequest) -> SandboxOutcome
where
    H: AgentHooks + Clone + Send + Sync + 'static,
{
    SandboxOutcome::failure(
        "CodeAct is supported on native and WASI wasm targets; browser wasm is not supported"
            .to_string(),
        Vec::new(),
        Vec::new(),
    )
}

fn build_runtime_script(transpiled: &str) -> String {
    format!(
        r#"
globalThis.eval = undefined;
globalThis.Function = undefined;
globalThis.require = undefined;
globalThis.process = undefined;
globalThis.fetch = undefined;
globalThis.Deno = undefined;

const __codeact_format = (value) => {{
  if (typeof value === "string") return value;
  try {{
    return JSON.stringify(value, (_, item) => typeof item === "bigint" ? item.toString() : item);
  }} catch (_error) {{
    return String(value);
  }}
}};

globalThis.console = {{
  log: (...args) => __codeact_console("log", args.map(__codeact_format).join(" ")),
  error: (...args) => __codeact_console("error", args.map(__codeact_format).join(" ")),
  warn: (...args) => __codeact_console("warn", args.map(__codeact_format).join(" ")),
  info: (...args) => __codeact_console("info", args.map(__codeact_format).join(" ")),
}};

{transpiled}

(async () => {{
  try {{
    const __codeact_value = await __codeact_main();
    return JSON.stringify(
      {{ value: __codeact_value === undefined ? null : __codeact_value }},
      (_, item) => typeof item === "bigint" ? item.toString() : item
    );
  }} catch (error) {{
    return JSON.stringify({{
      error: String(error?.stack ?? error),
    }});
  }}
}})()
"#
    )
}

fn failed_execution_record(
    execution_id: String,
    source: String,
    error: String,
    console: Vec<String>,
    tool_calls: Vec<ToolCallResult>,
    duration_ms: u64,
) -> CodeActExecutionRecord {
    CodeActExecutionRecord {
        execution_id,
        source,
        console,
        tool_calls,
        result: None,
        success: false,
        error: Some(error),
        duration_ms,
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

fn should_store_user(memory: &MemoryAdapter, stored_user: bool) -> bool {
    if !memory.is_enabled() {
        return false;
    }
    if !memory.policy().store_user {
        return false;
    }
    !stored_user
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

fn record_nested_tool_calls_state(context: &Context, tool_results: &[ToolCallResult]) {
    if tool_results.is_empty() {
        return;
    }

    let state = context.state();
    #[cfg(not(target_arch = "wasm32"))]
    let mut guard = match state.try_lock() {
        Ok(guard) => guard,
        Err(_) => return,
    };

    #[cfg(target_arch = "wasm32")]
    let mut guard = match state.try_lock() {
        Some(guard) => guard,
        None => return,
    };

    for result in tool_results {
        guard.record_tool_call(result.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AgentConfig, Context};
    use crate::tests::{MockAgentImpl, MockLLMProvider};
    use async_trait::async_trait;
    use autoagents_protocol::ActorID;
    use std::sync::Arc;

    #[derive(Debug)]
    struct LocalTool {
        name: String,
        output: serde_json::Value,
        output_schema: Option<serde_json::Value>,
    }

    impl LocalTool {
        fn new(name: &str, output: serde_json::Value) -> Self {
            Self {
                name: name.to_string(),
                output,
                output_schema: None,
            }
        }

        fn with_output_schema(
            name: &str,
            output: serde_json::Value,
            output_schema: serde_json::Value,
        ) -> Self {
            Self {
                name: name.to_string(),
                output,
                output_schema: Some(output_schema),
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

        fn output_schema(&self) -> Option<serde_json::Value> {
            self.output_schema.clone()
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

    #[cfg(not(target_arch = "wasm32"))]
    fn test_context_with_tools(tools: Vec<Box<dyn crate::tool::ToolT>>) -> Arc<Context> {
        let llm = Arc::new(MockLLMProvider {});
        let config = AgentConfig {
            id: ActorID::new_v4(),
            name: "codeact_test".to_string(),
            description: "desc".to_string(),
            output_schema: None,
        };
        Arc::new(
            Context::new(llm, None)
                .with_config(config)
                .with_tools(tools),
        )
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn sandbox_request(
        context: Arc<Context>,
        execution_id: &str,
        source: &str,
    ) -> CodeActSandboxRequest {
        let bindings = build_tool_bindings(context.tools()).unwrap();
        CodeActSandboxRequest {
            context,
            submission_id: SubmissionId::new_v4(),
            tx_event: None,
            execution_id: execution_id.to_string(),
            source: source.to_string(),
            limits: CodeActSandboxLimits::default(),
            tool_bindings: bindings,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn run_async_test<F>(future: F)
    where
        F: std::future::Future<Output = ()>,
    {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("codeact test runtime should build")
            .block_on(future);
    }

    #[test]
    fn test_sanitize_identifier_replaces_invalid_chars() {
        assert_eq!(sanitize_identifier("read-file"), "read_file");
        assert_eq!(sanitize_identifier("123tool"), "_123tool");
    }

    #[test]
    fn test_schema_to_typescript_object() {
        let schema = json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "recursive": { "type": "boolean" }
            },
            "required": ["path"]
        });

        let output = schema_to_typescript(&schema);
        assert!(output.contains("path: string;"));
        assert!(output.contains("recursive?: boolean;"));
    }

    #[test]
    fn test_schema_to_typescript_quotes_invalid_keys() {
        let schema = json!({
            "type": "object",
            "properties": {
                "content-type": { "type": "string" },
                "default": { "type": "boolean" }
            },
            "required": ["content-type"]
        });

        let output = schema_to_typescript(&schema);
        assert!(output.contains("\"content-type\": string;"));
        assert!(output.contains("\"default\"?: boolean;"));
    }

    #[test]
    fn test_build_codeact_system_prompt_escapes_doc_comments() {
        let bindings = vec![CodeActToolBinding {
            original_name: "dangerous".to_string(),
            js_name: "external_dangerous".to_string(),
            args_type_alias: "DangerousArgs".to_string(),
            args_type: "{ value: string; }".to_string(),
            result_type_alias: "DangerousResult".to_string(),
            result_type: "number".to_string(),
            description: "comment */ breaker".to_string(),
        }];

        let prompt = build_codeact_system_prompt("base", &bindings);
        assert!(prompt.contains("/** comment *\\/ breaker */"));
        assert!(prompt.contains("Promise<DangerousResult>"));
    }

    #[test]
    fn test_build_typescript_prelude_generates_typed_tool_helpers() {
        let bindings = vec![CodeActToolBinding {
            original_name: "dangerous".to_string(),
            js_name: "external_dangerous".to_string(),
            args_type_alias: "DangerousArgs".to_string(),
            args_type: "{ value: string; }".to_string(),
            result_type_alias: "DangerousResult".to_string(),
            result_type: "number".to_string(),
            description: "comment */ breaker".to_string(),
        }];

        let prelude = build_typescript_prelude(&bindings);
        assert!(prelude.contains("/** comment *\\/ breaker */"));
        assert!(prelude.contains(
            "async function external_dangerous(args: DangerousArgs): Promise<DangerousResult>"
        ));
        assert!(
            prelude.contains(r#"await __codeact_invoke("dangerous", JSON.stringify(args ?? {}))"#)
        );
    }

    #[test]
    fn test_validate_typescript_source_rejects_dynamic_imports() {
        let limits = CodeActSandboxLimits::default();
        let error = validate_typescript_source("await import('y');", &[], &limits).unwrap_err();
        assert!(matches!(error, CodeActExecutorError::ValidationError(_)));
    }

    #[test]
    fn test_validate_typescript_source_rejects_oversized_input() {
        let limits = CodeActSandboxLimits {
            max_source_bytes: 4,
            ..CodeActSandboxLimits::default()
        };
        let error = validate_typescript_source("hello", &[], &limits).unwrap_err();
        assert!(matches!(error, CodeActExecutorError::ValidationError(_)));
    }

    #[test]
    fn test_validate_typescript_source_allows_harmless_identifiers() {
        let limits = CodeActSandboxLimits::default();
        let parsed = validate_typescript_source(
            r#"
const processValue = "ok";
console.log(processValue);
return processValue;
"#,
            &[],
            &limits,
        )
        .expect("identifier-only usage should pass validation");

        assert!(parsed.text().contains("processValue"));
    }

    #[test]
    fn test_transpile_typescript_includes_generated_tool_helpers() {
        let limits = CodeActSandboxLimits::default();
        let bindings = vec![CodeActToolBinding {
            original_name: "tool_a".to_string(),
            js_name: "external_tool_a".to_string(),
            args_type_alias: "tool_aArgs".to_string(),
            args_type: "{ path: string; }".to_string(),
            result_type_alias: "tool_aResult".to_string(),
            result_type: "{ ok: boolean; }".to_string(),
            description: "read a path".to_string(),
        }];

        let transpiled = transpile_typescript(
            "return await external_tool_a({ path: 'x' });",
            &bindings,
            &limits,
        )
        .expect("transpile should include generated helpers");

        assert!(transpiled.contains("async function external_tool_a(args)"));
        assert!(
            transpiled.contains(r#"await __codeact_invoke("tool_a", JSON.stringify(args ?? {}))"#)
        );
    }

    #[test]
    fn test_build_runtime_script_keeps_runtime_shell_static() {
        let script = build_runtime_script("const __codeact_main = async () => 1;");

        assert!(script.contains("globalThis.console"));
        assert!(script.contains("const __codeact_main = async () => 1;"));
        assert!(!script.contains("__codeact_invoke(\"tool_a\""));
        assert!(!script.contains("async function external_tool_a"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_normalize_wrapped_typescript_source_promotes_trailing_expression() {
        let parsed = parse_typescript_program(&wrap_typescript_source(
            r#"
async function compute() {
  return 42;
}

compute();
"#,
        ))
        .unwrap();

        let normalized = normalize_wrapped_typescript_source(&parsed)
            .unwrap()
            .expect("expected trailing expression rewrite");

        assert!(normalized.contains("return await (compute());"));
    }

    #[test]
    fn test_codeact_output_try_parse() {
        let output = CodeActAgentOutput {
            response: r#"{"value":1}"#.to_string(),
            executions: Vec::new(),
            done: true,
        };

        let parsed: serde_json::Value = output.try_parse().unwrap();
        assert_eq!(parsed["value"], 1);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_run_typescript_sandbox_returns_serialized_result() {
        run_async_test(async {
            let context = test_context_with_tools(vec![Box::new(LocalTool::new(
                "noop",
                json!({"ok": true}),
            ))]);
            let outcome = run_typescript_sandbox(
                MockAgentImpl::new("codeact", "desc"),
                sandbox_request(context, "exec_1", "return { value: 42, ok: true };"),
            )
            .await;

            assert_eq!(outcome.result, Some(json!({ "value": 42, "ok": true })));
            assert!(outcome.error.is_none());
            assert!(outcome.console.is_empty());
            assert!(outcome.tool_calls.is_empty());
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_run_typescript_sandbox_executes_bound_tools() {
        run_async_test(async {
            let context = test_context_with_tools(vec![
                Box::new(LocalTool::new("tool_a", json!({"value": 1}))),
                Box::new(LocalTool::new("tool_b", json!({"value": 2}))),
            ]);
            let outcome = run_typescript_sandbox(
                MockAgentImpl::new("codeact", "desc"),
                sandbox_request(
                    context,
                    "exec_2",
                    r#"
const [left, right] = await Promise.all([
  external_tool_a({}),
  external_tool_b({}),
]);
console.log("sum", left.value + right.value);
return { total: left.value + right.value };
"#,
                ),
            )
            .await;

            assert_eq!(outcome.result, Some(json!({ "total": 3 })));
            assert!(outcome.error.is_none());
            assert_eq!(outcome.tool_calls.len(), 2);
            assert_eq!(outcome.console, vec!["[log] sum 3".to_string()]);
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_run_typescript_sandbox_promotes_trailing_expression_result() {
        run_async_test(async {
            let context = test_context_with_tools(vec![
                Box::new(LocalTool::new("AddNumbers", json!(42))),
                Box::new(LocalTool::new("MultiplyNumbers", json!(126))),
            ]);
            let outcome = run_typescript_sandbox(
                MockAgentImpl::new("codeact", "desc"),
                sandbox_request(
                    context,
                    "exec_3",
                    r#"
async function compute() {
  const intermediate = await external_AddNumbers({});
  console.log("Intermediate Sum:", intermediate);
  const total = await external_MultiplyNumbers({});
  return {
    intermediate_sum: intermediate,
    final_result: total,
  };
}

compute();
"#,
                ),
            )
            .await;

            assert_eq!(
                outcome.result,
                Some(json!({
                    "intermediate_sum": 42,
                    "final_result": 126,
                }))
            );
            assert!(outcome.error.is_none());
            assert_eq!(outcome.tool_calls.len(), 2);
            assert_eq!(
                outcome.console,
                vec!["[log] Intermediate Sum: 42".to_string()]
            );
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_run_typescript_sandbox_preserves_state_on_failure() {
        run_async_test(async {
            let context = test_context_with_tools(vec![Box::new(LocalTool::new(
                "tool_a",
                json!({"value": 7}),
            ))]);

            let outcome = run_typescript_sandbox(
                MockAgentImpl::new("codeact", "desc"),
                sandbox_request(
                    context,
                    "exec_4",
                    r#"
const value = await external_tool_a({});
console.log("tool value", value.value);
throw new Error("boom");
"#,
                ),
            )
            .await;

            assert_eq!(outcome.result, None);
            assert!(outcome.error.is_some());
            assert_eq!(outcome.console, vec!["[log] tool value 7".to_string()]);
            assert_eq!(outcome.tool_calls.len(), 1);
            assert_eq!(outcome.tool_calls[0].tool_name, "tool_a");
        });
    }

    #[test]
    fn test_build_tool_bindings_use_output_schema_when_available() {
        let tool = LocalTool::with_output_schema(
            "typed_tool",
            json!({"value": 1}),
            json!({
                "type": "object",
                "properties": {
                    "value": { "type": "integer" }
                },
                "required": ["value"]
            }),
        );
        let tools: Vec<Box<dyn crate::tool::ToolT>> = vec![Box::new(tool)];

        let bindings = build_tool_bindings(&tools).expect("bindings should build");
        assert_eq!(bindings[0].result_type_alias, "typed_toolResult");
        assert_eq!(bindings[0].result_type, "{ value: number; }");
    }
}
