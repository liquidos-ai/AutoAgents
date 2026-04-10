use crate::convert::{json_value_to_py, py_any_to_json_value};
use crate::tool::PyTool;
use async_trait::async_trait;
use autoagents_core::agent::error::RunnableAgentError;
use autoagents_core::agent::memory::MemoryProvider;
use autoagents_core::agent::prebuilt::executor::{
    BasicAgentOutput, CodeActAgentOutput, CodeActExecutionRecord, ReActAgentOutput,
};
use autoagents_core::agent::task::Task;
use autoagents_core::agent::{
    AgentDeriveT, AgentExecutor, AgentHooks, AgentOutputT, BaseAgent, Context, DirectAgent,
    HookOutcome,
};
use autoagents_core::runtime::Runtime;
use autoagents_core::tool::{ToolT, shared_tools_to_boxes, to_llm_tool};
use autoagents_core::utils::BoxEventStream;
use autoagents_llm::LLMProvider;
use autoagents_llm::ToolCall;
use autoagents_llm::chat::{
    ChatMessage, ChatResponse, ChatRole, MessageType, StructuredOutputFormat, Tool,
};
use autoagents_llm::error::LLMError;
use autoagents_protocol::{Event, ToolCallResult};
use futures::Stream;
use futures::StreamExt;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_async_runtimes::TaskLocals;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

/// Output type for `PyAgentDef`.
///
/// We cannot add a foreign-trait impl (`AgentOutputT`) to the foreign type
/// `ReActAgentOutput`, so we wrap it here and forward `From`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyAgentOutput {
    pub response: String,
    pub tool_calls: Vec<ToolCallResult>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub executions: Vec<Value>,
    pub done: bool,
}

#[derive(Clone, Default)]
pub(crate) struct HookErrorState(Arc<Mutex<Option<String>>>);

impl HookErrorState {
    pub(crate) fn clear(&self) {
        let mut guard = self
            .0
            .lock()
            .expect("hook error state mutex poisoned while clearing");
        *guard = None;
    }

    pub(crate) fn record(&self, message: impl Into<String>) {
        let mut guard = self
            .0
            .lock()
            .expect("hook error state mutex poisoned while recording");
        if guard.is_none() {
            *guard = Some(message.into());
        }
    }

    pub(crate) fn take(&self) -> Option<String> {
        self.0
            .lock()
            .expect("hook error state mutex poisoned while taking")
            .take()
    }
}

fn normalize_hook_error(err: &str) -> String {
    let trimmed = err.trim();
    if let Some((prefix, rest)) = trimmed.split_once(": ")
        && prefix
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '.')
        && (prefix.ends_with("Error") || prefix.ends_with("Exception"))
    {
        return rest.to_string();
    }
    trimmed.to_string()
}

impl AgentOutputT for PyAgentOutput {
    fn output_schema() -> &'static str {
        "{}"
    }
    fn structured_output_format() -> Value {
        Value::Null
    }
}

impl From<PyAgentOutput> for Value {
    fn from(output: PyAgentOutput) -> Self {
        serde_json::to_value(output).unwrap_or(Value::Null)
    }
}

fn collect_codeact_tool_calls(executions: &[CodeActExecutionRecord]) -> Vec<ToolCallResult> {
    executions
        .iter()
        .flat_map(|execution| execution.tool_calls.iter().cloned())
        .collect()
}

impl From<ReActAgentOutput> for PyAgentOutput {
    fn from(r: ReActAgentOutput) -> Self {
        PyAgentOutput {
            response: r.response,
            tool_calls: r.tool_calls,
            executions: Vec::new(),
            done: r.done,
        }
    }
}

impl From<CodeActAgentOutput> for PyAgentOutput {
    fn from(r: CodeActAgentOutput) -> Self {
        let tool_calls = collect_codeact_tool_calls(&r.executions);
        PyAgentOutput {
            response: r.response,
            tool_calls,
            executions: r
                .executions
                .into_iter()
                .map(|execution| serde_json::to_value(execution).unwrap_or(Value::Null))
                .collect(),
            done: r.done,
        }
    }
}

impl From<BasicAgentOutput> for PyAgentOutput {
    fn from(r: BasicAgentOutput) -> Self {
        PyAgentOutput {
            response: r.response,
            tool_calls: Vec::new(),
            executions: Vec::new(),
            done: r.done,
        }
    }
}

/// Core agent definition: satisfies `AgentDeriveT + AgentHooks`.
/// Wrapped in `ReActAgent<PyAgentDef>` which provides `AgentExecutor`.
pub struct PyAgentDef {
    pub name: String,
    pub description: String,
    pub tools: Vec<Arc<dyn ToolT>>,
    pub output_schema: Option<Value>,
    pub hooks: Option<Py<PyAny>>,
    pub task_locals: Option<TaskLocals>,
    pub hook_errors: HookErrorState,
}

impl Clone for PyAgentDef {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            name: self.name.clone(),
            description: self.description.clone(),
            tools: self.tools.clone(),
            output_schema: self.output_schema.clone(),
            hooks: self.hooks.as_ref().map(|h| h.clone_ref(py)),
            task_locals: self.task_locals.clone(),
            hook_errors: self.hook_errors.clone(),
        })
    }
}

impl fmt::Debug for PyAgentDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PyAgentDef")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish()
    }
}

#[async_trait]
impl AgentDeriveT for PyAgentDef {
    type Output = PyAgentOutput;

    fn description(&self) -> &str {
        &self.description
    }
    fn output_schema(&self) -> Option<Value> {
        self.output_schema.clone()
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn tools(&self) -> Vec<Box<dyn ToolT>> {
        shared_tools_to_boxes(&self.tools)
    }
}

type LlmTextStream = Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>;
type LlmJsonStream = Pin<Box<dyn Stream<Item = Result<Value, LLMError>> + Send>>;

#[pyclass(module = "autoagents_py", name = "ExecutionLLM", skip_from_py_object)]
#[derive(Clone)]
pub struct PyExecutionLLM {
    pub inner: Arc<dyn LLMProvider>,
}

impl PyExecutionLLM {
    pub fn new(inner: Arc<dyn LLMProvider>) -> Self {
        Self { inner }
    }

    async fn chat_value(
        &self,
        messages: Vec<ChatMessage>,
        schema: Option<StructuredOutputFormat>,
    ) -> PyResult<Value> {
        let response = self
            .inner
            .chat(&messages, schema)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("llm.chat failed: {e}")))?;
        Ok(chat_response_to_value(response.as_ref()))
    }

    async fn chat_with_tools_value(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        schema: Option<StructuredOutputFormat>,
    ) -> PyResult<Value> {
        let response = self
            .inner
            .chat_with_tools(&messages, tools.as_deref(), schema)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("llm.chat_with_tools failed: {e}")))?;
        Ok(chat_response_to_value(response.as_ref()))
    }

    async fn chat_with_web_search_value(&self, input: String) -> PyResult<Value> {
        let response = self.inner.chat_with_web_search(input).await.map_err(|e| {
            PyRuntimeError::new_err(format!("llm.chat_with_web_search failed: {e}"))
        })?;
        Ok(chat_response_to_value(response.as_ref()))
    }

    async fn build_string_stream(
        &self,
        messages: Vec<ChatMessage>,
        schema: Option<StructuredOutputFormat>,
    ) -> PyResult<PyExecutionStringStream> {
        let stream = self
            .inner
            .chat_stream(&messages, schema)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("llm.chat_stream failed: {e}")))?;

        Ok(PyExecutionStringStream {
            stream: Arc::new(tokio::sync::Mutex::new(stream)),
        })
    }

    async fn build_struct_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        schema: Option<StructuredOutputFormat>,
        label: &'static str,
        error_label: &'static str,
    ) -> PyResult<PyExecutionJsonStream> {
        let stream = self
            .inner
            .chat_stream_struct(&messages, tools.as_deref(), schema)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("{error_label} failed: {e}")))?;
        let stream = map_json_stream(stream, label)?;

        Ok(PyExecutionJsonStream {
            stream: Arc::new(tokio::sync::Mutex::new(stream)),
        })
    }

    async fn build_tool_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        schema: Option<StructuredOutputFormat>,
        label: &'static str,
        error_label: &'static str,
    ) -> PyResult<PyExecutionJsonStream> {
        let stream = self
            .inner
            .chat_stream_with_tools(&messages, tools.as_deref(), schema)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("{error_label} failed: {e}")))?;
        let stream = map_json_stream(stream, label)?;

        Ok(PyExecutionJsonStream {
            stream: Arc::new(tokio::sync::Mutex::new(stream)),
        })
    }
}

#[pymethods]
impl PyExecutionLLM {
    fn __repr__(&self) -> &str {
        "ExecutionLLM(<context>)"
    }

    /// Call the underlying Rust LLM provider from Python.
    ///
    /// `messages` accepts either:
    /// - a single string (treated as a user message),
    /// - a single dict {"role": "...", "content": "..."},
    /// - a list of message dicts.
    #[pyo3(signature = (messages, schema=None))]
    pub fn chat<'py>(
        &self,
        py: Python<'py>,
        messages: &Bound<'_, PyAny>,
        schema: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let messages = parse_chat_messages_from_any(messages)?;
        let schema = parse_structured_schema(schema)?;
        let llm = self.clone();

        crate::async_bridge::future_into_py(py, async move {
            let value = llm.chat_value(messages, schema).await?;
            Python::attach(|py| json_value_to_py(py, &value))
        })
    }

    #[pyo3(signature = (messages, schema))]
    pub fn chat_with_struct<'py>(
        &self,
        py: Python<'py>,
        messages: &Bound<'_, PyAny>,
        schema: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.chat(py, messages, Some(schema))
    }

    #[pyo3(signature = (messages, tools, schema=None))]
    pub fn chat_with_tools<'py>(
        &self,
        py: Python<'py>,
        messages: &Bound<'_, PyAny>,
        tools: &Bound<'_, PyAny>,
        schema: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let messages = parse_chat_messages_from_any(messages)?;
        let tools = parse_tools_from_any(Some(tools))?;
        let schema = parse_structured_schema(schema)?;
        let llm = self.clone();

        crate::async_bridge::future_into_py(py, async move {
            let value = llm.chat_with_tools_value(messages, tools, schema).await?;
            Python::attach(|py| json_value_to_py(py, &value))
        })
    }

    #[pyo3(signature = (messages, tools, schema))]
    pub fn chat_with_tools_struct<'py>(
        &self,
        py: Python<'py>,
        messages: &Bound<'_, PyAny>,
        tools: &Bound<'_, PyAny>,
        schema: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.chat_with_tools(py, messages, tools, Some(schema))
    }

    pub fn chat_with_web_search<'py>(
        &self,
        py: Python<'py>,
        input: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let llm = self.clone();

        crate::async_bridge::future_into_py(py, async move {
            let value = llm.chat_with_web_search_value(input).await?;
            Python::attach(|py| json_value_to_py(py, &value))
        })
    }

    #[pyo3(signature = (messages, schema=None))]
    pub fn chat_stream<'py>(
        &self,
        py: Python<'py>,
        messages: &Bound<'_, PyAny>,
        schema: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let messages = parse_chat_messages_from_any(messages)?;
        let schema = parse_structured_schema(schema)?;
        let llm = self.clone();

        crate::async_bridge::future_into_py(py, async move {
            let stream = llm.build_string_stream(messages, schema).await?;
            Python::attach(|py| {
                stream
                    .into_pyobject(py)
                    .map(|bound| bound.into_any().unbind())
            })
        })
    }

    #[pyo3(signature = (messages, tools=None, schema=None))]
    pub fn chat_stream_struct<'py>(
        &self,
        py: Python<'py>,
        messages: &Bound<'_, PyAny>,
        tools: Option<&Bound<'_, PyAny>>,
        schema: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let messages = parse_chat_messages_from_any(messages)?;
        let tools = parse_tools_from_any(tools)?;
        let schema = parse_structured_schema(schema)?;
        let llm = self.clone();

        crate::async_bridge::future_into_py(py, async move {
            let stream = llm
                .build_struct_stream(
                    messages,
                    tools,
                    schema,
                    "llm.chat_stream_struct",
                    "llm.chat_stream_struct",
                )
                .await?;
            Python::attach(|py| {
                stream
                    .into_pyobject(py)
                    .map(|bound| bound.into_any().unbind())
            })
        })
    }

    #[pyo3(signature = (messages, tools, schema=None))]
    pub fn chat_stream_with_tools<'py>(
        &self,
        py: Python<'py>,
        messages: &Bound<'_, PyAny>,
        tools: &Bound<'_, PyAny>,
        schema: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let messages = parse_chat_messages_from_any(messages)?;
        let tools = parse_tools_from_any(Some(tools))?;
        let schema = parse_structured_schema(schema)?;
        let llm = self.clone();

        crate::async_bridge::future_into_py(py, async move {
            let stream = llm
                .build_tool_stream(
                    messages,
                    tools,
                    schema,
                    "llm.chat_stream_with_tools",
                    "llm.chat_stream_with_tools",
                )
                .await?;
            Python::attach(|py| {
                stream
                    .into_pyobject(py)
                    .map(|bound| bound.into_any().unbind())
            })
        })
    }
}

#[pyclass(
    module = "autoagents_py",
    name = "ExecutionStringStream",
    skip_from_py_object
)]
pub struct PyExecutionStringStream {
    stream: Arc<tokio::sync::Mutex<LlmTextStream>>,
}

impl PyExecutionStringStream {
    async fn next_chunk(&self) -> PyResult<Option<String>> {
        let next = {
            let mut guard = self.stream.lock().await;
            guard.next().await
        };

        match next {
            Some(Ok(chunk)) => Ok(Some(chunk)),
            Some(Err(err)) => Err(PyRuntimeError::new_err(err.to_string())),
            None => Ok(None),
        }
    }
}

#[pymethods]
impl PyExecutionStringStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = PyExecutionStringStream {
            stream: Arc::clone(&self.stream),
        };

        crate::async_bridge::future_into_py(py, async move {
            match stream.next_chunk().await? {
                Some(chunk) => Ok(chunk),
                None => Err(pyo3::exceptions::PyStopAsyncIteration::new_err(
                    "stream ended",
                )),
            }
        })
    }
}

#[pyclass(
    module = "autoagents_py",
    name = "ExecutionJsonStream",
    skip_from_py_object
)]
pub struct PyExecutionJsonStream {
    stream: Arc<tokio::sync::Mutex<LlmJsonStream>>,
}

impl PyExecutionJsonStream {
    async fn next_chunk(&self) -> PyResult<Option<Value>> {
        let next = {
            let mut guard = self.stream.lock().await;
            guard.next().await
        };

        match next {
            Some(Ok(chunk)) => Ok(Some(chunk)),
            Some(Err(err)) => Err(PyRuntimeError::new_err(err.to_string())),
            None => Ok(None),
        }
    }
}

#[pymethods]
impl PyExecutionJsonStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = PyExecutionJsonStream {
            stream: Arc::clone(&self.stream),
        };

        crate::async_bridge::future_into_py(py, async move {
            match stream.next_chunk().await? {
                Some(chunk) => Python::attach(|py| json_value_to_py(py, &chunk)),
                None => Err(pyo3::exceptions::PyStopAsyncIteration::new_err(
                    "stream ended",
                )),
            }
        })
    }
}

#[pyclass(
    module = "autoagents_py",
    name = "ExecutionMemory",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyExecutionMemory {
    pub inner: Option<Arc<tokio::sync::Mutex<Box<dyn MemoryProvider>>>>,
}

impl PyExecutionMemory {
    pub fn new(inner: Option<Arc<tokio::sync::Mutex<Box<dyn MemoryProvider>>>>) -> Self {
        Self { inner }
    }

    fn configured_memory(&self) -> PyResult<Arc<tokio::sync::Mutex<Box<dyn MemoryProvider>>>> {
        self.inner
            .clone()
            .ok_or_else(|| PyRuntimeError::new_err("memory is not configured"))
    }

    async fn recall_messages(&self, query: String, limit: Option<usize>) -> PyResult<Value> {
        let memory = self.configured_memory()?;
        let guard = memory.lock().await;
        let messages = guard
            .recall(&query, limit)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("memory.recall failed: {e}")))?;
        serde_json::to_value(messages)
            .map_err(|e| PyRuntimeError::new_err(format!("serialize recall result: {e}")))
    }

    async fn remember_message(&self, message: ChatMessage) -> PyResult<()> {
        let memory = self.configured_memory()?;
        let mut guard = memory.lock().await;
        guard
            .remember(&message)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("memory.remember failed: {e}")))
    }

    async fn clear_messages(&self) -> PyResult<()> {
        let memory = self.configured_memory()?;
        let mut guard = memory.lock().await;
        guard
            .clear()
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("memory.clear failed: {e}")))
    }

    async fn message_count(&self) -> PyResult<usize> {
        let memory = self.configured_memory()?;
        let guard = memory.lock().await;
        Ok(guard.size())
    }
}

#[pymethods]
impl PyExecutionMemory {
    fn __repr__(&self) -> String {
        if self.inner.is_some() {
            "ExecutionMemory(<configured>)".to_string()
        } else {
            "ExecutionMemory(<none>)".to_string()
        }
    }

    pub fn is_configured(&self) -> bool {
        self.inner.is_some()
    }

    pub fn recall<'py>(
        &self,
        py: Python<'py>,
        query: String,
        limit: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.configured_memory()?;
        let memory = self.clone();

        crate::async_bridge::future_into_py(py, async move {
            let as_json = memory.recall_messages(query, limit).await?;
            Python::attach(|py| json_value_to_py(py, &as_json))
        })
    }

    pub fn remember<'py>(
        &self,
        py: Python<'py>,
        message: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.configured_memory()?;
        let msg = parse_chat_message_from_any(message)
            .map_err(|e| PyRuntimeError::new_err(format!("invalid message: {e}")))?;
        let memory = self.clone();

        crate::async_bridge::future_into_py(py, async move {
            memory.remember_message(msg).await?;
            Python::attach(|py| Ok(py.None()))
        })
    }

    pub fn clear<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.configured_memory()?;
        let memory = self.clone();

        crate::async_bridge::future_into_py(py, async move {
            memory.clear_messages().await?;
            Python::attach(|py| Ok(py.None()))
        })
    }

    pub fn size<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.configured_memory()?;
        let memory = self.clone();

        crate::async_bridge::future_into_py(py, async move { memory.message_count().await })
    }
}

fn parse_hook_outcome_label(label: &str) -> HookOutcome {
    match label.trim().to_ascii_lowercase().as_str() {
        "abort" | "hookoutcome.abort" => HookOutcome::Abort,
        _ => HookOutcome::Continue,
    }
}

fn parse_hook_outcome(value: Option<Py<PyAny>>) -> HookOutcome {
    let Some(value) = value else {
        return HookOutcome::Continue;
    };

    Python::attach(|py| {
        let bound = value.bind(py);
        if bound.is_none() {
            return HookOutcome::Continue;
        }
        if let Ok(s) = bound.extract::<String>() {
            return parse_hook_outcome_label(&s);
        }
        if let Ok(enum_value) = bound.getattr("value").and_then(|v| v.extract::<String>()) {
            return parse_hook_outcome_label(&enum_value);
        }
        if let Ok(enum_name) = bound.getattr("name").and_then(|v| v.extract::<String>()) {
            return parse_hook_outcome_label(&enum_name);
        }
        HookOutcome::Continue
    })
}

pub(crate) fn task_to_py(task: &Task, py: Python<'_>) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    d.set_item("prompt", task.prompt.clone())?;
    d.set_item("system_prompt", task.system_prompt.clone())?;
    d.set_item("submission_id", task.submission_id.to_string())?;
    d.set_item("completed", task.completed)?;
    d.set_item("result_json", task.result.as_ref().map(|v| v.to_string()))?;
    Ok(d.into_any().unbind())
}

fn parse_role(role: &str) -> ChatRole {
    match role.trim().to_ascii_lowercase().as_str() {
        "system" => ChatRole::System,
        "assistant" => ChatRole::Assistant,
        "tool" => ChatRole::Tool,
        _ => ChatRole::User,
    }
}

fn parse_message_type(message_type: &str) -> Result<MessageType, String> {
    match message_type.trim().to_ascii_lowercase().as_str() {
        "text" => Ok(MessageType::Text),
        other => Err(format!("unsupported message_type '{other}'")),
    }
}

fn chat_response_to_value(response: &dyn ChatResponse) -> Value {
    json!({
        "text": response.text(),
        "tool_calls": response.tool_calls(),
        "thinking": response.thinking(),
        "usage": response.usage(),
    })
}

fn parse_stream_item<T: Serialize>(item: T, label: &str) -> Result<Value, LLMError> {
    serde_json::to_value(item)
        .map_err(|e| LLMError::Generic(format!("failed to serialize {label}: {e}")))
}

fn map_json_stream<T, S>(stream: S, label: &'static str) -> PyResult<LlmJsonStream>
where
    T: Serialize + Send + 'static,
    S: Stream<Item = Result<T, LLMError>> + Send + 'static,
{
    Ok(Box::pin(stream.map(move |item| {
        item.and_then(|value| parse_stream_item(value, label))
    })))
}

fn parse_tools_from_any(tools: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<Tool>>> {
    let Some(tools) = tools else {
        return Ok(None);
    };
    if tools.is_none() {
        return Ok(None);
    }

    let tool_list = tools
        .cast::<PyList>()
        .map_err(|_| PyRuntimeError::new_err("tools must be a list of Tool instances"))?;
    let parsed = tool_list
        .iter()
        .map(|item| {
            let tool = item.extract::<PyRef<'_, PyTool>>().map_err(|_| {
                PyRuntimeError::new_err(
                    "tools must contain Tool instances created by autoagents_py.tool",
                )
            })?;
            let tool_box: Box<dyn ToolT> = Box::new((*tool).clone());
            Ok(to_llm_tool(&tool_box))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(Some(parsed))
}

fn parse_tool_calls_from_value(value: Option<&Value>) -> Result<Vec<ToolCall>, String> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    serde_json::from_value::<Vec<ToolCall>>(value.clone()).map_err(|e| e.to_string())
}

fn parse_chat_message_from_any(value: &Bound<'_, PyAny>) -> PyResult<ChatMessage> {
    if let Ok(content) = value.extract::<String>() {
        return Ok(ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content,
        });
    }

    let dict = value
        .cast::<PyDict>()
        .map_err(|_| PyRuntimeError::new_err("message must be a string or dict"))?;
    let json = py_any_to_json_value(dict.as_any())
        .map_err(|e| PyRuntimeError::new_err(format!("invalid message payload: {e}")))?;

    if let Ok(message) = serde_json::from_value::<ChatMessage>(json.clone()) {
        return Ok(message);
    }

    let object = json
        .as_object()
        .ok_or_else(|| PyRuntimeError::new_err("message must be a JSON object"))?;
    let role = object
        .get("role")
        .and_then(Value::as_str)
        .map(parse_role)
        .unwrap_or(ChatRole::User);
    let tool_calls = object.get("tool_calls");
    let message_type = match object.get("message_type").or_else(|| object.get("type")) {
        Some(Value::String(raw)) => match raw.trim().to_ascii_lowercase().as_str() {
            "tool_use" | "tooluse" => MessageType::ToolUse(
                parse_tool_calls_from_value(tool_calls).map_err(PyRuntimeError::new_err)?,
            ),
            "tool_result" | "toolresult" => MessageType::ToolResult(
                parse_tool_calls_from_value(tool_calls).map_err(PyRuntimeError::new_err)?,
            ),
            other => parse_message_type(other).map_err(PyRuntimeError::new_err)?,
        },
        Some(other) => serde_json::from_value::<MessageType>(other.clone())
            .map_err(|e| PyRuntimeError::new_err(format!("invalid message_type: {e}")))?,
        None => MessageType::Text,
    };
    let content = object
        .get("content")
        .or_else(|| object.get("text"))
        .and_then(Value::as_str)
        .unwrap_or_default();

    Ok(ChatMessage {
        role,
        message_type,
        content: content.to_string(),
    })
}

fn parse_chat_messages_from_any(value: &Bound<'_, PyAny>) -> PyResult<Vec<ChatMessage>> {
    if let Ok(list) = value.cast::<PyList>() {
        return list
            .iter()
            .map(|item| parse_chat_message_from_any(&item))
            .collect();
    }

    Ok(vec![parse_chat_message_from_any(value)?])
}

fn parse_structured_schema(
    schema: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<StructuredOutputFormat>> {
    let Some(schema) = schema else {
        return Ok(None);
    };
    if schema.is_none() {
        return Ok(None);
    }

    if let Ok(text) = schema.extract::<String>() {
        return serde_json::from_str::<StructuredOutputFormat>(&text)
            .map(Some)
            .map_err(|e| PyRuntimeError::new_err(format!("invalid schema: {e}")));
    }

    let value = py_any_to_json_value(schema)
        .map_err(|e| PyRuntimeError::new_err(format!("invalid schema: {e}")))?;
    serde_json::from_value::<StructuredOutputFormat>(value)
        .map(Some)
        .map_err(|e| PyRuntimeError::new_err(format!("invalid schema: {e}")))
}

pub(crate) fn context_to_py(ctx: &Context, py: Python<'_>) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    let cfg = ctx.config();
    d.set_item("id", cfg.id.to_string())?;
    d.set_item("name", cfg.name.clone())?;
    d.set_item("description", cfg.description.clone())?;
    d.set_item("stream", ctx.stream())?;

    let llm_inner = Py::new(py, PyExecutionLLM::new(Arc::clone(ctx.llm())))?;
    let memory_inner = Py::new(py, PyExecutionMemory::new(ctx.memory()))?;
    let execution_module = py.import("autoagents_py.execution")?;
    let llm = execution_module
        .getattr("ExecutionLLM")?
        .call1((llm_inner,))?;
    let memory = execution_module
        .getattr("ExecutionMemory")?
        .call1((memory_inner,))?;
    d.set_item("llm", llm)?;
    d.set_item("memory", memory)?;

    let messages_json = serde_json::to_value(ctx.messages())
        .map_err(|e| PyRuntimeError::new_err(format!("serialize context messages: {e}")))?;
    d.set_item("messages", json_value_to_py(py, &messages_json)?)?;
    Ok(d.into_any().unbind())
}

fn py_from_serializable<T: Serialize>(py: Python<'_>, value: &T) -> PyResult<Py<PyAny>> {
    let as_json = serde_json::to_value(value)
        .map_err(|e| PyRuntimeError::new_err(format!("serialize python hook payload: {e}")))?;
    json_value_to_py(py, &as_json)
}

fn tool_call_to_py(tool_call: &ToolCall, py: Python<'_>) -> PyResult<Py<PyAny>> {
    py_from_serializable(py, tool_call)
}

fn tool_result_to_py(result: &ToolCallResult, py: Python<'_>) -> PyResult<Py<PyAny>> {
    py_from_serializable(py, result)
}

fn agent_output_to_py(result: &PyAgentOutput, py: Python<'_>) -> PyResult<Py<PyAny>> {
    py_from_serializable(py, result)
}

impl PyAgentDef {
    async fn call_optional_hook<F>(
        &self,
        method: &str,
        call: F,
    ) -> Result<Option<Py<PyAny>>, String>
    where
        F: for<'py> FnOnce(Python<'py>, &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>>,
    {
        let Some(hooks) = self.hooks.as_ref() else {
            return Ok(None);
        };
        call_hook_method_async(hooks, self.task_locals.as_ref(), method, call).await
    }

    async fn fire_hook<F>(&self, method: &str, call: F)
    where
        F: for<'py> FnOnce(Python<'py>, &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>>,
    {
        if let Err(err) = self.call_optional_hook(method, call).await {
            self.hook_errors.record(format!(
                "hook {method} failed: {}",
                normalize_hook_error(&err)
            ));
        }
    }

    async fn outcome_hook<F>(&self, method: &str, call: F) -> HookOutcome
    where
        F: for<'py> FnOnce(Python<'py>, &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>>,
    {
        match self.call_optional_hook(method, call).await {
            Ok(value) => parse_hook_outcome(value),
            Err(err) => {
                self.hook_errors.record(format!(
                    "hook {method} failed: {}",
                    normalize_hook_error(&err)
                ));
                HookOutcome::Abort
            }
        }
    }
}

pub(crate) fn call_hook_method_sync<F>(
    hooks: &Py<PyAny>,
    method: &str,
    call: F,
) -> Result<Option<Py<PyAny>>, String>
where
    F: for<'py> FnOnce(Python<'py>, &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>>,
{
    Python::attach(|py| {
        let hooks_obj = hooks.bind(py);
        if !hooks_obj.hasattr(method)? {
            return Ok(None);
        }
        let result_bound = call(py, hooks_obj)?;
        let is_awaitable = crate::async_bridge::is_awaitable(&result_bound)?;
        if is_awaitable {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{method} cannot be async in this call site"
            )));
        }
        Ok(Some(result_bound.unbind()))
    })
    .map_err(|e| e.to_string())
}

pub(crate) async fn call_hook_method_async<F>(
    hooks: &Py<PyAny>,
    task_locals: Option<&TaskLocals>,
    method: &str,
    call: F,
) -> Result<Option<Py<PyAny>>, String>
where
    F: for<'py> FnOnce(Python<'py>, &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>>,
{
    let (result_obj, is_awaitable) = Python::attach(|py| -> PyResult<(Py<PyAny>, bool)> {
        let hooks_obj = hooks.bind(py);
        if !hooks_obj.hasattr(method)? {
            return Ok((py.None(), false));
        }
        let result_bound = call(py, hooks_obj)?;
        let awaitable = crate::async_bridge::is_awaitable(&result_bound)?;
        Ok((result_bound.unbind(), awaitable))
    })
    .map_err(|e| e.to_string())?;

    // Missing method path uses Python None sentinel.
    let is_none = Python::attach(|py| result_obj.bind(py).is_none());
    if is_none {
        return Ok(None);
    }

    crate::async_bridge::resolve_maybe_awaitable(result_obj, is_awaitable, task_locals)
        .await
        .map(Some)
        .map_err(|e| e.to_string())
}

#[async_trait]
impl AgentHooks for PyAgentDef {
    async fn on_agent_create(&self) {
        self.fire_hook("on_agent_create", |_py, hooks_obj| {
            hooks_obj.call_method0("on_agent_create")
        })
        .await;
    }

    async fn on_run_start(&self, task: &Task, ctx: &Context) -> HookOutcome {
        let task_cloned = task.clone();
        self.outcome_hook("on_run_start", |py, hooks_obj| {
            let py_task = task_to_py(&task_cloned, py)?;
            let py_ctx = context_to_py(ctx, py)?;
            hooks_obj.call_method1("on_run_start", (py_task.bind(py), py_ctx.bind(py)))
        })
        .await
    }

    async fn on_run_complete(&self, task: &Task, result: &Self::Output, ctx: &Context) {
        let task_cloned = task.clone();
        let result_cloned = result.clone();
        self.fire_hook("on_run_complete", |py, hooks_obj| {
            let py_task = task_to_py(&task_cloned, py)?;
            let py_result = agent_output_to_py(&result_cloned, py)?;
            let py_ctx = context_to_py(ctx, py)?;
            hooks_obj.call_method1(
                "on_run_complete",
                (py_task.bind(py), py_result.bind(py), py_ctx.bind(py)),
            )
        })
        .await;
    }

    async fn on_turn_start(&self, turn_index: usize, ctx: &Context) {
        self.fire_hook("on_turn_start", |py, hooks_obj| {
            let py_ctx = context_to_py(ctx, py)?;
            hooks_obj.call_method1("on_turn_start", (turn_index, py_ctx.bind(py)))
        })
        .await;
    }

    async fn on_turn_complete(&self, turn_index: usize, ctx: &Context) {
        self.fire_hook("on_turn_complete", |py, hooks_obj| {
            let py_ctx = context_to_py(ctx, py)?;
            hooks_obj.call_method1("on_turn_complete", (turn_index, py_ctx.bind(py)))
        })
        .await;
    }

    async fn on_tool_call(&self, tool_call: &ToolCall, ctx: &Context) -> HookOutcome {
        let tool_call = tool_call.clone();
        self.outcome_hook("on_tool_call", |py, hooks_obj| {
            let py_tool_call = tool_call_to_py(&tool_call, py)?;
            let py_ctx = context_to_py(ctx, py)?;
            hooks_obj.call_method1("on_tool_call", (py_tool_call.bind(py), py_ctx.bind(py)))
        })
        .await
    }

    async fn on_tool_start(&self, tool_call: &ToolCall, ctx: &Context) {
        let tool_call = tool_call.clone();
        self.fire_hook("on_tool_start", |py, hooks_obj| {
            let py_tool_call = tool_call_to_py(&tool_call, py)?;
            let py_ctx = context_to_py(ctx, py)?;
            hooks_obj.call_method1("on_tool_start", (py_tool_call.bind(py), py_ctx.bind(py)))
        })
        .await;
    }

    async fn on_tool_result(&self, tool_call: &ToolCall, result: &ToolCallResult, ctx: &Context) {
        let tool_call = tool_call.clone();
        let result = result.clone();
        self.fire_hook("on_tool_result", |py, hooks_obj| {
            let py_tool_call = tool_call_to_py(&tool_call, py)?;
            let py_result = tool_result_to_py(&result, py)?;
            let py_ctx = context_to_py(ctx, py)?;
            hooks_obj.call_method1(
                "on_tool_result",
                (py_tool_call.bind(py), py_result.bind(py), py_ctx.bind(py)),
            )
        })
        .await;
    }

    async fn on_tool_error(&self, tool_call: &ToolCall, err: Value, ctx: &Context) {
        let tool_call = tool_call.clone();
        self.fire_hook("on_tool_error", |py, hooks_obj| {
            let py_tool_call = tool_call_to_py(&tool_call, py)?;
            let py_err = json_value_to_py(py, &err)?;
            let py_ctx = context_to_py(ctx, py)?;
            hooks_obj.call_method1(
                "on_tool_error",
                (py_tool_call.bind(py), py_err.bind(py), py_ctx.bind(py)),
            )
        })
        .await;
    }

    async fn on_agent_shutdown(&self) {
        self.fire_hook("on_agent_shutdown", |_py, hooks_obj| {
            hooks_obj.call_method0("on_agent_shutdown")
        })
        .await;
    }
}

// ── Shared stream type ───────────────────────────────────────────────────────

pub type AgentOutputStream =
    Pin<Box<dyn Stream<Item = Result<PyAgentOutput, autoagents_core::error::Error>> + Send>>;

/// Opaque send handle returned by `PyExecutorBuildable::build_actor`.
pub type ActorSendFn = Arc<dyn Fn(Task) -> Result<(), String> + Send + Sync>;

/// Return type of `PyExecutorBuildable::build_direct`.
pub type BuildDirectResult =
    Pin<Box<dyn Future<Output = PyResult<(Arc<dyn PyRunnable>, BoxEventStream<Event>)>> + Send>>;

/// Return type of `PyExecutorBuildable::build_actor`.
pub type BuildActorResult = Pin<Box<dyn Future<Output = PyResult<ActorSendFn>> + Send>>;

// ── PyRunnable ───────────────────────────────────────────────────────────────

/// A fully built DirectAgent that can process tasks from Python.
///
/// Blanket-implemented for every `BaseAgent<T, DirectAgent>` whose executor
/// output converts to `PyAgentOutput`, so no new executor needs to be added
/// to any match arm.
#[async_trait]
pub trait PyRunnable: Send + Sync {
    async fn run(&self, task: Task) -> Result<PyAgentOutput, String>;
    async fn run_stream(&self, task: Task) -> Result<AgentOutputStream, String>;
}

#[async_trait]
impl<T> PyRunnable for BaseAgent<T, DirectAgent>
where
    T: AgentDeriveT<Output = PyAgentOutput> + AgentExecutor + AgentHooks + Send + Sync + 'static,
    PyAgentOutput: From<<T as AgentExecutor>::Output>,
    <T as AgentExecutor>::Error: Into<RunnableAgentError>,
{
    async fn run(&self, task: Task) -> Result<PyAgentOutput, String> {
        self.run(task).await.map_err(|e| e.to_string())
    }

    async fn run_stream(&self, task: Task) -> Result<AgentOutputStream, String> {
        self.run_stream(task).await.map_err(|e| e.to_string())
    }
}

// ── PyExecutorBuildable ──────────────────────────────────────────────────────

/// Mirrors Rust's `AgentBuilder<T: AgentExecutor>` generic bound for Python.
///
/// Implement this for a Python executor `#[pyclass]` to make it usable with
/// `PyAgentBuilder` without modifying `PyAgentBuilder` itself.
pub trait PyExecutorBuildable: Send + Sync {
    /// Build a `DirectAgent`-backed runnable and its event stream.
    fn build_direct(
        &self,
        agent_def: PyAgentDef,
        llm: Arc<dyn LLMProvider>,
        memory: Box<dyn MemoryProvider>,
    ) -> BuildDirectResult;

    /// Build an actor-based agent registered in the given runtime.
    /// Returns the mailbox send function; `PyAgentBuilder` wraps it in
    /// `PyActorAgentHandle`.
    fn build_actor(
        &self,
        agent_def: PyAgentDef,
        llm: Arc<dyn LLMProvider>,
        memory: Box<dyn MemoryProvider>,
        runtime: Arc<dyn Runtime>,
        topics: Vec<String>,
    ) -> BuildActorResult;
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoagents_core::agent::memory::SlidingWindowMemory;
    use autoagents_llm::chat::{ChatProvider, Usage};
    use autoagents_llm::completion::{CompletionProvider, CompletionRequest, CompletionResponse};
    use autoagents_llm::embedding::EmbeddingProvider;
    use autoagents_llm::models::ModelsProvider;
    use autoagents_llm::{FunctionCall, async_trait};
    use futures::stream;
    use serde_json::json;
    use std::ffi::CString;
    use std::future::Future;

    fn init_python() {
        Python::initialize();
    }

    fn init_runtime_bridge() {
        init_python();
        let runtime = crate::runtime::get_runtime().expect("shared runtime should initialize");
        let _ = pyo3_async_runtimes::tokio::init_with_runtime(runtime);
    }

    fn immediate_awaitable(py: Python<'_>, value: &str) -> PyResult<Py<PyAny>> {
        let module = PyModule::from_code(
            py,
            &CString::new(
                "class ImmediateAwaitable:\n\
                 \n\
                 \tdef __init__(self, value):\n\
                 \t\tself.value = value\n\
                 \n\
                 \tdef __await__(self):\n\
                 \t\tif False:\n\
                 \t\t\tyield None\n\
                 \t\treturn self.value\n",
            )
            .expect("python module source should be valid CString"),
            &CString::new("autoagents_py/tests/py_agent.py")
                .expect("filename should be a valid CString"),
            &CString::new("autoagents_py_agent_tests")
                .expect("module name should be a valid CString"),
        )?;
        let awaitable = module
            .getattr("ImmediateAwaitable")?
            .call1((value.to_string(),))?;
        Ok(awaitable.unbind())
    }

    fn block_on_test<T>(future: impl Future<Output = T>) -> T {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime should build")
            .block_on(future)
    }

    #[derive(Debug)]
    struct MockChatResponse {
        text: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
        thinking: Option<String>,
        usage: Option<Usage>,
    }

    impl std::fmt::Display for MockChatResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.text.as_deref().unwrap_or(""))
        }
    }

    impl ChatResponse for MockChatResponse {
        fn text(&self) -> Option<String> {
            self.text.clone()
        }

        fn tool_calls(&self) -> Option<Vec<ToolCall>> {
            self.tool_calls.clone()
        }

        fn thinking(&self) -> Option<String> {
            self.thinking.clone()
        }

        fn usage(&self) -> Option<Usage> {
            self.usage.clone()
        }
    }

    struct MockLLMProvider;

    #[async_trait]
    impl ChatProvider for MockLLMProvider {
        async fn chat_with_tools(
            &self,
            messages: &[ChatMessage],
            tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            let tool_calls = tools.and_then(|tools| {
                tools.first().map(|tool| {
                    vec![ToolCall {
                        id: "call_1".to_string(),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: tool.function.name.clone(),
                            arguments: "{\"city\":\"Bangalore\"}".to_string(),
                        },
                    }]
                })
            });

            Ok(Box::new(MockChatResponse {
                text: Some(format!("reply:{}", messages[0].content)),
                tool_calls,
                thinking: Some("reasoning".to_string()),
                usage: Some(Usage {
                    prompt_tokens: 3,
                    completion_tokens: 5,
                    total_tokens: 8,
                    completion_tokens_details: None,
                    prompt_tokens_details: None,
                }),
            }))
        }

        async fn chat_with_web_search(
            &self,
            input: String,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            Ok(Box::new(MockChatResponse {
                text: Some(format!("web:{input}")),
                tool_calls: None,
                thinking: None,
                usage: None,
            }))
        }

        async fn chat_stream(
            &self,
            _messages: &[ChatMessage],
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
        {
            Ok(Box::pin(stream::iter(vec![
                Ok("alpha".to_string()),
                Ok("beta".to_string()),
            ])))
        }

        async fn chat_stream_struct(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<
            std::pin::Pin<
                Box<
                    dyn Stream<Item = Result<autoagents_llm::chat::StreamResponse, LLMError>>
                        + Send,
                >,
            >,
            LLMError,
        > {
            Ok(Box::pin(stream::iter(vec![Ok(
                autoagents_llm::chat::StreamResponse {
                    choices: vec![autoagents_llm::chat::StreamChoice {
                        delta: autoagents_llm::chat::StreamDelta {
                            content: Some("json-chunk".to_string()),
                            reasoning_content: None,
                            tool_calls: None,
                        },
                    }],
                    usage: None,
                },
            )])))
        }

        async fn chat_stream_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<
            std::pin::Pin<
                Box<dyn Stream<Item = Result<autoagents_llm::chat::StreamChunk, LLMError>> + Send>,
            >,
            LLMError,
        > {
            Ok(Box::pin(stream::iter(vec![
                Ok(autoagents_llm::chat::StreamChunk::Text(
                    "tool-chunk".to_string(),
                )),
                Ok(autoagents_llm::chat::StreamChunk::Done {
                    stop_reason: "end_turn".to_string(),
                }),
            ])))
        }
    }

    #[async_trait]
    impl CompletionProvider for MockLLMProvider {
        async fn complete(
            &self,
            _req: &CompletionRequest,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<CompletionResponse, LLMError> {
            Ok(CompletionResponse {
                text: "completion".to_string(),
            })
        }
    }

    #[async_trait]
    impl EmbeddingProvider for MockLLMProvider {
        async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
            Ok(input.into_iter().map(|_| vec![0.1, 0.2]).collect())
        }
    }

    #[async_trait]
    impl ModelsProvider for MockLLMProvider {}

    impl LLMProvider for MockLLMProvider {}

    #[pyclass]
    struct HookObject;

    #[pymethods]
    impl HookObject {
        fn sync_value(&self) -> &str {
            "abort"
        }

        fn async_value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            immediate_awaitable(py, "abort")
        }
    }

    #[test]
    fn hook_and_message_helpers_cover_branching_cases() {
        init_runtime_bridge();
        let state = HookErrorState::default();
        assert_eq!(state.take(), None);
        state.record("first error");
        state.record("second error");
        assert_eq!(state.take(), Some("first error".to_string()));
        state.clear();
        assert_eq!(state.take(), None);

        assert_eq!(
            normalize_hook_error("ValueError: broken state"),
            "broken state"
        );
        assert_eq!(normalize_hook_error("plain message"), "plain message");
        assert!(matches!(
            parse_hook_outcome_label("abort"),
            HookOutcome::Abort
        ));
        assert!(matches!(
            parse_hook_outcome_label("HookOutcome.Abort"),
            HookOutcome::Abort
        ));
        assert!(matches!(
            parse_hook_outcome_label("continue"),
            HookOutcome::Continue
        ));

        Python::attach(|py| {
            let string_message = "hello".into_pyobject(py).expect("string should convert");
            let parsed = parse_chat_message_from_any(string_message.as_any())
                .expect("string message should parse");
            assert_eq!(parsed.role, ChatRole::User);
            assert_eq!(parsed.content, "hello");

            let tool_call = json!([{
                "id": "call_1",
                "type": "function",
                "function": {"name": "weather", "arguments": "{\"city\":\"Bangalore\"}"}
            }]);
            let message_dict = json_value_to_py(
                py,
                &json!({
                    "role": "assistant",
                    "message_type": "tool_use",
                    "content": "tool request",
                    "tool_calls": tool_call,
                }),
            )
            .expect("dict should convert");
            let parsed = parse_chat_message_from_any(message_dict.bind(py))
                .expect("tool message should parse");
            assert_eq!(parsed.role, ChatRole::Assistant);
            assert!(matches!(parsed.message_type, MessageType::ToolUse(_)));

            let list = json_value_to_py(
                py,
                &json!([
                    {"role": "system", "content": "setup"},
                    {"role": "tool", "text": "done"}
                ]),
            )
            .expect("list should convert");
            let parsed =
                parse_chat_messages_from_any(list.bind(py)).expect("message list should parse");
            assert_eq!(parsed.len(), 2);
            assert_eq!(parsed[0].role, ChatRole::System);
            assert_eq!(parsed[1].role, ChatRole::Tool);

            let bad_message = json_value_to_py(py, &json!({"message_type": "image"}))
                .expect("dict should convert");
            let err = parse_chat_message_from_any(bad_message.bind(py))
                .expect_err("unsupported message type should fail");
            assert!(err.to_string().contains("unsupported message_type"));

            let tool = PyTool::new(
                "weather".to_string(),
                "Fetch weather".to_string(),
                "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}".to_string(),
                py.None(),
                None,
            )
            .expect("tool should build");
            let tool = Py::new(py, tool).expect("tool pyclass should build");
            let tools = PyList::empty(py);
            tools.append(tool.bind(py)).expect("tool should append");
            let parsed_tools = parse_tools_from_any(Some(tools.as_any()))
                .expect("tool list should parse")
                .expect("tool list should exist");
            assert_eq!(parsed_tools.len(), 1);
            assert_eq!(parsed_tools[0].function.name, "weather");

            let schema = parse_structured_schema(Some(
                json_value_to_py(py, &json!({"name": "Answer", "schema": {"type": "object"}}))
                    .expect("schema dict should convert")
                    .bind(py),
            ))
            .expect("schema dict should parse")
            .expect("schema should exist");
            assert_eq!(schema.name, "Answer");

            let schema_text = json!({"name": "AnswerText", "schema": {"type": "object"}})
                .to_string()
                .into_pyobject(py)
                .expect("schema text should convert");
            let parsed = parse_structured_schema(Some(schema_text.as_any()))
                .expect("schema text should parse")
                .expect("schema should exist");
            assert_eq!(parsed.name, "AnswerText");

            let types = py.import("types").expect("types should import");
            let value_ns = types
                .getattr("SimpleNamespace")
                .expect("SimpleNamespace should exist")
                .call1(())
                .expect("namespace should create");
            value_ns
                .setattr("value", "abort")
                .expect("value should set");
            assert!(matches!(
                parse_hook_outcome(Some(value_ns.unbind())),
                HookOutcome::Abort
            ));

            let name_ns = types
                .getattr("SimpleNamespace")
                .expect("SimpleNamespace should exist")
                .call1(())
                .expect("namespace should create");
            name_ns.setattr("name", "Abort").expect("name should set");
            assert!(matches!(
                parse_hook_outcome(Some(name_ns.unbind())),
                HookOutcome::Abort
            ));
            assert!(matches!(parse_hook_outcome(None), HookOutcome::Continue));
        });
    }

    #[test]
    fn call_hook_helpers_support_sync_and_async_values() {
        init_runtime_bridge();
        Python::attach(|py| {
            let hooks = Py::new(py, HookObject)
                .expect("hooks should create")
                .into_any();

            let sync = call_hook_method_sync(&hooks, "sync_value", |_py, obj| {
                obj.call_method0("sync_value")
            })
            .expect("sync hook should succeed")
            .expect("sync hook should exist");
            assert_eq!(
                sync.bind(py)
                    .extract::<String>()
                    .expect("sync value should be a string"),
                "abort"
            );

            let missing = call_hook_method_sync(&hooks, "missing_method", |_py, obj| {
                obj.call_method0("missing_method")
            })
            .expect("missing sync hook should not fail");
            assert!(missing.is_none());

            let err = call_hook_method_sync(&hooks, "async_value", |py, obj| {
                let awaitable = obj.call_method0("async_value")?;
                let _ = py;
                Ok(awaitable)
            })
            .expect_err("async hook should be rejected at sync call site");
            assert!(err.contains("cannot be async"));
        });

        Python::attach(|py| -> PyResult<()> {
            let event_loop = py.import("asyncio")?.call_method0("new_event_loop")?;
            let hooks = Py::new(py, HookObject)?.into_any();
            let locals = TaskLocals::new(event_loop.clone()).copy_context(py)?;
            pyo3_async_runtimes::tokio::run_until_complete(event_loop, async move {
                let async_value =
                    call_hook_method_async(&hooks, Some(&locals), "async_value", |_py, obj| {
                        obj.call_method0("async_value")
                    })
                    .await
                    .expect("async hook should resolve")
                    .expect("async hook should exist");
                Python::attach(|py| {
                    assert_eq!(
                        async_value
                            .bind(py)
                            .extract::<String>()
                            .expect("async hook result should be a string"),
                        "abort"
                    );
                });

                let missing =
                    call_hook_method_async(&hooks, Some(&locals), "missing_method", |_py, obj| {
                        obj.call_method0("missing_method")
                    })
                    .await
                    .expect("missing async hook should not fail");
                assert!(missing.is_none());
                Ok(())
            })
        })
        .expect("temporary event loop should run");
    }

    #[test]
    fn execution_llm_methods_cover_provider_helpers_and_streams() {
        init_python();
        let llm = PyExecutionLLM::new(Arc::new(MockLLMProvider));
        assert_eq!(llm.__repr__(), "ExecutionLLM(<context>)");

        let chat =
            block_on_test(llm.chat_value(vec![ChatMessage::user().content("hello").build()], None))
                .expect("chat should succeed");
        assert_eq!(chat["text"], json!("reply:hello"));
        assert_eq!(chat["thinking"], json!("reasoning"));
        assert_eq!(chat["usage"]["total_tokens"], json!(8));

        let schema = serde_json::from_value::<StructuredOutputFormat>(json!({
            "name": "Answer",
            "schema": {"type": "object"}
        }))
        .expect("schema should parse");
        let chat_struct = block_on_test(llm.chat_value(
            vec![ChatMessage::user().content("schema").build()],
            Some(schema),
        ))
        .expect("structured chat should succeed");
        assert_eq!(chat_struct["text"], json!("reply:schema"));

        let weather_tool = Tool {
            tool_type: "function".to_string(),
            function: autoagents_llm::chat::FunctionTool {
                name: "weather".to_string(),
                description: "Fetch weather".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }),
            },
        };
        let tools = Some(vec![weather_tool.clone()]);
        let tool_chat = block_on_test(llm.chat_with_tools_value(
            vec![ChatMessage::user().content("tool").build()],
            tools.clone(),
            None,
        ))
        .expect("tool chat should succeed");
        assert_eq!(
            tool_chat["tool_calls"][0]["function"]["name"],
            json!("weather")
        );

        let web = block_on_test(llm.chat_with_web_search_value("current news".to_string()))
            .expect("web search chat should succeed");
        assert_eq!(web["text"], json!("web:current news"));

        let string_stream = block_on_test(
            llm.build_string_stream(vec![ChatMessage::user().content("stream").build()], None),
        )
        .expect("string stream should build");
        assert_eq!(
            block_on_test(string_stream.next_chunk()).expect("first string chunk should succeed"),
            Some("alpha".to_string())
        );
        assert_eq!(
            block_on_test(string_stream.next_chunk()).expect("second string chunk should succeed"),
            Some("beta".to_string())
        );
        assert!(
            block_on_test(string_stream.next_chunk())
                .expect("string stream should end cleanly")
                .is_none()
        );

        let json_stream = block_on_test(llm.build_struct_stream(
            vec![ChatMessage::user().content("json").build()],
            None,
            None,
            "llm.chat_stream_struct",
            "llm.chat_stream_struct",
        ))
        .expect("json stream should build");
        let json_chunk = block_on_test(json_stream.next_chunk())
            .expect("json stream chunk should succeed")
            .expect("json stream should yield a chunk");
        assert_eq!(
            json_chunk["choices"][0]["delta"]["content"],
            json!("json-chunk")
        );

        let tool_stream = block_on_test(llm.build_tool_stream(
            vec![ChatMessage::user().content("tool-stream").build()],
            tools,
            None,
            "llm.chat_stream_with_tools",
            "llm.chat_stream_with_tools",
        ))
        .expect("tool stream should build");
        let tool_chunk = block_on_test(tool_stream.next_chunk())
            .expect("tool stream chunk should succeed")
            .expect("tool stream should yield a chunk");
        assert_eq!(tool_chunk["Text"], json!("tool-chunk"));
    }

    #[test]
    fn execution_memory_methods_cover_provider_and_validate_configuration() {
        init_python();

        let unconfigured = PyExecutionMemory::new(None);
        assert_eq!(unconfigured.__repr__(), "ExecutionMemory(<none>)");
        assert!(!unconfigured.is_configured());

        Python::attach(|py| {
            let err = unconfigured
                .size(py)
                .expect_err("size should fail when memory is not configured");
            assert!(err.to_string().contains("memory is not configured"));
        });

        let memory = PyExecutionMemory::new(Some(Arc::new(tokio::sync::Mutex::new(Box::new(
            SlidingWindowMemory::new(8),
        )
            as Box<dyn MemoryProvider>))));
        assert_eq!(memory.__repr__(), "ExecutionMemory(<configured>)");
        assert!(memory.is_configured());

        block_on_test(memory.remember_message(ChatMessage::user().content("hello memory").build()))
            .expect("remember should succeed");
        assert_eq!(
            block_on_test(memory.message_count()).expect("size should succeed"),
            1
        );

        let recall = block_on_test(memory.recall_messages("hello".to_string(), Some(10)))
            .expect("recall should succeed");
        assert_eq!(recall[0]["content"], json!("hello memory"));

        block_on_test(memory.clear_messages()).expect("clear should succeed");
        assert_eq!(
            block_on_test(memory.message_count()).expect("size should succeed"),
            0
        );
    }

    #[test]
    fn serialization_helpers_convert_expected_values() {
        init_python();
        Python::attach(|py| {
            let task = Task::new("inspect me").with_system_prompt("system prompt");
            let task_value = task_to_py(&task, py).expect("task should convert");
            let task_json =
                py_any_to_json_value(task_value.bind(py)).expect("task json should convert");
            assert_eq!(task_json["prompt"], json!("inspect me"));
            assert_eq!(task_json["system_prompt"], json!("system prompt"));

            let tool_call = ToolCall {
                id: "call_1".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "weather".to_string(),
                    arguments: "{\"city\":\"Bangalore\"}".to_string(),
                },
            };
            let tool_result = ToolCallResult {
                tool_name: "weather".to_string(),
                success: true,
                arguments: json!({"city": "Bangalore"}),
                result: json!({"temp_c": 28}),
            };
            let output = PyAgentOutput {
                response: "done".to_string(),
                tool_calls: vec![tool_result.clone()],
                executions: vec![json!({"id": "exec_1"})],
                done: true,
            };

            let tool_call_json = py_any_to_json_value(
                tool_call_to_py(&tool_call, py)
                    .expect("tool call should convert")
                    .bind(py),
            )
            .expect("tool call json should convert");
            assert_eq!(tool_call_json["function"]["name"], json!("weather"));

            let tool_result_json = py_any_to_json_value(
                tool_result_to_py(&tool_result, py)
                    .expect("tool result should convert")
                    .bind(py),
            )
            .expect("tool result json should convert");
            assert_eq!(tool_result_json["result"]["temp_c"], json!(28));

            let output_json = py_any_to_json_value(
                agent_output_to_py(&output, py)
                    .expect("agent output should convert")
                    .bind(py),
            )
            .expect("agent output json should convert");
            assert_eq!(output_json["response"], json!("done"));

            let response_value = chat_response_to_value(&MockChatResponse {
                text: Some("text".to_string()),
                tool_calls: None,
                thinking: Some("think".to_string()),
                usage: None,
            });
            assert_eq!(response_value["text"], json!("text"));
            assert_eq!(response_value["thinking"], json!("think"));
        });

        let stream_item = parse_stream_item(json!({"ok": true}), "stream item")
            .expect("stream item should serialize");
        assert_eq!(stream_item["ok"], json!(true));

        let runtime = crate::runtime::get_runtime().expect("shared runtime should initialize");
        let stream = map_json_stream(
            stream::iter(vec![
                Ok(json!({"kind": "first"})),
                Err(LLMError::Generic("boom".to_string())),
            ]),
            "mapped stream",
        )
        .expect("stream should map");
        let mut collected = runtime.block_on(async { stream.collect::<Vec<_>>().await });
        assert_eq!(
            collected.remove(0).expect("first item should succeed")["kind"],
            json!("first")
        );
        assert!(
            collected
                .remove(0)
                .expect_err("second item should fail")
                .to_string()
                .contains("boom")
        );
    }

    #[test]
    fn agent_output_and_agent_def_helpers_cover_conversions_and_metadata() {
        init_python();
        let tool_call = ToolCallResult {
            tool_name: "lookup".to_string(),
            success: true,
            arguments: json!({"q": "rust"}),
            result: json!({"matches": 1}),
        };
        let execution = CodeActExecutionRecord {
            execution_id: "exec_1".to_string(),
            source: "return 1;".to_string(),
            console: vec!["ok".to_string()],
            tool_calls: vec![tool_call.clone()],
            result: Some(json!(1)),
            success: true,
            error: None,
            duration_ms: 5,
        };

        assert_eq!(PyAgentOutput::output_schema(), "{}");
        assert_eq!(PyAgentOutput::structured_output_format(), Value::Null);
        assert_eq!(
            collect_codeact_tool_calls(std::slice::from_ref(&execution)).len(),
            1
        );

        let react = PyAgentOutput::from(ReActAgentOutput {
            response: "react".to_string(),
            tool_calls: vec![tool_call.clone()],
            done: false,
        });
        assert_eq!(react.response, "react");
        assert_eq!(react.tool_calls.len(), 1);
        assert!(react.executions.is_empty());
        assert!(!react.done);

        let codeact = PyAgentOutput::from(CodeActAgentOutput {
            response: "codeact".to_string(),
            executions: vec![execution.clone()],
            done: true,
        });
        assert_eq!(codeact.response, "codeact");
        assert_eq!(codeact.tool_calls.len(), 1);
        assert_eq!(codeact.executions.len(), 1);
        assert!(codeact.done);

        let basic = PyAgentOutput::from(BasicAgentOutput {
            response: "basic".to_string(),
            done: true,
        });
        assert_eq!(basic.response, "basic");
        assert!(basic.tool_calls.is_empty());
        assert!(basic.executions.is_empty());
        assert!(basic.done);

        let value = Value::from(PyAgentOutput {
            response: "serialized".to_string(),
            tool_calls: vec![tool_call],
            executions: vec![json!({"id": "exec_1"})],
            done: true,
        });
        assert_eq!(value["response"], json!("serialized"));
        assert_eq!(value["executions"][0]["id"], json!("exec_1"));

        let agent_def = PyAgentDef {
            name: "planner".to_string(),
            description: "Plans work".to_string(),
            tools: Vec::new(),
            output_schema: Some(json!({"type": "object"})),
            hooks: None,
            task_locals: None,
            hook_errors: HookErrorState::default(),
        };
        let cloned = agent_def.clone();
        assert!(format!("{cloned:?}").contains("planner"));
        assert_eq!(cloned.name(), "planner");
        assert_eq!(cloned.description(), "Plans work");
        assert_eq!(cloned.output_schema(), Some(json!({"type": "object"})));
        assert!(cloned.tools().is_empty());
    }
}
