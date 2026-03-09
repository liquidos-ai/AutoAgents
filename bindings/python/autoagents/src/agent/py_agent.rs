use crate::convert::{json_value_to_py, py_any_to_json_value};
use crate::tool::PyTool;
use async_trait::async_trait;
use autoagents_core::agent::error::RunnableAgentError;
use autoagents_core::agent::memory::MemoryProvider;
use autoagents_core::agent::prebuilt::executor::BasicAgentOutput;
use autoagents_core::agent::prebuilt::executor::ReActAgentOutput;
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
use std::sync::Arc;

/// Output type for `PyAgentDef`.
///
/// We cannot add a foreign-trait impl (`AgentOutputT`) to the foreign type
/// `ReActAgentOutput`, so we wrap it here and forward `From`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyAgentOutput {
    pub response: String,
    pub tool_calls: Vec<ToolCallResult>,
    pub done: bool,
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

impl From<ReActAgentOutput> for PyAgentOutput {
    fn from(r: ReActAgentOutput) -> Self {
        PyAgentOutput {
            response: r.response,
            tool_calls: r.tool_calls,
            done: r.done,
        }
    }
}

impl From<BasicAgentOutput> for PyAgentOutput {
    fn from(r: BasicAgentOutput) -> Self {
        PyAgentOutput {
            response: r.response,
            tool_calls: Vec::new(),
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
        let llm = Arc::clone(&self.inner);

        crate::async_bridge::future_into_py(py, async move {
            let response = llm
                .chat(&messages, schema)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("llm.chat failed: {e}")))?;
            Python::attach(|py| json_value_to_py(py, &chat_response_to_value(response.as_ref())))
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
        let llm = Arc::clone(&self.inner);

        crate::async_bridge::future_into_py(py, async move {
            let response = llm
                .chat_with_tools(&messages, tools.as_deref(), schema)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("llm.chat_with_tools failed: {e}")))?;
            Python::attach(|py| json_value_to_py(py, &chat_response_to_value(response.as_ref())))
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
        let llm = Arc::clone(&self.inner);

        crate::async_bridge::future_into_py(py, async move {
            let response = llm.chat_with_web_search(input).await.map_err(|e| {
                PyRuntimeError::new_err(format!("llm.chat_with_web_search failed: {e}"))
            })?;
            Python::attach(|py| json_value_to_py(py, &chat_response_to_value(response.as_ref())))
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
        let llm = Arc::clone(&self.inner);

        crate::async_bridge::future_into_py(py, async move {
            let stream = llm
                .chat_stream(&messages, schema)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("llm.chat_stream failed: {e}")))?;

            Python::attach(|py| {
                PyExecutionStringStream {
                    stream: Arc::new(tokio::sync::Mutex::new(stream)),
                }
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
        let llm = Arc::clone(&self.inner);

        crate::async_bridge::future_into_py(py, async move {
            let stream = llm
                .chat_stream_struct(&messages, tools.as_deref(), schema)
                .await
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("llm.chat_stream_struct failed: {e}"))
                })?;
            let stream = map_json_stream(stream, "llm.chat_stream_struct")?;

            Python::attach(|py| {
                PyExecutionJsonStream {
                    stream: Arc::new(tokio::sync::Mutex::new(stream)),
                }
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
        let llm = Arc::clone(&self.inner);

        crate::async_bridge::future_into_py(py, async move {
            let stream = llm
                .chat_stream_with_tools(&messages, tools.as_deref(), schema)
                .await
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("llm.chat_stream_with_tools failed: {e}"))
                })?;
            let stream = map_json_stream(stream, "llm.chat_stream_with_tools")?;

            Python::attach(|py| {
                PyExecutionJsonStream {
                    stream: Arc::new(tokio::sync::Mutex::new(stream)),
                }
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

#[pymethods]
impl PyExecutionStringStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = Arc::clone(&self.stream);

        crate::async_bridge::future_into_py(py, async move {
            let next = {
                let mut guard = stream.lock().await;
                guard.next().await
            };

            match next {
                Some(Ok(chunk)) => Ok(chunk),
                Some(Err(err)) => Err(PyRuntimeError::new_err(err.to_string())),
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

#[pymethods]
impl PyExecutionJsonStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = Arc::clone(&self.stream);

        crate::async_bridge::future_into_py(py, async move {
            let next = {
                let mut guard = stream.lock().await;
                guard.next().await
            };

            match next {
                Some(Ok(chunk)) => Python::attach(|py| json_value_to_py(py, &chunk)),
                Some(Err(err)) => Err(PyRuntimeError::new_err(err.to_string())),
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
        let Some(memory) = self.inner.clone() else {
            return Err(PyRuntimeError::new_err("memory is not configured"));
        };

        crate::async_bridge::future_into_py(py, async move {
            let guard = memory.lock().await;
            let messages = guard
                .recall(&query, limit)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("memory.recall failed: {e}")))?;
            let as_json = serde_json::to_value(messages)
                .map_err(|e| PyRuntimeError::new_err(format!("serialize recall result: {e}")))?;
            Python::attach(|py| json_value_to_py(py, &as_json))
        })
    }

    pub fn remember<'py>(
        &self,
        py: Python<'py>,
        message: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let Some(memory) = self.inner.clone() else {
            return Err(PyRuntimeError::new_err("memory is not configured"));
        };
        let msg = parse_chat_message_from_any(message)
            .map_err(|e| PyRuntimeError::new_err(format!("invalid message: {e}")))?;

        crate::async_bridge::future_into_py(py, async move {
            let mut guard = memory.lock().await;
            guard
                .remember(&msg)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("memory.remember failed: {e}")))?;
            Python::attach(|py| Ok(py.None()))
        })
    }

    pub fn clear<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Some(memory) = self.inner.clone() else {
            return Err(PyRuntimeError::new_err("memory is not configured"));
        };

        crate::async_bridge::future_into_py(py, async move {
            let mut guard = memory.lock().await;
            guard
                .clear()
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("memory.clear failed: {e}")))?;
            Python::attach(|py| Ok(py.None()))
        })
    }

    pub fn size<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Some(memory) = self.inner.clone() else {
            return Err(PyRuntimeError::new_err("memory is not configured"));
        };

        crate::async_bridge::future_into_py(py, async move {
            let guard = memory.lock().await;
            Ok(guard.size())
        })
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
                    "tools must contain Tool instances created by autoagents.tool",
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
    let execution_module = py.import("autoagents.execution")?;
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
        let _ = self.call_optional_hook(method, call).await;
    }

    async fn outcome_hook<F>(&self, method: &str, call: F) -> HookOutcome
    where
        F: for<'py> FnOnce(Python<'py>, &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>>,
    {
        match self.call_optional_hook(method, call).await {
            Ok(value) => parse_hook_outcome(value),
            Err(_) => HookOutcome::Continue,
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
