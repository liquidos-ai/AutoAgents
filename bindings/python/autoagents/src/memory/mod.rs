use crate::convert::{json_value_to_py, py_any_to_json_value};
use async_trait::async_trait;
use autoagents_core::agent::memory::{
    MemoryProvider as CoreMemoryProvider, MemoryType, SlidingWindowMemory,
};
use autoagents_llm::chat::{ChatMessage, ChatRole, MessageType};
use autoagents_llm::error::LLMError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[pyclass(module = "autoagents_py", name = "MemoryProvider", frozen)]
pub struct PyMemoryProvider {
    inner: Box<dyn CoreMemoryProvider>,
    repr: String,
}

#[pymethods]
impl PyMemoryProvider {
    fn __repr__(&self) -> String {
        self.repr.clone()
    }
}

impl PyMemoryProvider {
    pub fn new(inner: Box<dyn CoreMemoryProvider>, repr: String) -> Self {
        Self { inner, repr }
    }

    pub fn clone_memory(&self) -> Box<dyn CoreMemoryProvider> {
        self.inner.clone_box()
    }
}

struct PyInjectedMemory {
    inner: Py<PyAny>,
    size_cache: Arc<AtomicUsize>,
}

impl Clone for PyInjectedMemory {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            inner: self.inner.clone_ref(py),
            size_cache: Arc::clone(&self.size_cache),
        })
    }
}

fn provider_error(message: impl Into<String>) -> LLMError {
    LLMError::ProviderError(message.into())
}

fn parse_role(role: &str) -> ChatRole {
    match role.trim().to_ascii_lowercase().as_str() {
        "system" => ChatRole::System,
        "assistant" => ChatRole::Assistant,
        "tool" => ChatRole::Tool,
        _ => ChatRole::User,
    }
}

fn parse_chat_message_value(value: serde_json::Value) -> Result<ChatMessage, LLMError> {
    if let Ok(message) = serde_json::from_value::<ChatMessage>(value.clone()) {
        return Ok(message);
    }

    let object = value
        .as_object()
        .ok_or_else(|| provider_error("memory message must be a string or object"))?;
    let role = object
        .get("role")
        .and_then(serde_json::Value::as_str)
        .map(parse_role)
        .unwrap_or(ChatRole::User);
    let content = object
        .get("content")
        .or_else(|| object.get("text"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();

    Ok(ChatMessage {
        role,
        message_type: MessageType::Text,
        content,
    })
}

fn parse_chat_messages_from_py(value: &Bound<'_, PyAny>) -> Result<Vec<ChatMessage>, LLMError> {
    let json = py_any_to_json_value(value)
        .map_err(|e| provider_error(format!("invalid memory recall payload: {e}")))?;
    match json {
        serde_json::Value::Array(items) => {
            items.into_iter().map(parse_chat_message_value).collect()
        }
        _ => Err(provider_error(
            "memory recall() must return a list of messages",
        )),
    }
}

fn method_not_awaitable(method: &str) -> PyErr {
    PyRuntimeError::new_err(format!("memory.{method} must be synchronous"))
}

fn call_memory_size(memory: &Py<PyAny>) -> Result<usize, LLMError> {
    Python::attach(|py| -> Result<usize, LLMError> {
        let obj = memory.bind(py);
        if !obj
            .hasattr("size")
            .map_err(|e| provider_error(e.to_string()))?
        {
            return Err(provider_error(
                "custom memory provider must implement size()",
            ));
        }
        let result = obj
            .call_method0("size")
            .map_err(|e| provider_error(e.to_string()))?;
        if crate::async_bridge::is_awaitable(&result).map_err(|e| provider_error(e.to_string()))? {
            return Err(provider_error(method_not_awaitable("size").to_string()));
        }
        result
            .extract::<usize>()
            .map_err(|e| provider_error(format!("memory.size returned invalid value: {e}")))
    })
}

async fn call_memory_method_async<F>(
    memory: &Py<PyAny>,
    method: &str,
    call: F,
) -> Result<Py<PyAny>, LLMError>
where
    F: for<'py> FnOnce(Python<'py>, &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>>,
{
    let (result_obj, is_awaitable) = Python::attach(|py| -> Result<(Py<PyAny>, bool), LLMError> {
        let obj = memory.bind(py);
        if !obj
            .hasattr(method)
            .map_err(|e| provider_error(e.to_string()))?
        {
            return Err(provider_error(format!(
                "custom memory provider must implement {method}()"
            )));
        }
        let result = call(py, obj).map_err(|e| provider_error(e.to_string()))?;
        let awaitable = crate::async_bridge::is_awaitable(&result)
            .map_err(|e| provider_error(e.to_string()))?;
        Ok((result.unbind(), awaitable))
    })?;

    crate::async_bridge::resolve_maybe_awaitable(result_obj, is_awaitable, None)
        .await
        .map_err(|e| provider_error(e.to_string()))
}

impl PyInjectedMemory {
    fn refresh_size(&self) -> Result<(), LLMError> {
        let size = call_memory_size(&self.inner)?;
        self.size_cache.store(size, Ordering::Relaxed);
        Ok(())
    }
}

#[async_trait]
impl CoreMemoryProvider for PyInjectedMemory {
    async fn remember(&mut self, message: &ChatMessage) -> Result<(), LLMError> {
        let message = message.clone();
        call_memory_method_async(&self.inner, "remember", |py, obj| {
            let payload = serde_json::to_value(&message)
                .map_err(|e| PyRuntimeError::new_err(format!("serialize memory message: {e}")))?;
            let py_message = json_value_to_py(py, &payload)?;
            obj.call_method1("remember", (py_message.bind(py),))
        })
        .await?;
        self.refresh_size()?;
        Ok(())
    }

    async fn recall(
        &self,
        query: &str,
        limit: Option<usize>,
    ) -> Result<Vec<ChatMessage>, LLMError> {
        let result = call_memory_method_async(&self.inner, "recall", |_py, obj| {
            obj.call_method1("recall", (query, limit))
        })
        .await?;

        Python::attach(|py| parse_chat_messages_from_py(result.bind(py)))
    }

    async fn clear(&mut self) -> Result<(), LLMError> {
        call_memory_method_async(&self.inner, "clear", |_py, obj| obj.call_method0("clear"))
            .await?;
        self.refresh_size()?;
        Ok(())
    }

    fn memory_type(&self) -> MemoryType {
        MemoryType::Custom
    }

    fn size(&self) -> usize {
        self.size_cache.load(Ordering::Relaxed)
    }

    fn clone_box(&self) -> Box<dyn CoreMemoryProvider> {
        Box::new(self.clone())
    }

    fn export(&self) -> Vec<ChatMessage> {
        Vec::new()
    }
}

#[pyfunction]
pub fn sliding_window_memory(window_size: usize) -> PyMemoryProvider {
    PyMemoryProvider::new(
        Box::new(SlidingWindowMemory::new(window_size)),
        format!("SlidingWindowMemory(window_size={window_size})"),
    )
}

#[pyfunction]
pub fn memory_provider_from_impl(memory: &Bound<'_, PyAny>) -> PyResult<PyMemoryProvider> {
    let memory_impl = memory.clone().unbind();
    let size =
        call_memory_size(&memory_impl).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let repr = if let Ok(text) = memory.repr() {
        text.to_string()
    } else {
        "MemoryProvider(<custom>)".to_string()
    };

    Ok(PyMemoryProvider::new(
        Box::new(PyInjectedMemory {
            inner: memory_impl,
            size_cache: Arc::new(AtomicUsize::new(size)),
        }),
        repr,
    ))
}
