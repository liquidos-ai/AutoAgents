use crate::agent::py_agent::PyExecutionLLM;
use crate::convert::{json_value_to_py, py_any_to_json_value};
use crate::llm::builder::{PyLLMProvider, extract_llm_provider};
use crate::tool::PyTool;
use async_trait::async_trait;
use autoagents_llm::chat::{
    ChatMessage, ChatProvider, ChatResponse, StructuredOutputFormat, Tool, Usage,
};
use autoagents_llm::completion::{CompletionProvider, CompletionRequest, CompletionResponse};
use autoagents_llm::embedding::EmbeddingProvider;
use autoagents_llm::error::LLMError;
use autoagents_llm::models::{ModelListRequest, ModelListResponse, ModelsProvider};
use autoagents_llm::pipeline::PipelineBuilder;
use autoagents_llm::{LLMProvider, ToolCall};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use serde_json::Value;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

fn provider_error(message: impl Into<String>) -> LLMError {
    LLMError::ProviderError(message.into())
}

fn extract_provider(provider: &Bound<'_, PyAny>) -> PyResult<Arc<dyn LLMProvider>> {
    extract_llm_provider(provider)
        .map_err(|_| {
            PyRuntimeError::new_err(
                "expected an AutoAgents LLMProvider returned by LLMBuilder.build() or PipelineBuilder.build()",
            )
        })
}

#[derive(Debug, Clone)]
struct LayerChatResponse {
    text: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
    thinking: Option<String>,
    usage: Option<Usage>,
}

impl ChatResponse for LayerChatResponse {
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

impl fmt::Display for LayerChatResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.text.as_deref().unwrap_or(""))
    }
}

fn parse_layer_chat_response(value: &Bound<'_, PyAny>) -> Result<LayerChatResponse, LLMError> {
    let value = py_any_to_json_value(value)
        .map_err(|e| provider_error(format!("invalid layer chat response: {e}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| provider_error("layer chat response must be a dict"))?;

    let text = object
        .get("text")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let thinking = object
        .get("thinking")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let tool_calls = match object.get("tool_calls") {
        Some(raw) if !raw.is_null() => Some(
            serde_json::from_value::<Vec<ToolCall>>(raw.clone())
                .map_err(|e| provider_error(format!("invalid layer tool_calls: {e}")))?,
        ),
        _ => None,
    };
    let usage = match object.get("usage") {
        Some(raw) if !raw.is_null() => Some(
            serde_json::from_value::<Usage>(raw.clone())
                .map_err(|e| provider_error(format!("invalid layer usage: {e}")))?,
        ),
        _ => None,
    };

    Ok(LayerChatResponse {
        text,
        tool_calls,
        thinking,
        usage,
    })
}

async fn call_layer_method_async<F>(
    layer: &Py<PyAny>,
    method: &str,
    call: F,
) -> Result<Option<Py<PyAny>>, LLMError>
where
    F: for<'py> FnOnce(Python<'py>, &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>>,
{
    let (result_obj, is_awaitable, missing_method) =
        Python::attach(|py| -> Result<(Py<PyAny>, bool, bool), LLMError> {
            let layer_obj = layer.bind(py);
            if !layer_obj
                .hasattr(method)
                .map_err(|e| provider_error(e.to_string()))?
            {
                return Ok((py.None(), false, true));
            }
            let result = call(py, layer_obj).map_err(|e| provider_error(e.to_string()))?;
            let awaitable = crate::async_bridge::is_awaitable(&result)
                .map_err(|e| provider_error(e.to_string()))?;
            Ok((result.unbind(), awaitable, false))
        })?;

    if missing_method {
        return Ok(None);
    }

    let resolved = crate::async_bridge::resolve_maybe_awaitable(result_obj, is_awaitable, None)
        .await
        .map_err(|e| provider_error(e.to_string()))?;
    Ok(Some(resolved))
}

fn messages_to_py(py: Python<'_>, messages: &[ChatMessage]) -> Result<Py<PyAny>, LLMError> {
    let value = serde_json::to_value(messages)
        .map_err(|e| provider_error(format!("serialize layer messages: {e}")))?;
    json_value_to_py(py, &value).map_err(|e| provider_error(e.to_string()))
}

fn schema_to_py(
    py: Python<'_>,
    schema: Option<StructuredOutputFormat>,
) -> Result<Py<PyAny>, LLMError> {
    match schema {
        Some(schema) => {
            let value = serde_json::to_value(schema)
                .map_err(|e| provider_error(format!("serialize layer schema: {e}")))?;
            json_value_to_py(py, &value).map_err(|e| provider_error(e.to_string()))
        }
        None => Ok(py.None()),
    }
}

fn tool_to_py(py: Python<'_>, tool: &Tool) -> Result<Py<PyAny>, LLMError> {
    let schema_json = serde_json::to_string(&tool.function.parameters)
        .map_err(|e| provider_error(format!("serialize layer tool schema: {e}")))?;
    let py_tool = PyTool::new(
        tool.function.name.clone(),
        tool.function.description.clone(),
        schema_json,
        py.None(),
    )
    .map_err(|e| provider_error(e.to_string()))?;
    let bound = Py::new(py, py_tool).map_err(|e| provider_error(e.to_string()))?;
    Ok(bound.into_any())
}

fn tools_to_py(py: Python<'_>, tools: Option<&[Tool]>) -> Result<Py<PyAny>, LLMError> {
    let Some(tools) = tools else {
        return Ok(py.None());
    };

    let list = PyList::empty(py);
    for tool in tools {
        list.append(tool_to_py(py, tool)?)
            .map_err(|e| provider_error(e.to_string()))?;
    }
    Ok(list.into_any().unbind())
}

struct PyLayeredProvider {
    layer: Py<PyAny>,
    next: Arc<dyn LLMProvider>,
}

impl Clone for PyLayeredProvider {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            layer: self.layer.clone_ref(py),
            next: Arc::clone(&self.next),
        })
    }
}

impl fmt::Debug for PyLayeredProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("PyLayeredProvider")
    }
}

#[async_trait]
impl ChatProvider for PyLayeredProvider {
    async fn chat(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        if let Some(result) = call_layer_method_async(&self.layer, "chat", |py, layer_obj| {
            let next = Py::new(py, PyExecutionLLM::new(Arc::clone(&self.next)))?;
            let messages =
                messages_to_py(py, messages).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let schema = schema_to_py(py, json_schema.clone())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            layer_obj.call_method1("chat", (next.bind(py), messages.bind(py), schema.bind(py)))
        })
        .await?
        {
            return Python::attach(|py| {
                parse_layer_chat_response(result.bind(py))
                    .map(|response| Box::new(response) as Box<dyn ChatResponse>)
            });
        }

        self.next.chat(messages, json_schema).await
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        if let Some(result) =
            call_layer_method_async(&self.layer, "chat_with_tools", |py, layer_obj| {
                let next = Py::new(py, PyExecutionLLM::new(Arc::clone(&self.next)))?;
                let messages = messages_to_py(py, messages)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let tools =
                    tools_to_py(py, tools).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let schema = schema_to_py(py, json_schema.clone())
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                layer_obj.call_method1(
                    "chat_with_tools",
                    (
                        next.bind(py),
                        messages.bind(py),
                        tools.bind(py),
                        schema.bind(py),
                    ),
                )
            })
            .await?
        {
            return Python::attach(|py| {
                parse_layer_chat_response(result.bind(py))
                    .map(|response| Box::new(response) as Box<dyn ChatResponse>)
            });
        }

        self.next
            .chat_with_tools(messages, tools, json_schema)
            .await
    }

    async fn chat_with_web_search(&self, input: String) -> Result<Box<dyn ChatResponse>, LLMError> {
        if let Some(result) =
            call_layer_method_async(&self.layer, "chat_with_web_search", |py, layer_obj| {
                let next = Py::new(py, PyExecutionLLM::new(Arc::clone(&self.next)))?;
                layer_obj.call_method1("chat_with_web_search", (next.bind(py), input.clone()))
            })
            .await?
        {
            return Python::attach(|py| {
                parse_layer_chat_response(result.bind(py))
                    .map(|response| Box::new(response) as Box<dyn ChatResponse>)
            });
        }

        self.next.chat_with_web_search(input).await
    }

    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<
        std::pin::Pin<Box<dyn futures::Stream<Item = Result<String, LLMError>> + Send>>,
        LLMError,
    > {
        self.next.chat_stream(messages, json_schema).await
    }

    async fn chat_stream_struct(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<
        std::pin::Pin<
            Box<
                dyn futures::Stream<Item = Result<autoagents_llm::chat::StreamResponse, LLMError>>
                    + Send,
            >,
        >,
        LLMError,
    > {
        self.next
            .chat_stream_struct(messages, tools, json_schema)
            .await
    }

    async fn chat_stream_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<
        std::pin::Pin<
            Box<
                dyn futures::Stream<Item = Result<autoagents_llm::chat::StreamChunk, LLMError>>
                    + Send,
            >,
        >,
        LLMError,
    > {
        self.next
            .chat_stream_with_tools(messages, tools, json_schema)
            .await
    }
}

#[async_trait]
impl CompletionProvider for PyLayeredProvider {
    async fn complete(
        &self,
        req: &CompletionRequest,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        self.next.complete(req, json_schema).await
    }
}

#[async_trait]
impl EmbeddingProvider for PyLayeredProvider {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        self.next.embed(input).await
    }
}

#[async_trait]
impl ModelsProvider for PyLayeredProvider {
    async fn list_models(
        &self,
        request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        self.next.list_models(request).await
    }
}

impl LLMProvider for PyLayeredProvider {}

#[pyfunction]
pub fn pipeline_cache_layer(
    provider: &Bound<'_, PyAny>,
    ttl_seconds: Option<u64>,
) -> PyResult<PyLLMProvider> {
    use autoagents_llm::optim::cache::{CacheConfig, CacheLayer};

    let provider = extract_provider(provider)?;
    let mut cfg = CacheConfig::default();
    if let Some(ttl) = ttl_seconds {
        cfg.ttl = Some(Duration::from_secs(ttl));
    }

    Ok(PyLLMProvider::new(
        PipelineBuilder::new(provider)
            .add_layer(CacheLayer::new(cfg))
            .build(),
    ))
}

#[pyfunction]
pub fn pipeline_retry_layer(
    provider: &Bound<'_, PyAny>,
    max_attempts: Option<u32>,
    initial_backoff_ms: Option<u64>,
    max_backoff_ms: Option<u64>,
    jitter: Option<bool>,
) -> PyResult<PyLLMProvider> {
    use autoagents_llm::optim::retry::{RetryConfig, RetryLayer};

    let provider = extract_provider(provider)?;
    let mut cfg = RetryConfig::default();
    if let Some(value) = max_attempts {
        cfg.max_attempts = value.max(1);
    }
    if let Some(value) = initial_backoff_ms {
        cfg.initial_backoff = Duration::from_millis(value);
    }
    if let Some(value) = max_backoff_ms {
        cfg.max_backoff = Duration::from_millis(value);
    }
    if let Some(value) = jitter {
        cfg.jitter = value;
    }

    Ok(PyLLMProvider::new(
        PipelineBuilder::new(provider)
            .add_layer(RetryLayer::new(cfg))
            .build(),
    ))
}

#[pyfunction]
pub fn pipeline_python_layer(
    provider: &Bound<'_, PyAny>,
    layer: &Bound<'_, PyAny>,
) -> PyResult<PyLLMProvider> {
    let provider = extract_provider(provider)?;
    Ok(PyLLMProvider::new(Arc::new(PyLayeredProvider {
        layer: layer.clone().unbind(),
        next: provider,
    })))
}
