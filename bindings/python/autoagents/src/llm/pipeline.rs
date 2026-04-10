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
        None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use autoagents_llm::chat::{
        ChatRole, CompletionTokensDetails, FunctionTool, MessageType, PromptTokensDetails,
        StreamChoice, StreamChunk, StreamDelta, StreamResponse,
    };
    use autoagents_llm::completion::{CompletionProvider, CompletionRequest, CompletionResponse};
    use autoagents_llm::embedding::EmbeddingProvider;
    use autoagents_llm::models::ModelsProvider;
    use autoagents_llm::{FunctionCall, ToolCall};
    use futures::{StreamExt, stream};
    use pyo3::types::{PyList, PyModule};
    use std::ffi::CString;

    fn init_runtime_bridge() {
        Python::initialize();
        let runtime = crate::runtime::get_runtime().expect("shared runtime should initialize");
        let _ = pyo3_async_runtimes::tokio::init_with_runtime(runtime);
    }

    fn module_from_code<'py>(
        py: Python<'py>,
        code: &str,
        filename: &str,
        module_name: &str,
    ) -> PyResult<Bound<'py, PyModule>> {
        PyModule::from_code(
            py,
            &CString::new(code).expect("python source should be valid CString"),
            &CString::new(filename).expect("filename should be a valid CString"),
            &CString::new(module_name).expect("module name should be a valid CString"),
        )
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
            let tool_calls = tools.and_then(|items| {
                items.first().map(|tool| {
                    vec![ToolCall {
                        id: "call_fallback".to_string(),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: tool.function.name.clone(),
                            arguments: "{\"city\":\"blr\"}".to_string(),
                        },
                    }]
                })
            });

            Ok(Box::new(MockChatResponse {
                text: Some(format!("fallback:{}", messages[0].content)),
                tool_calls,
                thinking: Some("fallback-thinking".to_string()),
                usage: Some(sample_usage()),
            }))
        }

        async fn chat_with_web_search(
            &self,
            input: String,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            Ok(Box::new(MockChatResponse {
                text: Some(format!("fallback-web:{input}")),
                tool_calls: None,
                thinking: None,
                usage: None,
            }))
        }

        async fn chat_stream(
            &self,
            _messages: &[ChatMessage],
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<
            std::pin::Pin<Box<dyn futures::Stream<Item = Result<String, LLMError>> + Send>>,
            LLMError,
        > {
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
            std::pin::Pin<Box<dyn futures::Stream<Item = Result<StreamResponse, LLMError>> + Send>>,
            LLMError,
        > {
            Ok(Box::pin(stream::iter(vec![Ok(StreamResponse {
                choices: vec![StreamChoice {
                    delta: StreamDelta {
                        content: Some("structured".to_string()),
                        reasoning_content: Some("reason".to_string()),
                        tool_calls: None,
                    },
                }],
                usage: Some(sample_usage()),
            })])))
        }

        async fn chat_stream_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<
            std::pin::Pin<Box<dyn futures::Stream<Item = Result<StreamChunk, LLMError>> + Send>>,
            LLMError,
        > {
            Ok(Box::pin(stream::iter(vec![
                Ok(StreamChunk::Text("tool-alpha".to_string())),
                Ok(StreamChunk::Done {
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

    fn sample_usage() -> Usage {
        Usage {
            prompt_tokens: 3,
            completion_tokens: 5,
            total_tokens: 8,
            completion_tokens_details: Some(CompletionTokensDetails {
                reasoning_tokens: Some(2),
                audio_tokens: None,
            }),
            prompt_tokens_details: Some(PromptTokensDetails {
                cached_tokens: Some(1),
                audio_tokens: None,
            }),
        }
    }

    fn sample_message() -> ChatMessage {
        ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "hello".to_string(),
        }
    }

    fn sample_tool() -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: FunctionTool {
                name: "lookup".to_string(),
                description: "Look up details".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" }
                    },
                    "required": ["city"]
                }),
            },
        }
    }

    fn sample_schema() -> StructuredOutputFormat {
        StructuredOutputFormat {
            name: "Summary".to_string(),
            description: Some("Return a summary".to_string()),
            schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "summary": { "type": "string" }
                },
                "required": ["summary"]
            })),
            strict: Some(true),
        }
    }

    #[test]
    fn parse_layer_chat_response_and_py_converters_cover_branches() {
        init_runtime_bridge();
        Python::attach(|py| {
            let valid = serde_json::json!({
                "text": "layered",
                "thinking": "trace",
                "tool_calls": [serde_json::to_value(ToolCall {
                    id: "call_1".to_string(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: "lookup".to_string(),
                        arguments: "{\"city\":\"blr\"}".to_string(),
                    },
                })
                .expect("tool call should serialize")],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                    "completion_tokens_details": null,
                    "prompt_tokens_details": null
                }
            });
            let valid_py = crate::convert::json_value_to_py(py, &valid)
                .expect("valid json should convert to Python");
            let parsed = parse_layer_chat_response(valid_py.bind(py))
                .expect("valid layer response should parse");
            assert_eq!(parsed.text.as_deref(), Some("layered"));
            assert_eq!(parsed.thinking.as_deref(), Some("trace"));
            assert_eq!(
                parsed.tool_calls.as_ref().expect("tool calls should parse")[0]
                    .function
                    .name,
                "lookup"
            );
            assert_eq!(parsed.usage.expect("usage should parse").total_tokens, 3);

            let invalid_obj = PyList::empty(py).into_any().unbind();
            assert!(
                parse_layer_chat_response(invalid_obj.bind(py))
                    .expect_err("lists should be rejected")
                    .to_string()
                    .contains("layer chat response must be a dict")
            );

            let invalid_usage = crate::convert::json_value_to_py(
                py,
                &serde_json::json!({"text": "oops", "usage": {"prompt_tokens": "bad"}}),
            )
            .expect("invalid usage json should still convert to Python");
            assert!(
                parse_layer_chat_response(invalid_usage.bind(py))
                    .expect_err("invalid usage should be rejected")
                    .to_string()
                    .contains("invalid layer usage")
            );

            let messages_py =
                messages_to_py(py, &[sample_message()]).expect("messages should serialize");
            let messages_json = crate::convert::py_any_to_json_value(messages_py.bind(py))
                .expect("messages python value should round-trip");
            assert_eq!(messages_json[0]["content"], serde_json::json!("hello"));

            let none_schema = schema_to_py(py, None).expect("none schema should convert");
            assert!(none_schema.bind(py).is_none());

            let schema_py =
                schema_to_py(py, Some(sample_schema())).expect("schema should serialize");
            let schema_json = crate::convert::py_any_to_json_value(schema_py.bind(py))
                .expect("schema python value should round-trip");
            assert_eq!(schema_json["name"], serde_json::json!("Summary"));

            let tool_py = tool_to_py(py, &sample_tool()).expect("tool should convert");
            let repr = tool_py
                .bind(py)
                .repr()
                .expect("tool repr should be available")
                .to_string();
            assert!(repr.contains("Tool(name='lookup')"));

            let tools_py =
                tools_to_py(py, Some(&[sample_tool()])).expect("tool list should convert");
            let tools_list = tools_py
                .bind(py)
                .cast::<PyList>()
                .expect("tool list should be a Python list");
            assert_eq!(tools_list.len(), 1);

            let provider = Py::new(py, PyLLMProvider::new(Arc::new(MockLLMProvider)))
                .expect("provider should be allocated");
            let provider_any = provider.bind(py).as_any();
            assert!(extract_provider(provider_any).is_ok());

            let none = py.None();
            match extract_provider(none.bind(py)) {
                Ok(_) => panic!("plain Python objects should be rejected"),
                Err(err) => {
                    assert!(
                        err.to_string()
                            .contains("expected an AutoAgents LLMProvider")
                    );
                }
            }
        });
    }

    #[test]
    fn call_layer_method_async_handles_missing_sync_and_async_methods() {
        init_runtime_bridge();

        Python::attach(|py| -> PyResult<()> {
            let event_loop = py.import("asyncio")?.call_method0("new_event_loop")?;
            let layer = {
                let module = module_from_code(
                    py,
                    "class Layer:\n\
                     \tdef sync_value(self):\n\
                     \t\treturn {\"text\": \"sync\"}\n\
                     \n\
                     \tasync def async_value(self):\n\
                     \t\treturn {\"text\": \"async\"}\n",
                    "autoagents_py/tests/pipeline_layer_methods.py",
                    "autoagents_pipeline_layer_methods",
                )?;
                module.getattr("Layer")?.call0()?.unbind()
            };
            pyo3_async_runtimes::tokio::run_until_complete(event_loop, async move {
                let missing = call_layer_method_async(&layer, "missing", |_py, obj| {
                    obj.call_method0("missing")
                })
                .await
                .expect("missing methods should return Ok(None)");
                assert!(missing.is_none());

                let sync = call_layer_method_async(&layer, "sync_value", |_py, obj| {
                    obj.call_method0("sync_value")
                })
                .await
                .expect("sync method should resolve")
                .expect("sync method should return a value");
                Python::attach(|py| {
                    let json = crate::convert::py_any_to_json_value(sync.bind(py))
                        .expect("sync result should round-trip");
                    assert_eq!(json["text"], serde_json::json!("sync"));
                });

                let async_result = call_layer_method_async(&layer, "async_value", |_py, obj| {
                    obj.call_method0("async_value")
                })
                .await
                .expect("async method should resolve")
                .expect("async method should return a value");
                Python::attach(|py| {
                    let json = crate::convert::py_any_to_json_value(async_result.bind(py))
                        .expect("async result should round-trip");
                    assert_eq!(json["text"], serde_json::json!("async"));
                });
                Ok(())
            })
        })
        .expect("temporary Python event loop should run");
    }

    #[test]
    fn layered_provider_overrides_and_falls_back() {
        init_runtime_bridge();

        let fallback = Python::attach(|py| {
            let empty_layer = module_from_code(
                py,
                "class EmptyLayer:\n\
                 \tpass\n",
                "autoagents_py/tests/pipeline_empty_layer.py",
                "autoagents_pipeline_empty_layer",
            )
            .expect("python module should compile")
            .getattr("EmptyLayer")
            .expect("empty layer class should exist")
            .call0()
            .expect("empty layer should instantiate")
            .unbind();

            PyLayeredProvider {
                layer: empty_layer,
                next: Arc::new(MockLLMProvider),
            }
        });

        let override_provider = Python::attach(|py| {
            let module = module_from_code(
                py,
                "class OverrideLayer:\n\
                 \tdef chat(self, next_provider, messages, schema):\n\
                 \t\treturn {\n\
                 \t\t\t\"text\": \"override-chat\",\n\
                 \t\t\t\"thinking\": \"layer-thinking\",\n\
                 \t\t\t\"usage\": {\n\
                 \t\t\t\t\"prompt_tokens\": 1,\n\
                 \t\t\t\t\"completion_tokens\": 2,\n\
                 \t\t\t\t\"total_tokens\": 3,\n\
                 \t\t\t\t\"completion_tokens_details\": None,\n\
                 \t\t\t\t\"prompt_tokens_details\": None,\n\
                 \t\t\t},\n\
                 \t\t}\n\
                 \n\
                 \tdef chat_with_tools(self, next_provider, messages, tools, schema):\n\
                 \t\treturn {\n\
                 \t\t\t\"text\": \"override-tools\",\n\
                 \t\t\t\"tool_calls\": [{\n\
                 \t\t\t\t\"id\": \"call_override\",\n\
                 \t\t\t\t\"type\": \"function\",\n\
                 \t\t\t\t\"function\": {\n\
                 \t\t\t\t\t\"name\": \"lookup\",\n\
                 \t\t\t\t\t\"arguments\": \"{}\"\n\
                 \t\t\t\t}\n\
                 \t\t\t}],\n\
                 \t\t}\n\
                 \n\
                 \tdef chat_with_web_search(self, next_provider, input):\n\
                 \t\treturn {\"text\": f\"override-web:{input}\"}\n",
                "autoagents_py/tests/pipeline_override_layer.py",
                "autoagents_pipeline_override_layer",
            )
            .expect("python module should compile");

            PyLayeredProvider {
                layer: module
                    .getattr("OverrideLayer")
                    .expect("override layer class should exist")
                    .call0()
                    .expect("override layer should instantiate")
                    .unbind(),
                next: Arc::new(MockLLMProvider),
            }
        });

        let messages = vec![sample_message()];
        let tools = vec![sample_tool()];

        let fallback_chat = futures::executor::block_on(fallback.chat(&messages, None))
            .expect("fallback chat should succeed");
        assert_eq!(fallback_chat.text().as_deref(), Some("fallback:hello"));

        let fallback_web =
            futures::executor::block_on(fallback.chat_with_web_search("rust".to_string()))
                .expect("fallback web search should succeed");
        assert_eq!(fallback_web.text().as_deref(), Some("fallback-web:rust"));

        let mut text_stream = futures::executor::block_on(fallback.chat_stream(&messages, None))
            .expect("fallback stream should succeed");
        assert_eq!(
            futures::executor::block_on(text_stream.next())
                .expect("first item")
                .expect("stream item"),
            "alpha"
        );

        let mut struct_stream =
            futures::executor::block_on(fallback.chat_stream_struct(&messages, Some(&tools), None))
                .expect("fallback structured stream should succeed");
        let struct_item = futures::executor::block_on(struct_stream.next())
            .expect("structured item")
            .expect("structured stream item");
        assert_eq!(
            struct_item.choices[0].delta.content.as_deref(),
            Some("structured")
        );

        let mut tool_stream = futures::executor::block_on(fallback.chat_stream_with_tools(
            &messages,
            Some(&tools),
            None,
        ))
        .expect("fallback tool stream should succeed");
        match futures::executor::block_on(tool_stream.next())
            .expect("tool stream item")
            .expect("tool stream chunk")
        {
            StreamChunk::Text(text) => assert_eq!(text, "tool-alpha"),
            other => panic!("unexpected tool stream chunk: {other:?}"),
        }

        let override_chat =
            futures::executor::block_on(override_provider.chat(&messages, Some(sample_schema())))
                .expect("override chat should succeed");
        assert_eq!(override_chat.text().as_deref(), Some("override-chat"));
        assert_eq!(override_chat.thinking().as_deref(), Some("layer-thinking"));
        assert_eq!(
            override_chat
                .usage()
                .expect("usage should exist")
                .total_tokens,
            3
        );

        let override_tools = futures::executor::block_on(override_provider.chat_with_tools(
            &messages,
            Some(&tools),
            None,
        ))
        .expect("override tool chat should succeed");
        assert_eq!(override_tools.text().as_deref(), Some("override-tools"));
        assert_eq!(
            override_tools
                .tool_calls()
                .expect("tool calls should exist")[0]
                .function
                .name,
            "lookup"
        );

        let override_web =
            futures::executor::block_on(override_provider.chat_with_web_search("docs".to_string()))
                .expect("override web search should succeed");
        assert_eq!(override_web.text().as_deref(), Some("override-web:docs"));
    }

    #[test]
    fn pipeline_layer_builders_wrap_valid_providers_and_reject_invalid_inputs() {
        init_runtime_bridge();
        Python::attach(|py| {
            let provider = Py::new(py, PyLLMProvider::new(Arc::new(MockLLMProvider)))
                .expect("provider should be allocated");
            let layer = module_from_code(
                py,
                "class EmptyLayer:\n\
                 \tpass\n",
                "autoagents_py/tests/pipeline_builder_layer.py",
                "autoagents_pipeline_builder_layer",
            )
            .expect("python module should compile")
            .getattr("EmptyLayer")
            .expect("empty layer class should exist")
            .call0()
            .expect("empty layer should instantiate");

            let cache = pipeline_cache_layer(provider.bind(py).as_any(), Some(30))
                .expect("cache layer should wrap provider");
            let cache_obj = Py::new(py, cache).expect("cache layer should convert to Python");
            assert_eq!(
                cache_obj
                    .bind(py)
                    .repr()
                    .expect("cache repr should succeed")
                    .to_string(),
                "LLMProvider(<built>)"
            );

            let retry = pipeline_retry_layer(
                provider.bind(py).as_any(),
                Some(0),
                Some(25),
                Some(50),
                Some(false),
            )
            .expect("retry layer should wrap provider");
            let retry_obj = Py::new(py, retry).expect("retry layer should convert to Python");
            assert_eq!(
                retry_obj
                    .bind(py)
                    .repr()
                    .expect("retry repr should succeed")
                    .to_string(),
                "LLMProvider(<built>)"
            );

            let python_layer = pipeline_python_layer(provider.bind(py).as_any(), &layer.into_any())
                .expect("python layer should wrap provider");
            let python_layer_obj =
                Py::new(py, python_layer).expect("python layer should convert to Python");
            assert_eq!(
                python_layer_obj
                    .bind(py)
                    .repr()
                    .expect("python layer repr should succeed")
                    .to_string(),
                "LLMProvider(<built>)"
            );

            let none = py.None();
            match pipeline_cache_layer(none.bind(py), None) {
                Ok(_) => panic!("plain objects should be rejected"),
                Err(err) => {
                    assert!(
                        err.to_string()
                            .contains("expected an AutoAgents LLMProvider")
                    );
                }
            }
        });
    }
}
