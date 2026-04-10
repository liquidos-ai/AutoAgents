use autoagents_llm::chat::ReasoningEffort;
use autoagents_llm::{HasConfig, LLMProvider};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use std::ffi::CString;
use std::sync::Arc;

/// Opaque wrapper around a built LLM provider. Passed to `PyAgentBuilder.llm()`.
#[pyclass(
    module = "autoagents_py._autoagents_py",
    name = "LLMProvider",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyLLMProvider {
    pub inner: Arc<dyn LLMProvider>,
}

impl PyLLMProvider {
    pub fn new(inner: Arc<dyn LLMProvider>) -> Self {
        Self { inner }
    }
}

const LLM_PROVIDER_CAPSULE_NAME: &str = "autoagents_py._autoagents_py.LLMProvider";

fn llm_provider_capsule_name() -> PyResult<CString> {
    CString::new(LLM_PROVIDER_CAPSULE_NAME)
        .map_err(|_| PyRuntimeError::new_err("invalid llm provider capsule name"))
}

fn llm_provider_to_capsule<'py>(
    py: Python<'py>,
    inner: Arc<dyn LLMProvider>,
) -> PyResult<Bound<'py, PyCapsule>> {
    let name = llm_provider_capsule_name()?;
    PyCapsule::new(py, inner, Some(name))
}

fn llm_provider_from_capsule(capsule: &Bound<'_, PyCapsule>) -> PyResult<Arc<dyn LLMProvider>> {
    let name = llm_provider_capsule_name()?;
    let pointer = capsule
        .pointer_checked(Some(name.as_c_str()))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let provider = unsafe { pointer.cast::<Arc<dyn LLMProvider>>().as_ref() };
    Ok(Arc::clone(provider))
}

fn llm_provider_type_error() -> PyErr {
    PyRuntimeError::new_err(
        "expected an AutoAgents LLMProvider returned by LLMBuilder.build() or PipelineBuilder.build()",
    )
}

#[pyfunction]
pub fn _llm_provider_from_capsule(capsule: &Bound<'_, PyCapsule>) -> PyResult<PyLLMProvider> {
    Ok(PyLLMProvider::new(llm_provider_from_capsule(capsule)?))
}

pub fn extract_llm_provider(provider: &Bound<'_, PyAny>) -> PyResult<Arc<dyn LLMProvider>> {
    if let Ok(provider) = provider.extract::<PyRef<'_, PyLLMProvider>>() {
        return Ok(Arc::clone(&provider.inner));
    }

    let capsule = provider
        .call_method0("_autoagents_llm_provider_capsule")
        .map_err(|_| llm_provider_type_error())?
        .cast_into::<PyCapsule>()
        .map_err(|_| llm_provider_type_error())?;

    llm_provider_from_capsule(&capsule).map_err(|_| llm_provider_type_error())
}

pub fn canonicalize_llm_provider(
    py: Python<'_>,
    inner: Arc<dyn LLMProvider>,
) -> PyResult<Py<PyAny>> {
    let capsule = llm_provider_to_capsule(py, inner)?;
    let module = py.import("autoagents_py._autoagents_py")?;
    let provider = module
        .getattr("_llm_provider_from_capsule")?
        .call1((capsule,))?;
    Ok(provider.unbind())
}

#[pymethods]
impl PyLLMProvider {
    fn __repr__(&self) -> &str {
        "LLMProvider(<built>)"
    }

    fn _autoagents_llm_provider_capsule<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        llm_provider_to_capsule(py, Arc::clone(&self.inner))
    }
}

/// Builder for constructing an LLM provider from Python.
///
/// Usage:
/// ```python
/// llm = PyLLMBuilder("openai").api_key("sk-...").model("gpt-4o").build()
/// ```
#[pyclass(name = "LLMBuilder")]
pub struct PyLLMBuilder {
    backend: String,
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    timeout_seconds: Option<u64>,
    reasoning: Option<bool>,
    reasoning_effort: Option<String>,
    reasoning_budget_tokens: Option<u32>,
    normalize_response: Option<bool>,
    enable_parallel_tool_use: Option<bool>,
    api_version: Option<String>,
    deployment_id: Option<String>,
    extra_body: Option<serde_json::Value>,
}

#[pymethods]
impl PyLLMBuilder {
    #[new]
    pub fn new(backend: String) -> Self {
        Self {
            backend,
            api_key: None,
            base_url: None,
            model: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            timeout_seconds: None,
            reasoning: None,
            reasoning_effort: None,
            reasoning_budget_tokens: None,
            normalize_response: None,
            enable_parallel_tool_use: None,
            api_version: None,
            deployment_id: None,
            extra_body: None,
        }
    }

    pub fn api_key(mut slf: PyRefMut<'_, Self>, key: String) -> PyRefMut<'_, Self> {
        slf.api_key = Some(key);
        slf
    }

    pub fn base_url(mut slf: PyRefMut<'_, Self>, url: String) -> PyRefMut<'_, Self> {
        slf.base_url = Some(url);
        slf
    }

    pub fn model(mut slf: PyRefMut<'_, Self>, model: String) -> PyRefMut<'_, Self> {
        slf.model = Some(model);
        slf
    }

    pub fn max_tokens(mut slf: PyRefMut<'_, Self>, max_tokens: u32) -> PyRefMut<'_, Self> {
        slf.max_tokens = Some(max_tokens);
        slf
    }

    pub fn temperature(mut slf: PyRefMut<'_, Self>, temperature: f32) -> PyRefMut<'_, Self> {
        slf.temperature = Some(temperature);
        slf
    }

    pub fn top_p(mut slf: PyRefMut<'_, Self>, top_p: f32) -> PyRefMut<'_, Self> {
        slf.top_p = Some(top_p);
        slf
    }

    pub fn top_k(mut slf: PyRefMut<'_, Self>, top_k: u32) -> PyRefMut<'_, Self> {
        slf.top_k = Some(top_k);
        slf
    }

    pub fn timeout_seconds(mut slf: PyRefMut<'_, Self>, secs: u64) -> PyRefMut<'_, Self> {
        slf.timeout_seconds = Some(secs);
        slf
    }

    pub fn reasoning(mut slf: PyRefMut<'_, Self>, enable: bool) -> PyRefMut<'_, Self> {
        slf.reasoning = Some(enable);
        slf
    }

    pub fn reasoning_effort(mut slf: PyRefMut<'_, Self>, effort: String) -> PyRefMut<'_, Self> {
        slf.reasoning_effort = Some(effort);
        slf
    }

    pub fn reasoning_budget_tokens(mut slf: PyRefMut<'_, Self>, tokens: u32) -> PyRefMut<'_, Self> {
        slf.reasoning_budget_tokens = Some(tokens);
        slf
    }

    pub fn normalize_response(mut slf: PyRefMut<'_, Self>, value: bool) -> PyRefMut<'_, Self> {
        slf.normalize_response = Some(value);
        slf
    }

    pub fn enable_parallel_tool_use(
        mut slf: PyRefMut<'_, Self>,
        value: bool,
    ) -> PyRefMut<'_, Self> {
        slf.enable_parallel_tool_use = Some(value);
        slf
    }

    pub fn api_version(mut slf: PyRefMut<'_, Self>, value: String) -> PyRefMut<'_, Self> {
        slf.api_version = Some(value);
        slf
    }

    pub fn deployment_id(mut slf: PyRefMut<'_, Self>, value: String) -> PyRefMut<'_, Self> {
        slf.deployment_id = Some(value);
        slf
    }

    pub fn extra_body_json(
        mut slf: PyRefMut<'_, Self>,
        json_body: String,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let parsed = serde_json::from_str::<serde_json::Value>(&json_body)
            .map_err(|e| PyValueError::new_err(format!("invalid extra_body_json: {e}")))?;
        slf.extra_body = Some(parsed);
        Ok(slf)
    }

    /// Build and return a `LLMProvider`. Synchronous — no async needed.
    pub fn build(&self) -> PyResult<PyLLMProvider> {
        let provider = build_provider(self)?;
        Ok(PyLLMProvider::new(provider))
    }
}

fn parse_reasoning_effort(value: &str) -> Option<ReasoningEffort> {
    match value.to_ascii_lowercase().as_str() {
        "low" => Some(ReasoningEffort::Low),
        "medium" => Some(ReasoningEffort::Medium),
        "high" => Some(ReasoningEffort::High),
        _ => None,
    }
}

fn apply_common_builder_fields<L>(
    mut builder: autoagents_llm::builder::LLMBuilder<L>,
    b: &PyLLMBuilder,
) -> autoagents_llm::builder::LLMBuilder<L>
where
    L: LLMProvider + HasConfig,
{
    if let Some(k) = &b.api_key {
        builder = builder.api_key(k);
    }
    if let Some(u) = &b.base_url {
        builder = builder.base_url(u);
    }
    if let Some(m) = &b.model {
        builder = builder.model(m);
    }
    if let Some(t) = b.max_tokens {
        builder = builder.max_tokens(t);
    }
    if let Some(t) = b.temperature {
        builder = builder.temperature(t);
    }
    if let Some(p) = b.top_p {
        builder = builder.top_p(p);
    }
    if let Some(k) = b.top_k {
        builder = builder.top_k(k);
    }
    if let Some(s) = b.timeout_seconds {
        builder = builder.timeout_seconds(s);
    }
    if let Some(r) = b.reasoning {
        builder = builder.reasoning(r);
    }
    if let Some(effort) = b
        .reasoning_effort
        .as_deref()
        .and_then(parse_reasoning_effort)
    {
        builder = builder.reasoning_effort(effort);
    }
    if let Some(tokens) = b.reasoning_budget_tokens {
        builder = builder.reasoning_budget_tokens(tokens);
    }
    if let Some(v) = b.normalize_response {
        builder = builder.normalize_response(v);
    }
    if let Some(v) = b.enable_parallel_tool_use {
        builder = builder.enable_parallel_tool_use(v);
    }
    if let Some(v) = &b.api_version {
        builder = builder.api_version(v);
    }
    if let Some(v) = &b.deployment_id {
        builder = builder.deployment_id(v);
    }
    if let Some(v) = &b.extra_body {
        builder = builder.extra_body(v);
    }
    builder
}

/// Dispatch to the right LLM backend based on the backend string.
fn build_provider(b: &PyLLMBuilder) -> PyResult<Arc<dyn LLMProvider>> {
    match b.backend.to_lowercase().as_str() {
        "openai" => {
            use autoagents_llm::backends::openai::OpenAI;
            use autoagents_llm::builder::LLMBuilder;
            let builder = apply_common_builder_fields(LLMBuilder::<OpenAI>::new(), b);
            builder
                .build()
                .map(|p| p as Arc<dyn LLMProvider>)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        }

        "anthropic" => {
            use autoagents_llm::backends::anthropic::Anthropic;
            use autoagents_llm::builder::LLMBuilder;
            let builder = apply_common_builder_fields(LLMBuilder::<Anthropic>::new(), b);
            builder
                .build()
                .map(|p| p as Arc<dyn LLMProvider>)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        }

        "ollama" => {
            use autoagents_llm::backends::ollama::Ollama;
            use autoagents_llm::builder::LLMBuilder;
            let builder = apply_common_builder_fields(LLMBuilder::<Ollama>::new(), b);
            builder
                .build()
                .map(|p| p as Arc<dyn LLMProvider>)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        }

        "deepseek" => {
            use autoagents_llm::backends::deepseek::DeepSeek;
            use autoagents_llm::builder::LLMBuilder;
            let builder = apply_common_builder_fields(LLMBuilder::<DeepSeek>::new(), b);
            builder
                .build()
                .map(|p| p as Arc<dyn LLMProvider>)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        }

        "google" => {
            use autoagents_llm::backends::google::Google;
            use autoagents_llm::builder::LLMBuilder;
            let builder = apply_common_builder_fields(LLMBuilder::<Google>::new(), b);
            builder
                .build()
                .map(|p| p as Arc<dyn LLMProvider>)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        }

        "groq" => {
            use autoagents_llm::backends::groq::Groq;
            use autoagents_llm::builder::LLMBuilder;
            let builder = apply_common_builder_fields(LLMBuilder::<Groq>::new(), b);
            builder
                .build()
                .map(|p| p as Arc<dyn LLMProvider>)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        }

        "xai" => {
            use autoagents_llm::backends::xai::XAI;
            use autoagents_llm::builder::LLMBuilder;
            let builder = apply_common_builder_fields(LLMBuilder::<XAI>::new(), b);
            builder
                .build()
                .map(|p| p as Arc<dyn LLMProvider>)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        }

        "openrouter" => {
            use autoagents_llm::backends::openrouter::OpenRouter;
            use autoagents_llm::builder::LLMBuilder;
            let builder = apply_common_builder_fields(LLMBuilder::<OpenRouter>::new(), b);
            builder
                .build()
                .map(|p| p as Arc<dyn LLMProvider>)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        }

        "azure_openai" => {
            use autoagents_llm::backends::azure_openai::AzureOpenAI;
            use autoagents_llm::builder::LLMBuilder;
            let builder = apply_common_builder_fields(LLMBuilder::<AzureOpenAI>::new(), b);
            builder
                .build()
                .map(|p| p as Arc<dyn LLMProvider>)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        }

        "phind" => {
            use autoagents_llm::backends::phind::Phind;
            use autoagents_llm::builder::LLMBuilder;
            let builder = apply_common_builder_fields(LLMBuilder::<Phind>::new(), b);
            builder
                .build()
                .map(|p| p as Arc<dyn LLMProvider>)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        }

        "minimax" => {
            use autoagents_llm::backends::minimax::MiniMax;
            use autoagents_llm::builder::LLMBuilder;
            let builder = apply_common_builder_fields(LLMBuilder::<MiniMax>::new(), b);
            builder
                .build()
                .map(|p| p as Arc<dyn LLMProvider>)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        }

        name => Err(PyRuntimeError::new_err(format!(
            "Unknown LLM backend: '{name}'. Supported backends: \
                 openai, anthropic, ollama, deepseek, google, groq, xai, \
                 openrouter, azure_openai, phind, minimax"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoagents_llm::chat::{
        ChatMessage, ChatProvider, ChatResponse, StructuredOutputFormat, Tool,
    };
    use autoagents_llm::completion::{CompletionProvider, CompletionRequest, CompletionResponse};
    use autoagents_llm::embedding::EmbeddingProvider;
    use autoagents_llm::error::LLMError;
    use autoagents_llm::models::ModelsProvider;
    use autoagents_llm::{HasConfig, NoConfig, ToolCall, async_trait};
    use std::fmt;

    fn init_python() {
        Python::initialize();
    }

    struct MockLLMProvider;

    #[async_trait]
    impl ChatProvider for MockLLMProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            Ok(Box::new(MockChatResponse("mock".to_string())))
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
                text: "mock".to_string(),
            })
        }
    }

    #[async_trait]
    impl EmbeddingProvider for MockLLMProvider {
        async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
            Ok(input.into_iter().map(|_| vec![1.0, 2.0]).collect())
        }
    }

    #[async_trait]
    impl ModelsProvider for MockLLMProvider {}

    impl LLMProvider for MockLLMProvider {}

    impl HasConfig for MockLLMProvider {
        type Config = NoConfig;
    }

    #[derive(Debug)]
    struct MockChatResponse(String);

    impl fmt::Display for MockChatResponse {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(&self.0)
        }
    }

    impl ChatResponse for MockChatResponse {
        fn text(&self) -> Option<String> {
            Some(self.0.clone())
        }

        fn tool_calls(&self) -> Option<Vec<ToolCall>> {
            None
        }
    }

    #[test]
    fn builder_setters_store_configuration() {
        init_python();
        Python::attach(|py| {
            let builder = Py::new(py, PyLLMBuilder::new("openai".to_string()))
                .expect("builder should create");

            {
                let mut builder_ref = builder.borrow_mut(py);
                builder_ref = PyLLMBuilder::api_key(builder_ref, "sk-test".to_string());
                builder_ref =
                    PyLLMBuilder::base_url(builder_ref, "https://example.test".to_string());
                builder_ref = PyLLMBuilder::model(builder_ref, "gpt-4.1-mini".to_string());
                builder_ref = PyLLMBuilder::max_tokens(builder_ref, 128);
                builder_ref = PyLLMBuilder::temperature(builder_ref, 0.5);
                builder_ref = PyLLMBuilder::top_p(builder_ref, 0.8);
                builder_ref = PyLLMBuilder::top_k(builder_ref, 16);
                builder_ref = PyLLMBuilder::timeout_seconds(builder_ref, 45);
                builder_ref = PyLLMBuilder::reasoning(builder_ref, true);
                builder_ref = PyLLMBuilder::reasoning_effort(builder_ref, "high".to_string());
                builder_ref = PyLLMBuilder::reasoning_budget_tokens(builder_ref, 256);
                builder_ref = PyLLMBuilder::normalize_response(builder_ref, false);
                builder_ref = PyLLMBuilder::enable_parallel_tool_use(builder_ref, true);
                builder_ref = PyLLMBuilder::api_version(builder_ref, "2025-01-01".to_string());
                builder_ref = PyLLMBuilder::deployment_id(builder_ref, "dep-123".to_string());
                builder_ref =
                    PyLLMBuilder::extra_body_json(builder_ref, "{\"seed\":7}".to_string())
                        .expect("extra body should parse");
                assert_eq!(builder_ref.backend, "openai");
            }

            let builder_ref = builder.borrow(py);
            assert_eq!(builder_ref.api_key.as_deref(), Some("sk-test"));
            assert_eq!(
                builder_ref.base_url.as_deref(),
                Some("https://example.test")
            );
            assert_eq!(builder_ref.model.as_deref(), Some("gpt-4.1-mini"));
            assert_eq!(builder_ref.max_tokens, Some(128));
            assert_eq!(builder_ref.temperature, Some(0.5));
            assert_eq!(builder_ref.top_p, Some(0.8));
            assert_eq!(builder_ref.top_k, Some(16));
            assert_eq!(builder_ref.timeout_seconds, Some(45));
            assert_eq!(builder_ref.reasoning, Some(true));
            assert_eq!(builder_ref.reasoning_effort.as_deref(), Some("high"));
            assert_eq!(builder_ref.reasoning_budget_tokens, Some(256));
            assert_eq!(builder_ref.normalize_response, Some(false));
            assert_eq!(builder_ref.enable_parallel_tool_use, Some(true));
            assert_eq!(builder_ref.api_version.as_deref(), Some("2025-01-01"));
            assert_eq!(builder_ref.deployment_id.as_deref(), Some("dep-123"));
            assert_eq!(builder_ref.extra_body, Some(serde_json::json!({"seed": 7})));
        });
    }

    #[test]
    fn parse_reasoning_effort_and_extra_body_validate_inputs() {
        assert!(matches!(
            parse_reasoning_effort("LOW"),
            Some(ReasoningEffort::Low)
        ));
        assert!(matches!(
            parse_reasoning_effort("medium"),
            Some(ReasoningEffort::Medium)
        ));
        assert!(matches!(
            parse_reasoning_effort("High"),
            Some(ReasoningEffort::High)
        ));
        assert!(parse_reasoning_effort("unsupported").is_none());

        init_python();
        Python::attach(|py| {
            let builder = Py::new(py, PyLLMBuilder::new("openai".to_string()))
                .expect("builder should create");
            let err = match PyLLMBuilder::extra_body_json(
                builder.borrow_mut(py),
                "{not-json}".to_string(),
            ) {
                Ok(_) => panic!("invalid json should fail"),
                Err(err) => err,
            };
            assert!(err.to_string().contains("invalid extra_body_json"));
        });
    }

    #[test]
    fn provider_capsules_round_trip_and_extract() {
        init_python();
        Python::attach(|py| {
            let provider = Arc::new(MockLLMProvider) as Arc<dyn LLMProvider>;
            let capsule =
                llm_provider_to_capsule(py, Arc::clone(&provider)).expect("capsule should build");
            let extracted = llm_provider_from_capsule(&capsule).expect("capsule should unwrap");
            assert!(Arc::ptr_eq(&provider, &extracted));

            let wrapper = PyLLMProvider::new(Arc::clone(&provider));
            let wrapper_any = Py::new(py, wrapper)
                .expect("wrapper should create")
                .into_any();
            let extracted =
                extract_llm_provider(wrapper_any.bind(py)).expect("wrapper should extract");
            assert!(Arc::ptr_eq(&provider, &extracted));

            let module =
                PyModule::new(py, "provider_capsule_module").expect("module should create");
            module
                .add_class::<PyLLMProvider>()
                .expect("class should register");
            let helper = module.getattr("LLMProvider").expect("class should exist");
            assert!(helper.is_truthy().expect("class should be truthy"));
        });
    }

    #[test]
    fn capsule_helpers_validate_foreign_inputs_and_unknown_backends() {
        init_python();
        Python::attach(|py| {
            let module = PyModule::new(py, "foreign_capsule_mod").expect("module should create");
            let capsule = PyCapsule::new(py, 42_u32, None).expect("foreign capsule should build");
            let err = match _llm_provider_from_capsule(&capsule) {
                Ok(_) => panic!("foreign capsule should fail"),
                Err(err) => err,
            };
            assert!(err.is_instance_of::<PyRuntimeError>(py));

            module
                .add("bogus", "value")
                .and_then(|_| module.getattr("bogus").map(|_| ()))
                .expect("module attribute should register");
            let bogus = module.getattr("bogus").expect("bogus attr should exist");
            let err = match extract_llm_provider(&bogus) {
                Ok(_) => panic!("plain string should fail"),
                Err(err) => err,
            };
            assert!(
                err.to_string()
                    .contains("expected an AutoAgents LLMProvider")
            );

            let err = match build_provider(&PyLLMBuilder::new("unknown-backend".to_string())) {
                Ok(_) => panic!("unknown backend should fail"),
                Err(err) => err,
            };
            assert!(err.to_string().contains("Unknown LLM backend"));
        });
    }
}
