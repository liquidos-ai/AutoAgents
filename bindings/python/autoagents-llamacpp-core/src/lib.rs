use _autoagents_py::llm::builder::canonicalize_llm_provider;
use autoagents_llamacpp::{LlamaCppProvider, ModelSource as LlamaModelSource};
use autoagents_llm::LLMProvider;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

#[pyclass(name = "LlamaCppBuilder", skip_from_py_object)]
#[derive(Clone, Default)]
pub struct PyLlamaCppBuilder {
    model_path: Option<String>,
    repo_id: Option<String>,
    hf_filename: Option<String>,
    mmproj_filename: Option<String>,
    hf_revision: Option<String>,
    model_dir: Option<String>,
    chat_template: Option<String>,
    force_json_grammar: Option<bool>,
    reasoning_format: Option<String>,
    extra_body: Option<serde_json::Value>,
    mmproj_use_gpu: Option<bool>,
    media_marker: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    repeat_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    repeat_last_n: Option<i32>,
    seed: Option<u32>,
    n_ctx: Option<u32>,
    n_batch: Option<u32>,
    n_ubatch: Option<u32>,
    n_threads: Option<i32>,
    n_threads_batch: Option<i32>,
    n_gpu_layers: Option<u32>,
    main_gpu: Option<i32>,
    split_mode: Option<String>,
    use_mlock: Option<bool>,
    devices: Option<Vec<usize>>,
    system_prompt: Option<String>,
}

#[pymethods]
impl PyLlamaCppBuilder {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn model_path(mut slf: PyRefMut<'_, Self>, path: String) -> PyRefMut<'_, Self> {
        slf.model_path = Some(path);
        slf.repo_id = None;
        slf
    }

    pub fn repo_id(mut slf: PyRefMut<'_, Self>, repo_id: String) -> PyRefMut<'_, Self> {
        slf.repo_id = Some(repo_id);
        slf
    }

    pub fn hf_filename(mut slf: PyRefMut<'_, Self>, filename: String) -> PyRefMut<'_, Self> {
        slf.hf_filename = Some(filename);
        slf
    }

    pub fn mmproj_filename(mut slf: PyRefMut<'_, Self>, filename: String) -> PyRefMut<'_, Self> {
        slf.mmproj_filename = Some(filename);
        slf
    }

    pub fn hf_revision(mut slf: PyRefMut<'_, Self>, revision: String) -> PyRefMut<'_, Self> {
        slf.hf_revision = Some(revision);
        slf
    }

    pub fn model_dir(mut slf: PyRefMut<'_, Self>, dir: String) -> PyRefMut<'_, Self> {
        slf.model_dir = Some(dir);
        slf
    }

    pub fn chat_template(mut slf: PyRefMut<'_, Self>, template: String) -> PyRefMut<'_, Self> {
        slf.chat_template = Some(template);
        slf
    }

    pub fn force_json_grammar(mut slf: PyRefMut<'_, Self>, force: bool) -> PyRefMut<'_, Self> {
        slf.force_json_grammar = Some(force);
        slf
    }

    pub fn reasoning_format(mut slf: PyRefMut<'_, Self>, format: String) -> PyRefMut<'_, Self> {
        slf.reasoning_format = Some(format);
        slf
    }

    pub fn extra_body_json(
        mut slf: PyRefMut<'_, Self>,
        body_json: String,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let parsed = serde_json::from_str::<serde_json::Value>(&body_json)
            .map_err(|e| PyRuntimeError::new_err(format!("invalid extra_body_json: {e}")))?;
        slf.extra_body = Some(parsed);
        Ok(slf)
    }

    pub fn mmproj_use_gpu(mut slf: PyRefMut<'_, Self>, value: bool) -> PyRefMut<'_, Self> {
        slf.mmproj_use_gpu = Some(value);
        slf
    }

    pub fn media_marker(mut slf: PyRefMut<'_, Self>, marker: String) -> PyRefMut<'_, Self> {
        slf.media_marker = Some(marker);
        slf
    }

    pub fn max_tokens(mut slf: PyRefMut<'_, Self>, tokens: u32) -> PyRefMut<'_, Self> {
        slf.max_tokens = Some(tokens);
        slf
    }

    pub fn temperature(mut slf: PyRefMut<'_, Self>, value: f32) -> PyRefMut<'_, Self> {
        slf.temperature = Some(value);
        slf
    }

    pub fn top_p(mut slf: PyRefMut<'_, Self>, value: f32) -> PyRefMut<'_, Self> {
        slf.top_p = Some(value);
        slf
    }

    pub fn top_k(mut slf: PyRefMut<'_, Self>, value: u32) -> PyRefMut<'_, Self> {
        slf.top_k = Some(value);
        slf
    }

    pub fn repeat_penalty(mut slf: PyRefMut<'_, Self>, value: f32) -> PyRefMut<'_, Self> {
        slf.repeat_penalty = Some(value);
        slf
    }

    pub fn frequency_penalty(mut slf: PyRefMut<'_, Self>, value: f32) -> PyRefMut<'_, Self> {
        slf.frequency_penalty = Some(value);
        slf
    }

    pub fn presence_penalty(mut slf: PyRefMut<'_, Self>, value: f32) -> PyRefMut<'_, Self> {
        slf.presence_penalty = Some(value);
        slf
    }

    pub fn repeat_last_n(mut slf: PyRefMut<'_, Self>, value: i32) -> PyRefMut<'_, Self> {
        slf.repeat_last_n = Some(value);
        slf
    }

    pub fn seed(mut slf: PyRefMut<'_, Self>, value: u32) -> PyRefMut<'_, Self> {
        slf.seed = Some(value);
        slf
    }

    pub fn n_ctx(mut slf: PyRefMut<'_, Self>, value: u32) -> PyRefMut<'_, Self> {
        slf.n_ctx = Some(value);
        slf
    }

    pub fn n_batch(mut slf: PyRefMut<'_, Self>, value: u32) -> PyRefMut<'_, Self> {
        slf.n_batch = Some(value);
        slf
    }

    pub fn n_ubatch(mut slf: PyRefMut<'_, Self>, value: u32) -> PyRefMut<'_, Self> {
        slf.n_ubatch = Some(value);
        slf
    }

    pub fn n_threads(mut slf: PyRefMut<'_, Self>, value: i32) -> PyRefMut<'_, Self> {
        slf.n_threads = Some(value);
        slf
    }

    pub fn n_threads_batch(mut slf: PyRefMut<'_, Self>, value: i32) -> PyRefMut<'_, Self> {
        slf.n_threads_batch = Some(value);
        slf
    }

    pub fn n_gpu_layers(mut slf: PyRefMut<'_, Self>, value: u32) -> PyRefMut<'_, Self> {
        slf.n_gpu_layers = Some(value);
        slf
    }

    pub fn main_gpu(mut slf: PyRefMut<'_, Self>, value: i32) -> PyRefMut<'_, Self> {
        slf.main_gpu = Some(value);
        slf
    }

    pub fn split_mode(mut slf: PyRefMut<'_, Self>, mode: String) -> PyRefMut<'_, Self> {
        slf.split_mode = Some(mode);
        slf
    }

    pub fn use_mlock(mut slf: PyRefMut<'_, Self>, value: bool) -> PyRefMut<'_, Self> {
        slf.use_mlock = Some(value);
        slf
    }

    pub fn devices(mut slf: PyRefMut<'_, Self>, values: Vec<usize>) -> PyRefMut<'_, Self> {
        slf.devices = Some(values);
        slf
    }

    pub fn system_prompt(mut slf: PyRefMut<'_, Self>, value: String) -> PyRefMut<'_, Self> {
        slf.system_prompt = Some(value);
        slf
    }

    pub fn build<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let config = self.clone();
        _autoagents_py::async_bridge::future_into_py(py, async move {
            let provider = build_provider(config).await?;
            Python::attach(|py| {
                canonicalize_llm_provider(py, Arc::new(provider) as Arc<dyn LLMProvider>)
            })
        })
    }
}

async fn build_provider(config: PyLlamaCppBuilder) -> PyResult<LlamaCppProvider> {
    let PyLlamaCppBuilder {
        model_path,
        repo_id,
        hf_filename,
        mmproj_filename,
        hf_revision,
        model_dir,
        chat_template,
        force_json_grammar,
        reasoning_format,
        extra_body,
        mmproj_use_gpu,
        media_marker,
        max_tokens,
        temperature,
        top_p,
        top_k,
        repeat_penalty,
        frequency_penalty,
        presence_penalty,
        repeat_last_n,
        seed,
        n_ctx,
        n_batch,
        n_ubatch,
        n_threads,
        n_threads_batch,
        n_gpu_layers,
        main_gpu,
        split_mode,
        use_mlock,
        devices,
        system_prompt,
    } = config;

    let behavior_options = BehaviorOptions {
        chat_template,
        force_json_grammar,
        reasoning_format,
        extra_body,
        model_dir,
        hf_revision,
        mmproj_use_gpu,
        media_marker,
        system_prompt,
    };
    let sampling_options = SamplingOptions {
        max_tokens,
        temperature,
        top_p,
        top_k,
        repeat_penalty,
        frequency_penalty,
        presence_penalty,
        repeat_last_n,
        seed,
    };
    let runtime_options = RuntimeOptions {
        n_ctx,
        n_batch,
        n_ubatch,
        n_threads,
        n_threads_batch,
        n_gpu_layers,
        main_gpu,
        split_mode,
        use_mlock,
        devices,
    };

    let model_source = parse_model_source(model_path, repo_id, hf_filename, mmproj_filename)?;
    let builder = LlamaCppProvider::builder().model_source(model_source);
    let builder = apply_behavior_options(builder, behavior_options)?;
    let builder = apply_sampling_options(builder, sampling_options);
    let builder = apply_runtime_options(builder, runtime_options)?;

    builder
        .build()
        .await
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

fn apply_behavior_options(
    mut builder: autoagents_llamacpp::LlamaCppProviderBuilder,
    options: BehaviorOptions,
) -> PyResult<autoagents_llamacpp::LlamaCppProviderBuilder> {
    if let Some(v) = options.chat_template {
        builder = builder.chat_template(v);
    }
    if let Some(v) = options.force_json_grammar {
        builder = builder.force_json_grammar(v);
    }
    if let Some(v) = options.reasoning_format {
        builder = builder.reasoning_format(parse_reasoning_format(&v)?);
    }
    if let Some(v) = options.extra_body {
        builder = builder.extra_body(v);
    }
    if let Some(v) = options.model_dir {
        builder = builder.model_dir(v);
    }
    if let Some(v) = options.hf_revision {
        builder = builder.hf_revision(v);
    }
    if let Some(v) = options.mmproj_use_gpu {
        builder = builder.mmproj_use_gpu(v);
    }
    if let Some(v) = options.media_marker {
        builder = builder.media_marker(v);
    }
    if let Some(v) = options.system_prompt {
        builder = builder.system_prompt(v);
    }
    Ok(builder)
}

fn apply_sampling_options(
    mut builder: autoagents_llamacpp::LlamaCppProviderBuilder,
    options: SamplingOptions,
) -> autoagents_llamacpp::LlamaCppProviderBuilder {
    if let Some(v) = options.max_tokens {
        builder = builder.max_tokens(v);
    }
    if let Some(v) = options.temperature {
        builder = builder.temperature(v);
    }
    if let Some(v) = options.top_p {
        builder = builder.top_p(v);
    }
    if let Some(v) = options.top_k {
        builder = builder.top_k(v);
    }
    if let Some(v) = options.repeat_penalty {
        builder = builder.repeat_penalty(v);
    }
    if let Some(v) = options.frequency_penalty {
        builder = builder.frequency_penalty(v);
    }
    if let Some(v) = options.presence_penalty {
        builder = builder.presence_penalty(v);
    }
    if let Some(v) = options.repeat_last_n {
        builder = builder.repeat_last_n(v);
    }
    if let Some(v) = options.seed {
        builder = builder.seed(v);
    }
    builder
}

fn apply_runtime_options(
    mut builder: autoagents_llamacpp::LlamaCppProviderBuilder,
    options: RuntimeOptions,
) -> PyResult<autoagents_llamacpp::LlamaCppProviderBuilder> {
    if let Some(v) = options.n_ctx {
        builder = builder.n_ctx(v);
    }
    if let Some(v) = options.n_batch {
        builder = builder.n_batch(v);
    }
    if let Some(v) = options.n_ubatch {
        builder = builder.n_ubatch(v);
    }
    if let Some(v) = options.n_threads {
        builder = builder.n_threads(v);
    }
    if let Some(v) = options.n_threads_batch {
        builder = builder.n_threads_batch(v);
    }
    if let Some(v) = options.n_gpu_layers {
        builder = builder.n_gpu_layers(v);
    }
    if let Some(v) = options.main_gpu {
        builder = builder.main_gpu(v);
    }
    if let Some(v) = options.split_mode {
        builder = builder.split_mode(parse_split_mode(&v)?);
    }
    if let Some(v) = options.use_mlock {
        builder = builder.use_mlock(v);
    }
    if let Some(v) = options.devices {
        builder = builder.devices(v);
    }
    Ok(builder)
}

struct BehaviorOptions {
    chat_template: Option<String>,
    force_json_grammar: Option<bool>,
    reasoning_format: Option<String>,
    extra_body: Option<serde_json::Value>,
    model_dir: Option<String>,
    hf_revision: Option<String>,
    mmproj_use_gpu: Option<bool>,
    media_marker: Option<String>,
    system_prompt: Option<String>,
}

struct SamplingOptions {
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    repeat_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    repeat_last_n: Option<i32>,
    seed: Option<u32>,
}

struct RuntimeOptions {
    n_ctx: Option<u32>,
    n_batch: Option<u32>,
    n_ubatch: Option<u32>,
    n_threads: Option<i32>,
    n_threads_batch: Option<i32>,
    n_gpu_layers: Option<u32>,
    main_gpu: Option<i32>,
    split_mode: Option<String>,
    use_mlock: Option<bool>,
    devices: Option<Vec<usize>>,
}

fn parse_model_source(
    model_path: Option<String>,
    repo_id: Option<String>,
    hf_filename: Option<String>,
    mmproj_filename: Option<String>,
) -> PyResult<LlamaModelSource> {
    match (model_path, repo_id) {
        (Some(path), None) => Ok(LlamaModelSource::gguf(path)),
        (None, Some(repo)) => Ok(LlamaModelSource::HuggingFace {
            repo_id: repo,
            filename: hf_filename,
            mmproj_filename,
        }),
        (None, None) => Err(PyRuntimeError::new_err(
            "either model_path (GGUF) or repo_id (HuggingFace) is required",
        )),
        (Some(_), Some(_)) => Err(PyRuntimeError::new_err(
            "set only one source: model_path (GGUF) or repo_id (HuggingFace)",
        )),
    }
}

fn parse_reasoning_format(
    value: &str,
) -> PyResult<autoagents_llamacpp::config::LlamaCppReasoningFormat> {
    match value.to_ascii_lowercase().as_str() {
        "none" => Ok(autoagents_llamacpp::config::LlamaCppReasoningFormat::None),
        "auto" => Ok(autoagents_llamacpp::config::LlamaCppReasoningFormat::Auto),
        "deepseek" => Ok(autoagents_llamacpp::config::LlamaCppReasoningFormat::Deepseek),
        "deepseek_legacy" => {
            Ok(autoagents_llamacpp::config::LlamaCppReasoningFormat::DeepseekLegacy)
        }
        other => Err(PyRuntimeError::new_err(format!(
            "invalid reasoning_format '{other}', expected one of: none, auto, deepseek, deepseek_legacy"
        ))),
    }
}

fn parse_split_mode(value: &str) -> PyResult<autoagents_llamacpp::config::LlamaCppSplitMode> {
    match value.to_ascii_lowercase().as_str() {
        "none" => Ok(autoagents_llamacpp::config::LlamaCppSplitMode::None),
        "layer" => Ok(autoagents_llamacpp::config::LlamaCppSplitMode::Layer),
        "row" => Ok(autoagents_llamacpp::config::LlamaCppSplitMode::Row),
        other => Err(PyRuntimeError::new_err(format!(
            "invalid split_mode '{other}', expected one of: none, layer, row"
        ))),
    }
}

#[pyfunction]
fn backend_build_info(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let info = PyDict::new(py);
    info.set_item("backend", "llamacpp")?;
    info.set_item("cuda", cfg!(feature = "cuda"))?;
    info.set_item("cuda_no_vmm", cfg!(feature = "cuda-no-vmm"))?;
    info.set_item("metal", cfg!(feature = "metal"))?;
    info.set_item("vulkan", cfg!(feature = "vulkan"))?;
    info.set_item("mtmd", true)?;
    Ok(info.unbind())
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Share the tokio runtime with the base autoagents-py binding so that
    // tokio::task::spawn_blocking works correctly inside future_into_py futures.
    let runtime = _autoagents_py::runtime::get_runtime()
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    pyo3_async_runtimes::tokio::init_with_runtime(runtime).map_err(|_| {
        pyo3::exceptions::PyRuntimeError::new_err("tokio runtime bridge already initialized")
    })?;
    m.add_class::<PyLlamaCppBuilder>()?;
    m.add_function(wrap_pyfunction!(backend_build_info, m)?)?;
    Ok(())
}
