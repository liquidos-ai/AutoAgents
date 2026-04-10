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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn init_python() {
        Python::initialize();
    }

    #[test]
    fn builder_setters_store_configuration_values() {
        init_python();
        Python::attach(|py| {
            let builder = Py::new(py, PyLlamaCppBuilder::new()).expect("builder should create");
            {
                let mut builder_ref = builder.borrow_mut(py);
                builder_ref =
                    PyLlamaCppBuilder::model_path(builder_ref, "/tmp/model.gguf".to_string());
                builder_ref = PyLlamaCppBuilder::repo_id(builder_ref, "repo/model".to_string());
                builder_ref = PyLlamaCppBuilder::hf_filename(builder_ref, "model.gguf".to_string());
                builder_ref =
                    PyLlamaCppBuilder::mmproj_filename(builder_ref, "mmproj.gguf".to_string());
                builder_ref = PyLlamaCppBuilder::hf_revision(builder_ref, "main".to_string());
                builder_ref = PyLlamaCppBuilder::model_dir(builder_ref, "/tmp/models".to_string());
                builder_ref =
                    PyLlamaCppBuilder::chat_template(builder_ref, "chat template".to_string());
                builder_ref = PyLlamaCppBuilder::force_json_grammar(builder_ref, true);
                builder_ref =
                    PyLlamaCppBuilder::reasoning_format(builder_ref, "deepseek".to_string());
                builder_ref =
                    PyLlamaCppBuilder::extra_body_json(builder_ref, "{\"seed\":99}".to_string())
                        .expect("extra body should parse");
                builder_ref = PyLlamaCppBuilder::mmproj_use_gpu(builder_ref, false);
                builder_ref = PyLlamaCppBuilder::media_marker(builder_ref, "<image>".to_string());
                builder_ref = PyLlamaCppBuilder::max_tokens(builder_ref, 256);
                builder_ref = PyLlamaCppBuilder::temperature(builder_ref, 0.4);
                builder_ref = PyLlamaCppBuilder::top_p(builder_ref, 0.9);
                builder_ref = PyLlamaCppBuilder::top_k(builder_ref, 30);
                builder_ref = PyLlamaCppBuilder::repeat_penalty(builder_ref, 1.1);
                builder_ref = PyLlamaCppBuilder::frequency_penalty(builder_ref, 0.2);
                builder_ref = PyLlamaCppBuilder::presence_penalty(builder_ref, 0.3);
                builder_ref = PyLlamaCppBuilder::repeat_last_n(builder_ref, 64);
                builder_ref = PyLlamaCppBuilder::seed(builder_ref, 7);
                builder_ref = PyLlamaCppBuilder::n_ctx(builder_ref, 4096);
                builder_ref = PyLlamaCppBuilder::n_batch(builder_ref, 128);
                builder_ref = PyLlamaCppBuilder::n_ubatch(builder_ref, 32);
                builder_ref = PyLlamaCppBuilder::n_threads(builder_ref, 8);
                builder_ref = PyLlamaCppBuilder::n_threads_batch(builder_ref, 4);
                builder_ref = PyLlamaCppBuilder::n_gpu_layers(builder_ref, 20);
                builder_ref = PyLlamaCppBuilder::main_gpu(builder_ref, 0);
                builder_ref = PyLlamaCppBuilder::split_mode(builder_ref, "layer".to_string());
                builder_ref = PyLlamaCppBuilder::use_mlock(builder_ref, true);
                builder_ref = PyLlamaCppBuilder::devices(builder_ref, vec![0, 2]);
                let _builder_ref =
                    PyLlamaCppBuilder::system_prompt(builder_ref, "system".to_string());
            }

            let builder_ref = builder.borrow(py);
            assert_eq!(builder_ref.model_path.as_deref(), Some("/tmp/model.gguf"));
            assert_eq!(builder_ref.repo_id.as_deref(), Some("repo/model"));
            assert_eq!(builder_ref.hf_filename.as_deref(), Some("model.gguf"));
            assert_eq!(builder_ref.mmproj_filename.as_deref(), Some("mmproj.gguf"));
            assert_eq!(builder_ref.hf_revision.as_deref(), Some("main"));
            assert_eq!(builder_ref.model_dir.as_deref(), Some("/tmp/models"));
            assert_eq!(builder_ref.chat_template.as_deref(), Some("chat template"));
            assert_eq!(builder_ref.force_json_grammar, Some(true));
            assert_eq!(builder_ref.reasoning_format.as_deref(), Some("deepseek"));
            assert_eq!(builder_ref.extra_body, Some(json!({"seed": 99})));
            assert_eq!(builder_ref.mmproj_use_gpu, Some(false));
            assert_eq!(builder_ref.media_marker.as_deref(), Some("<image>"));
            assert_eq!(builder_ref.max_tokens, Some(256));
            assert_eq!(builder_ref.temperature, Some(0.4));
            assert_eq!(builder_ref.top_p, Some(0.9));
            assert_eq!(builder_ref.top_k, Some(30));
            assert_eq!(builder_ref.repeat_penalty, Some(1.1));
            assert_eq!(builder_ref.frequency_penalty, Some(0.2));
            assert_eq!(builder_ref.presence_penalty, Some(0.3));
            assert_eq!(builder_ref.repeat_last_n, Some(64));
            assert_eq!(builder_ref.seed, Some(7));
            assert_eq!(builder_ref.n_ctx, Some(4096));
            assert_eq!(builder_ref.n_batch, Some(128));
            assert_eq!(builder_ref.n_ubatch, Some(32));
            assert_eq!(builder_ref.n_threads, Some(8));
            assert_eq!(builder_ref.n_threads_batch, Some(4));
            assert_eq!(builder_ref.n_gpu_layers, Some(20));
            assert_eq!(builder_ref.main_gpu, Some(0));
            assert_eq!(builder_ref.split_mode.as_deref(), Some("layer"));
            assert_eq!(builder_ref.use_mlock, Some(true));
            assert_eq!(builder_ref.devices.as_deref(), Some(&[0, 2][..]));
            assert_eq!(builder_ref.system_prompt.as_deref(), Some("system"));
        });
    }

    #[test]
    fn parsing_helpers_cover_supported_sources_and_errors() {
        let gguf = parse_model_source(Some("/tmp/model.gguf".to_string()), None, None, None)
            .expect("gguf source should parse");
        assert!(matches!(gguf, LlamaModelSource::Gguf { .. }));

        let huggingface = parse_model_source(
            None,
            Some("repo/model".to_string()),
            Some("model.gguf".to_string()),
            Some("mmproj.gguf".to_string()),
        )
        .expect("huggingface source should parse");
        assert!(matches!(huggingface, LlamaModelSource::HuggingFace { .. }));

        let err =
            parse_model_source(None, None, None, None).expect_err("missing source should fail");
        assert!(err.to_string().contains("either model_path"));

        let err = parse_model_source(
            Some("/tmp/model.gguf".to_string()),
            Some("repo/model".to_string()),
            None,
            None,
        )
        .expect_err("multiple sources should fail");
        assert!(err.to_string().contains("set only one source"));

        assert!(matches!(
            parse_reasoning_format("auto").expect("auto should parse"),
            autoagents_llamacpp::config::LlamaCppReasoningFormat::Auto
        ));
        assert!(matches!(
            parse_reasoning_format("deepseek_legacy").expect("legacy should parse"),
            autoagents_llamacpp::config::LlamaCppReasoningFormat::DeepseekLegacy
        ));
        assert!(
            parse_reasoning_format("bogus")
                .expect_err("invalid reasoning format should fail")
                .to_string()
                .contains("invalid reasoning_format")
        );

        assert!(matches!(
            parse_split_mode("none").expect("none should parse"),
            autoagents_llamacpp::config::LlamaCppSplitMode::None
        ));
        assert!(matches!(
            parse_split_mode("ROW").expect("row should parse"),
            autoagents_llamacpp::config::LlamaCppSplitMode::Row
        ));
        assert!(
            parse_split_mode("bogus")
                .expect_err("invalid split mode should fail")
                .to_string()
                .contains("invalid split_mode")
        );
    }

    #[test]
    fn backend_build_info_reports_current_backend_flags() {
        init_python();
        Python::attach(|py| {
            let info = backend_build_info(py).expect("build info should succeed");
            let info = info.bind(py);
            assert_eq!(
                info.get_item("backend")
                    .expect("backend key should exist")
                    .expect("backend value should exist")
                    .extract::<String>()
                    .expect("backend should be a string"),
                "llamacpp"
            );
            assert!(
                info.get_item("mtmd")
                    .expect("mtmd key should exist")
                    .expect("mtmd value should exist")
                    .extract::<bool>()
                    .expect("mtmd should be bool")
            );
        });
    }
}
