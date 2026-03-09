use autoagents_llm::LLMProvider;
use autoagents_mistral_rs::{
    IsqType, MistralRsProvider, ModelSource as MistralModelSource, models::ModelType,
};
use autoagents_py::llm::builder::canonicalize_llm_provider;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

fn parse_model_type(value: Option<&str>) -> PyResult<ModelType> {
    match value.unwrap_or("auto").to_ascii_lowercase().as_str() {
        "auto" => Ok(ModelType::Auto),
        "text" => Ok(ModelType::Text),
        "vision" => Ok(ModelType::Vision),
        other => Err(PyRuntimeError::new_err(format!(
            "invalid model_type '{other}', use one of: auto, text, vision"
        ))),
    }
}

fn parse_model_source(
    model_dir: Option<String>,
    gguf_files: Option<Vec<String>>,
    tokenizer: Option<String>,
    chat_template: Option<String>,
    repo_id: Option<String>,
    revision: Option<String>,
    model_type: Option<String>,
) -> PyResult<MistralModelSource> {
    match (model_dir, gguf_files, repo_id) {
        (Some(dir), Some(files), None) => Ok(MistralModelSource::Gguf {
            model_dir: dir,
            files,
            tokenizer,
            chat_template,
        }),
        (None, None, repo) => Ok(MistralModelSource::HuggingFace {
            repo_id: repo.unwrap_or_else(|| "microsoft/Phi-3.5-mini-instruct".to_string()),
            revision,
            model_type: parse_model_type(model_type.as_deref())?,
        }),
        _ => Err(PyRuntimeError::new_err(
            "for GGUF source set both model_dir and gguf_files; otherwise use HuggingFace repo_id",
        )),
    }
}

fn parse_isq_type(value: &str) -> PyResult<IsqType> {
    let normalized = value.to_ascii_uppercase();
    serde_json::from_str(&format!("\"{normalized}\""))
        .map_err(|_| PyRuntimeError::new_err(format!("invalid isq_type '{value}'")))
}

#[pyclass(name = "MistralRsBuilder")]
#[derive(Default)]
pub struct PyMistralRsBuilder {
    model_dir: Option<String>,
    gguf_files: Option<Vec<String>>,
    tokenizer: Option<String>,
    chat_template: Option<String>,
    repo_id: Option<String>,
    revision: Option<String>,
    model_type: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    system_prompt: Option<String>,
    isq_type: Option<String>,
    paged_attention: Option<bool>,
    logging: Option<bool>,
}

#[pymethods]
impl PyMistralRsBuilder {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn repo_id(mut slf: PyRefMut<'_, Self>, repo_id: String) -> PyRefMut<'_, Self> {
        slf.repo_id = Some(repo_id);
        slf
    }

    pub fn revision(mut slf: PyRefMut<'_, Self>, revision: String) -> PyRefMut<'_, Self> {
        slf.revision = Some(revision);
        slf
    }

    pub fn model_type(mut slf: PyRefMut<'_, Self>, model_type: String) -> PyRefMut<'_, Self> {
        slf.model_type = Some(model_type);
        slf
    }

    pub fn model_dir(mut slf: PyRefMut<'_, Self>, model_dir: String) -> PyRefMut<'_, Self> {
        slf.model_dir = Some(model_dir);
        slf
    }

    pub fn gguf_files(mut slf: PyRefMut<'_, Self>, files: Vec<String>) -> PyRefMut<'_, Self> {
        slf.gguf_files = Some(files);
        slf
    }

    pub fn tokenizer(mut slf: PyRefMut<'_, Self>, tokenizer: String) -> PyRefMut<'_, Self> {
        slf.tokenizer = Some(tokenizer);
        slf
    }

    pub fn chat_template(mut slf: PyRefMut<'_, Self>, chat_template: String) -> PyRefMut<'_, Self> {
        slf.chat_template = Some(chat_template);
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

    pub fn system_prompt(mut slf: PyRefMut<'_, Self>, value: String) -> PyRefMut<'_, Self> {
        slf.system_prompt = Some(value);
        slf
    }

    pub fn isq_type(mut slf: PyRefMut<'_, Self>, isq_type: String) -> PyRefMut<'_, Self> {
        slf.isq_type = Some(isq_type);
        slf
    }

    pub fn paged_attention(mut slf: PyRefMut<'_, Self>, enable: bool) -> PyRefMut<'_, Self> {
        slf.paged_attention = Some(enable);
        slf
    }

    pub fn logging(mut slf: PyRefMut<'_, Self>, enable: bool) -> PyRefMut<'_, Self> {
        slf.logging = Some(enable);
        slf
    }

    pub fn build<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let model_dir = self.model_dir.clone();
        let gguf_files = self.gguf_files.clone();
        let tokenizer = self.tokenizer.clone();
        let chat_template = self.chat_template.clone();
        let repo_id = self.repo_id.clone();
        let revision = self.revision.clone();
        let model_type = self.model_type.clone();
        let max_tokens = self.max_tokens;
        let temperature = self.temperature;
        let top_p = self.top_p;
        let top_k = self.top_k;
        let system_prompt = self.system_prompt.clone();
        let isq_type = self.isq_type.clone();
        let paged_attention = self.paged_attention;
        let logging = self.logging;

        autoagents_py::async_bridge::future_into_py(py, async move {
            let source = parse_model_source(
                model_dir,
                gguf_files,
                tokenizer,
                chat_template,
                repo_id,
                revision,
                model_type,
            )?;

            let mut builder = MistralRsProvider::builder().model_source(source);

            if let Some(v) = isq_type {
                builder = builder.with_isq(parse_isq_type(&v)?);
            }
            if paged_attention.unwrap_or(false) {
                builder = builder.with_paged_attention();
            }
            if logging.unwrap_or(false) {
                builder = builder.with_logging();
            }
            if let Some(v) = max_tokens {
                builder = builder.max_tokens(v);
            }
            if let Some(v) = temperature {
                builder = builder.temperature(v);
            }
            if let Some(v) = top_p {
                builder = builder.top_p(v);
            }
            if let Some(v) = top_k {
                builder = builder.top_k(v);
            }
            if let Some(v) = system_prompt {
                builder = builder.system_prompt(v);
            }

            let provider = builder
                .build()
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Python::attach(|py| {
                canonicalize_llm_provider(py, Arc::new(provider) as Arc<dyn LLMProvider>)
            })
        })
    }
}

#[pyfunction]
fn backend_build_info(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let info = PyDict::new(py);
    info.set_item("backend", "mistral-rs")?;
    info.set_item("cuda", cfg!(feature = "cuda"))?;
    info.set_item("cudnn", cfg!(feature = "cudnn"))?;
    info.set_item("metal", cfg!(feature = "metal"))?;
    info.set_item("flash_attn", cfg!(feature = "flash-attn"))?;
    info.set_item("accelerate", cfg!(feature = "accelerate"))?;
    info.set_item("mkl", cfg!(feature = "mkl"))?;
    info.set_item("nccl", cfg!(feature = "nccl"))?;
    info.set_item("ring", cfg!(feature = "ring"))?;
    Ok(info.unbind())
}

#[pymodule]
fn _autoagents_mistral_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Share the tokio runtime with the base autoagents-py binding so that
    // tokio APIs work correctly inside future_into_py futures.
    let runtime =
        autoagents_py::runtime::get_runtime().map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    pyo3_async_runtimes::tokio::init_with_runtime(runtime).map_err(|_| {
        pyo3::exceptions::PyRuntimeError::new_err("tokio runtime bridge already initialized")
    })?;
    m.add_class::<PyMistralRsBuilder>()?;
    m.add_function(wrap_pyfunction!(backend_build_info, m)?)?;
    Ok(())
}
