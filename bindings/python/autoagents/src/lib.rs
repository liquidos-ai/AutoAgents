mod agent;
pub mod async_bridge;
mod convert;
mod events;
pub mod llm;
mod memory;
pub mod runtime;
mod runtime_env;
mod tool;

use agent::builder::{PyActorAgentHandle, PyAgentBuilder, PyAgentHandle, PyRunStream};
use agent::py_agent::{
    PyExecutionJsonStream, PyExecutionLLM, PyExecutionMemory, PyExecutionStringStream,
};
use events::PyEventStream;
use llm::builder::{_llm_provider_from_capsule, PyLLMBuilder, PyLLMProvider};
use llm::pipeline::{pipeline_cache_layer, pipeline_python_layer, pipeline_retry_layer};
use memory::{PyMemoryProvider, memory_provider_from_impl, sliding_window_memory};
use pyo3::prelude::*;
use runtime_env::topic::PyTopic;
use runtime_env::{PyEnvironment, PySingleThreadedRuntime};
use tool::PyTool;

#[pymodule]
fn autoagents_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    init_runtime_bridge()?;
    register_llm_api(m)?;
    register_memory_api(m)?;
    register_tool_api(m)?;
    register_agent_api(m)?;
    register_event_api(m)?;
    register_runtime_api(m)?;
    Ok(())
}

fn init_runtime_bridge() -> PyResult<()> {
    let runtime = runtime::get_runtime().map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    pyo3_async_runtimes::tokio::init_with_runtime(runtime).map_err(|_| {
        pyo3::exceptions::PyRuntimeError::new_err("tokio runtime bridge already initialized")
    })?;
    Ok(())
}

fn register_llm_api(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLLMBuilder>()?;
    m.add_class::<PyLLMProvider>()?;
    m.add_function(wrap_pyfunction!(_llm_provider_from_capsule, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline_cache_layer, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline_retry_layer, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline_python_layer, m)?)?;
    m.add_class::<PyExecutionLLM>()?;
    m.add_class::<PyExecutionMemory>()?;
    m.add_class::<PyExecutionStringStream>()?;
    m.add_class::<PyExecutionJsonStream>()?;
    Ok(())
}

fn register_memory_api(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMemoryProvider>()?;
    m.add_function(wrap_pyfunction!(sliding_window_memory, m)?)?;
    m.add_function(wrap_pyfunction!(memory_provider_from_impl, m)?)?;
    Ok(())
}

fn register_tool_api(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTool>()?;
    Ok(())
}

fn register_agent_api(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAgentBuilder>()?;
    m.add_class::<PyAgentHandle>()?;
    m.add_class::<PyActorAgentHandle>()?;
    m.add_class::<PyRunStream>()?;
    Ok(())
}

fn register_event_api(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEventStream>()?;
    Ok(())
}

fn register_runtime_api(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySingleThreadedRuntime>()?;
    m.add_class::<PyEnvironment>()?;
    m.add_class::<PyTopic>()?;
    Ok(())
}
