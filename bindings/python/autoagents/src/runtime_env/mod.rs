pub mod topic;

use crate::events::PyEventStream;
use autoagents_core::environment::Environment;
use autoagents_core::runtime::RuntimeError;
use autoagents_core::runtime::SingleThreadedRuntime;
use autoagents_core::runtime::{Runtime, TypedRuntime};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::Arc;

pub use topic::PyTopic;

/// Wrapper around `SingleThreadedRuntime`.
#[pyclass(name = "Runtime")]
pub struct PySingleThreadedRuntime {
    pub inner: Arc<SingleThreadedRuntime>,
}

#[pymethods]
impl PySingleThreadedRuntime {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: SingleThreadedRuntime::new(None),
        }
    }

    /// Publish a task string to a named topic. Returns a coroutine.
    pub fn publish<'py>(
        &self,
        py: Python<'py>,
        topic: &PyTopic,
        task: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        use autoagents_core::actor::Topic;
        use autoagents_core::agent::task::Task;

        let rt = Arc::clone(&self.inner);
        let topic_name = topic.name.clone();

        crate::async_bridge::future_into_py(py, async move {
            let typed_topic = Topic::<Task>::new(&topic_name);
            rt.publish(&typed_topic, Task::new(task))
                .await
                .map_err(|e: RuntimeError| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Python::attach(|py: Python<'_>| py.None()))
        })
    }

    /// Returns a coroutine resolving to an `EventStream`.
    pub fn event_stream<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let rt = Arc::clone(&self.inner);
        crate::async_bridge::future_into_py(py, async move {
            let stream = rt.subscribe_events().await;
            Python::attach(|py: Python<'_>| -> PyResult<Py<PyAny>> {
                PyEventStream {
                    rx: Arc::new(tokio::sync::Mutex::new(stream)),
                }
                .into_pyobject(py)
                .map(|b| b.into_any().unbind())
            })
        })
    }
}

/// Wrapper around `Environment`.
#[pyclass(name = "Environment")]
pub struct PyEnvironment {
    inner: Environment,
}

#[pymethods]
impl PyEnvironment {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: Environment::new(None),
        }
    }

    /// Register a runtime. Blocks briefly on the tokio runtime (no GIL held).
    pub fn register_runtime(
        &mut self,
        py: Python<'_>,
        runtime: &PySingleThreadedRuntime,
    ) -> PyResult<()> {
        let rt = Arc::clone(&runtime.inner) as Arc<dyn Runtime>;
        let runtime = crate::runtime::get_runtime().map_err(PyRuntimeError::new_err)?;
        py.detach(|| runtime.block_on(async { self.inner.register_runtime(rt).await }))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Start registered runtimes in the background using the shared Tokio runtime.
    pub fn run(&mut self, py: Python<'_>) -> PyResult<()> {
        let runtime = crate::runtime::get_runtime().map_err(PyRuntimeError::new_err)?;
        py.detach(|| runtime.block_on(async { self.inner.run_background().await }))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Return an `EventStream` over all environment events.
    pub fn event_stream(&mut self, py: Python<'_>) -> PyResult<PyEventStream> {
        let runtime = crate::runtime::get_runtime().map_err(PyRuntimeError::new_err)?;
        let stream = py
            .detach(|| runtime.block_on(async { self.inner.subscribe_events(None).await }))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyEventStream {
            rx: Arc::new(tokio::sync::Mutex::new(stream)),
        })
    }
}
