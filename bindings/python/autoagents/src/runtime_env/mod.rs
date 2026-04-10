pub mod topic;

use crate::events::PyEventStream;
use autoagents_core::environment::Environment;
use autoagents_core::runtime::RuntimeError;
use autoagents_core::runtime::SingleThreadedRuntime;
use autoagents_core::runtime::{Runtime, TypedRuntime};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

fn to_py_runtime_error(err: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}
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
        py.detach(|| -> PyResult<_> {
            runtime
                .block_on(async { self.inner.register_runtime(rt).await })
                .map_err(to_py_runtime_error)
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Start registered runtimes in the background using the shared Tokio runtime.
    pub fn run(&mut self, py: Python<'_>) -> PyResult<()> {
        let runtime = crate::runtime::get_runtime().map_err(PyRuntimeError::new_err)?;
        py.detach(|| -> PyResult<_> {
            runtime
                .block_on(async { self.inner.run_background().await })
                .map_err(to_py_runtime_error)
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Return an `EventStream` over all environment events.
    pub fn event_stream(&mut self, py: Python<'_>) -> PyResult<PyEventStream> {
        let runtime = crate::runtime::get_runtime().map_err(PyRuntimeError::new_err)?;
        let stream = py
            .detach(|| -> PyResult<_> {
                runtime
                    .block_on(async { self.inner.subscribe_events(None).await })
                    .map_err(to_py_runtime_error)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyEventStream {
            rx: Arc::new(tokio::sync::Mutex::new(stream)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_env::topic::PyTopic;
    use pyo3_async_runtimes::TaskLocals;

    fn init_runtime_bridge() {
        Python::initialize();
        let runtime = crate::runtime::get_runtime().expect("shared runtime should initialize");
        let _ = pyo3_async_runtimes::tokio::init_with_runtime(runtime);
    }

    async fn await_py_any(
        value: Py<PyAny>,
        task_locals: Option<&TaskLocals>,
    ) -> PyResult<Py<PyAny>> {
        let future = Python::attach(|py| {
            crate::async_bridge::into_future(value.bind(py).clone(), task_locals)
        })?;
        future.await
    }

    #[test]
    fn runtime_and_environment_methods_bridge_async_operations() {
        init_runtime_bridge();

        let (runtime, topic, env_stream_obj) = Python::attach(|py| {
            let runtime = PySingleThreadedRuntime::new();
            let topic = PyTopic::new("jobs".to_string());

            let mut env = PyEnvironment::new();
            env.register_runtime(py, &runtime)?;
            env.run(py)?;
            let env_stream = Py::new(py, env.event_stream(py)?)?.into_any();

            Ok::<_, PyErr>((runtime, topic, env_stream))
        })
        .expect("runtime api should construct environment wrappers");

        Python::attach(|py| -> PyResult<()> {
            let event_loop = py.import("asyncio")?.call_method0("new_event_loop")?;
            let task_locals = TaskLocals::new(event_loop.clone()).copy_context(py)?;
            pyo3_async_runtimes::tokio::run_until_complete(event_loop, async move {
                let (publish_coro, runtime_stream_coro) = Python::attach(|py| {
                    let publish_coro = runtime
                        .publish(py, &topic, "run this task".to_string())?
                        .unbind();
                    let runtime_stream_coro = runtime.event_stream(py)?.unbind();
                    Ok::<_, PyErr>((publish_coro, runtime_stream_coro))
                })?;

                let publish_result = await_py_any(publish_coro, Some(&task_locals))
                    .await
                    .expect("publish coroutine should resolve");
                let runtime_stream = await_py_any(runtime_stream_coro, Some(&task_locals))
                    .await
                    .expect("event_stream coroutine should resolve");

                Python::attach(|py| {
                    assert!(publish_result.bind(py).is_none());
                    assert!(
                        runtime_stream
                            .bind(py)
                            .hasattr("__anext__")
                            .unwrap_or(false)
                    );
                    assert!(
                        env_stream_obj
                            .bind(py)
                            .hasattr("__anext__")
                            .unwrap_or(false)
                    );
                });
                Ok(())
            })
        })
        .expect("temporary Python event loop should run");
    }

    #[test]
    fn to_py_runtime_error_preserves_display_text() {
        let err = to_py_runtime_error("runtime failed");
        assert_eq!(err.to_string(), "RuntimeError: runtime failed");
    }
}
