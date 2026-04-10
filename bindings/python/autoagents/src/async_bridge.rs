use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_async_runtimes::TaskLocals;
use std::future::Future;
use std::pin::Pin;

pub(crate) type PyAwaitableFuture = Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>;

fn resolve_task_locals(py: Python<'_>, task_locals: Option<&TaskLocals>) -> PyResult<TaskLocals> {
    match task_locals {
        Some(task_locals) => Ok(task_locals.clone()),
        None => pyo3_async_runtimes::tokio::get_current_locals(py),
    }
}

pub fn future_into_py<F, T>(py: Python<'_>, fut: F) -> PyResult<Bound<'_, PyAny>>
where
    F: Future<Output = PyResult<T>> + Send + 'static,
    T: for<'py> IntoPyObject<'py> + Send + 'static,
{
    pyo3_async_runtimes::tokio::future_into_py(py, fut)
}

pub(crate) fn is_awaitable(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    value.hasattr("__await__")
}

pub(crate) fn into_future(
    awaitable: Bound<'_, PyAny>,
    task_locals: Option<&TaskLocals>,
) -> PyResult<PyAwaitableFuture> {
    let py = awaitable.py();
    let task_locals = resolve_task_locals(py, task_locals)?;
    let future = pyo3_async_runtimes::into_future_with_locals(&task_locals, awaitable)?;
    Ok(Box::pin(future))
}

pub(crate) async fn resolve_maybe_awaitable(
    result_obj: Py<PyAny>,
    is_awaitable: bool,
    task_locals: Option<&TaskLocals>,
) -> PyResult<Py<PyAny>> {
    if !is_awaitable {
        return Ok(result_obj);
    }

    let task_locals = task_locals.cloned();
    let future =
        Python::attach(|py| into_future(result_obj.bind(py).clone(), task_locals.as_ref()))?;
    future.await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    fn init_runtime_bridge() {
        Python::initialize();
        let runtime = crate::runtime::get_runtime().expect("shared runtime should initialize");
        let _ = pyo3_async_runtimes::tokio::init_with_runtime(runtime);
    }

    fn immediate_awaitable(py: Python<'_>, value: &str) -> PyResult<Py<PyAny>> {
        let module = PyModule::from_code(
            py,
            &CString::new(
                "class ImmediateAwaitable:\n\
                 \n\
                 \tdef __init__(self, value):\n\
                 \t\tself.value = value\n\
                 \n\
                 \tdef __await__(self):\n\
                 \t\tif False:\n\
                 \t\t\tyield None\n\
                 \t\treturn self.value\n",
            )
            .expect("python module source should be valid CString"),
            &CString::new("autoagents_py/tests/async_bridge.py")
                .expect("filename should be a valid CString"),
            &CString::new("autoagents_async_bridge_tests")
                .expect("module name should be a valid CString"),
        )?;
        let awaitable = module
            .getattr("ImmediateAwaitable")?
            .call1((value.to_string(),))?;
        Ok(awaitable.unbind())
    }

    #[test]
    fn is_awaitable_detects_plain_values_and_custom_awaitables() {
        init_runtime_bridge();

        Python::attach(|py| {
            let value = 7_i32.into_pyobject(py).expect("integer should convert");
            assert!(!is_awaitable(value.as_any()).expect("plain value should be inspectable"));

            let awaitable = immediate_awaitable(py, "done").expect("awaitable should build");
            assert!(is_awaitable(awaitable.bind(py)).expect("awaitable should be inspectable"));
        });
    }

    #[test]
    fn resolve_maybe_awaitable_handles_sync_and_async_results() {
        init_runtime_bridge();
        Python::attach(|py| -> PyResult<()> {
            let event_loop = py.import("asyncio")?.call_method0("new_event_loop")?;
            let task_locals = TaskLocals::new(event_loop.clone()).copy_context(py)?;
            pyo3_async_runtimes::tokio::run_until_complete(event_loop, async move {
                let plain = Python::attach(|py| py.None());
                let plain_resolved = resolve_maybe_awaitable(plain, false, None)
                    .await
                    .expect("plain value should resolve immediately");
                Python::attach(|py| assert!(plain_resolved.bind(py).is_none()));

                let awaitable = Python::attach(|py| immediate_awaitable(py, "resolved"))?;

                let resolved = resolve_maybe_awaitable(awaitable, true, Some(&task_locals))
                    .await
                    .expect("awaitable should resolve");
                Python::attach(|py| {
                    assert_eq!(
                        resolved
                            .bind(py)
                            .extract::<String>()
                            .expect("resolved awaitable should produce string"),
                        "resolved"
                    );
                });
                Ok(())
            })
        })
        .expect("temporary Python event loop should run");
    }
}
