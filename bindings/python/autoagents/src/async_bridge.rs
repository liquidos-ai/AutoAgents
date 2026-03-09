use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyAny;
use pyo3::types::PyModule;
use pyo3_async_runtimes::TaskLocals;
use std::future::Future;
use std::pin::Pin;

pub(crate) type PyAwaitableFuture = Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>;
static NATIVE_FUTURE_WRAPPER: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

fn native_future_wrapper(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    NATIVE_FUTURE_WRAPPER
        .get_or_try_init(py, || {
            let module = PyModule::from_code(
                py,
                pyo3::ffi::c_str!(concat!(
                    "import asyncio\n",
                    "\n",
                    "async def _await_native_future(future):\n",
                    "    loop = asyncio.get_running_loop()\n",
                    "    waiter = loop.create_future()\n",
                    "\n",
                    "    def _complete(_):\n",
                    "        if not waiter.done():\n",
                    "            waiter.set_result(None)\n",
                    "\n",
                    "    future.add_done_callback(_complete)\n",
                    "    try:\n",
                    "        if not future.done():\n",
                    "            await waiter\n",
                    "        return await future\n",
                    "    except BaseException:\n",
                    "        if not future.done():\n",
                    "            future.cancel()\n",
                    "            try:\n",
                    "                await future\n",
                    "            except BaseException:\n",
                    "                pass\n",
                    "        raise\n",
                )),
                pyo3::ffi::c_str!("_autoagents_async_bridge.py"),
                pyo3::ffi::c_str!("_autoagents_async_bridge"),
            )?;
            module
                .getattr("_await_native_future")
                .map(|func| func.unbind())
        })
        .map(|func| func.bind(py))
}

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
    let native_future = pyo3_async_runtimes::tokio::future_into_py(py, fut)?;
    native_future_wrapper(py)?.call1((native_future,))
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
