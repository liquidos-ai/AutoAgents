use crate::agent::executor::parse_executor_spec;
use crate::agent::py_agent::{
    ActorSendFn, AgentOutputStream, HookErrorState, PyAgentDef, PyAgentOutput, PyExecutorBuildable,
    PyRunnable,
};
use crate::convert::json_value_to_py;
use crate::events::{PyEventStream, PySharedEventStream};
use crate::llm::builder::extract_llm_provider;
use crate::memory::PyMemoryProvider;
use crate::runtime_env::PySingleThreadedRuntime;
use autoagents_core::agent::memory::MemoryProvider;
use autoagents_core::agent::task::Task;
use autoagents_core::runtime::Runtime;
use autoagents_llm::LLMProvider;
use autoagents_protocol::{Event, ImageMime, SubmissionId};
use futures::StreamExt;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::types::PyDict;
use std::sync::Arc;

type BuiltAgentInputs = (Arc<dyn LLMProvider>, Box<dyn MemoryProvider>, PyAgentDef);

fn hook_error_to_pyerr(hook_errors: &HookErrorState) -> PyResult<()> {
    if let Some(error) = hook_errors.take() {
        return Err(PyRuntimeError::new_err(error));
    }
    Ok(())
}

/// Builder that mirrors `AgentBuilder::<Executor, DirectAgent>::new(executor)` in Rust.
///
/// Example:
/// ```python
/// agent = ReActAgent("planner", "Plans tasks").tools([t]).max_turns(5)
/// handle = await AgentBuilder(agent).llm(llm).memory(SlidingWindowMemory(20)).build()
/// ```
#[pyclass(name = "AgentBuilder")]
pub struct PyAgentBuilder {
    // Executor config — owns the build logic, carries only per-executor state.
    executor: Box<dyn PyExecutorBuildable>,
    // Shared agent-definition fields extracted from the executor handle.
    agent_def: PyAgentDef,
    // Builder-level configuration (LLM, memory, runtime).
    llm: Option<Arc<dyn autoagents_llm::LLMProvider>>,
    memory: Option<Box<dyn MemoryProvider>>,
    runtime: Option<Arc<dyn Runtime>>,
    topics: Vec<String>,
}

#[pymethods]
impl PyAgentBuilder {
    #[new]
    pub fn new(agent: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (executor, agent_def) = parse_executor_spec(agent)?;

        Ok(Self {
            executor,
            agent_def,
            llm: None,
            memory: None,
            runtime: None,
            topics: Vec::new(),
        })
    }

    pub fn llm<'a>(
        mut slf: PyRefMut<'a, Self>,
        provider: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        slf.llm = Some(extract_llm_provider(provider).map_err(|_| {
            PyRuntimeError::new_err(
                "llm() expects an AutoAgents LLMProvider returned by LLMBuilder.build()",
            )
        })?);
        Ok(slf)
    }

    pub fn memory<'a>(
        mut slf: PyRefMut<'a, Self>,
        mem: PyRef<'_, PyMemoryProvider>,
    ) -> PyRefMut<'a, Self> {
        slf.memory = Some(mem.clone_memory());
        slf
    }

    /// Set the runtime for actor-based builds (`build_actor()`).
    pub fn runtime<'a>(
        mut slf: PyRefMut<'a, Self>,
        rt: PyRef<'_, PySingleThreadedRuntime>,
    ) -> PyRefMut<'a, Self> {
        slf.runtime = Some(Arc::clone(&rt.inner) as Arc<dyn Runtime>);
        slf
    }

    /// Subscribe to a named topic (actor agents only).
    pub fn subscribe<'a>(mut slf: PyRefMut<'a, Self>, topic: String) -> PyRefMut<'a, Self> {
        slf.topics.push(topic);
        slf
    }

    /// Build a `DirectAgent`-backed handle. Returns a coroutine → `AgentHandle`.
    pub fn build<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (llm, memory, mut agent_def) = self.build_inputs()?;
        let hook_errors = agent_def.hook_errors.clone();
        hook_errors.clear();
        agent_def.task_locals = Some(pyo3_async_runtimes::tokio::get_current_locals(py)?);
        let fut = self.executor.build_direct(agent_def, llm, memory);

        crate::async_bridge::future_into_py(py, async move {
            let (agent, event_stream) = fut.await?;
            hook_error_to_pyerr(&hook_errors)?;
            let events = Arc::new(PySharedEventStream::new(event_stream));
            Python::attach(|py: Python<'_>| -> PyResult<Py<PyAny>> {
                let result = PyAgentHandle {
                    agent,
                    events,
                    hook_errors,
                }
                .into_pyobject(py)
                .map(|b| b.into_any().unbind())?;
                Ok(result)
            })
        })
    }

    /// Build an `ActorAgent`-backed handle registered in the given runtime.
    /// Returns a coroutine → `ActorAgentHandle`.
    pub fn build_actor<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (llm, memory, mut agent_def) = self.build_inputs()?;
        let hook_errors = agent_def.hook_errors.clone();
        hook_errors.clear();
        agent_def.task_locals = Some(pyo3_async_runtimes::tokio::get_current_locals(py)?);
        let runtime = self
            .runtime
            .clone()
            .ok_or_else(|| PyRuntimeError::new_err("runtime is required for build_actor()"))?;
        let fut = self
            .executor
            .build_actor(agent_def, llm, memory, runtime, self.topics.clone());

        crate::async_bridge::future_into_py(py, async move {
            let send_fn: ActorSendFn = fut.await?;
            hook_error_to_pyerr(&hook_errors)?;
            Python::attach(|py: Python<'_>| -> PyResult<Py<PyAny>> {
                PyActorAgentHandle { send_fn }
                    .into_pyobject(py)
                    .map(|b| b.into_any().unbind())
            })
        })
    }
}

impl PyAgentBuilder {
    fn build_inputs(&self) -> PyResult<BuiltAgentInputs> {
        let llm = self
            .llm
            .clone()
            .ok_or_else(|| PyRuntimeError::new_err("LLM provider is required"))?;
        let memory = self
            .memory
            .as_ref()
            .map(|memory| memory.clone_box())
            .ok_or_else(|| PyRuntimeError::new_err("Memory provider is required"))?;
        Ok((llm, memory, self.clone_agent_def()))
    }

    fn clone_agent_def(&self) -> PyAgentDef {
        let hooks =
            Python::attach(|py| self.agent_def.hooks.as_ref().map(|hook| hook.clone_ref(py)));
        PyAgentDef {
            name: self.agent_def.name.clone(),
            description: self.agent_def.description.clone(),
            tools: self.agent_def.tools.clone(),
            output_schema: self.agent_def.output_schema.clone(),
            hooks,
            task_locals: self.agent_def.task_locals.clone(),
            hook_errors: self.agent_def.hook_errors.clone(),
        }
    }
}

/// Handle to a built direct agent. Produced by `AgentBuilder.build()`.
#[pyclass(name = "AgentHandle")]
pub struct PyAgentHandle {
    agent: Arc<dyn PyRunnable>,
    events: Arc<PySharedEventStream>,
    hook_errors: HookErrorState,
}

#[pymethods]
impl PyAgentHandle {
    fn __repr__(&self) -> &str {
        "AgentHandle(<built>)"
    }

    /// Run on *task*. Returns a coroutine resolving to a Python dict with
    /// `response`, `tool_calls`, `done`, and `events` keys.
    pub fn run<'py>(
        &self,
        py: Python<'py>,
        task: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let task = py_task_to_rust_task(task)?;
        let submission_id = task.submission_id;
        let agent = Arc::clone(&self.agent);
        let events = Arc::clone(&self.events);
        let hook_errors = self.hook_errors.clone();
        let mut event_rx = events.subscribe_receiver();

        crate::async_bridge::future_into_py(py, async move {
            hook_errors.clear();
            let output: PyAgentOutput = match agent.run(task).await {
                Ok(output) => output,
                Err(error) => {
                    hook_error_to_pyerr(&hook_errors)?;
                    return Err(PyRuntimeError::new_err(error));
                }
            };
            hook_error_to_pyerr(&hook_errors)?;
            events.flush().await;
            let collected_events = collect_run_events(&mut event_rx, submission_id);

            Python::attach(|py| {
                let output_val = serde_json::to_value(&output)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

                let dict = PyDict::new(py);
                if let serde_json::Value::Object(map) = output_val {
                    for (k, v) in map {
                        dict.set_item(&k, json_value_to_py(py, &v)?)?;
                    }
                }

                let events_py = crate::events::events_to_py_list(py, collected_events)?;
                dict.set_item("events", events_py)?;

                hook_error_to_pyerr(&hook_errors)?;
                Ok(dict.into_any().unbind())
            })
        })
    }

    /// Return an async event stream for this agent.
    pub fn event_stream(&self) -> PyEventStream {
        self.events.subscribe_py()
    }

    /// Run on *task* and return an async stream of partial outputs.
    pub fn run_stream<'py>(
        &self,
        py: Python<'py>,
        task: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let task = py_task_to_rust_task(task)?;
        let agent = Arc::clone(&self.agent);
        let hook_errors = self.hook_errors.clone();

        crate::async_bridge::future_into_py(py, async move {
            hook_errors.clear();
            let stream: AgentOutputStream = match agent.run_stream(task).await {
                Ok(stream) => stream,
                Err(error) => {
                    hook_error_to_pyerr(&hook_errors)?;
                    return Err(PyRuntimeError::new_err(error));
                }
            };
            hook_error_to_pyerr(&hook_errors)?;

            Python::attach(|py: Python<'_>| -> PyResult<Py<PyAny>> {
                PyRunStream {
                    stream: Arc::new(tokio::sync::Mutex::new(stream)),
                    hook_errors,
                }
                .into_pyobject(py)
                .map(|b| b.into_any().unbind())
            })
        })
    }
}

fn event_submission_id(event: &Event) -> Option<SubmissionId> {
    match event {
        Event::TaskStarted { sub_id, .. }
        | Event::TaskComplete { sub_id, .. }
        | Event::TaskError { sub_id, .. }
        | Event::ToolCallRequested { sub_id, .. }
        | Event::ToolCallCompleted { sub_id, .. }
        | Event::ToolCallFailed { sub_id, .. }
        | Event::TurnStarted { sub_id, .. }
        | Event::TurnCompleted { sub_id, .. }
        | Event::StreamChunk { sub_id, .. }
        | Event::StreamToolCall { sub_id, .. }
        | Event::StreamComplete { sub_id, .. } => Some(*sub_id),
        Event::PublishMessage { .. } | Event::NewTask { .. } | Event::SendMessage { .. } => None,
    }
}

fn collect_run_events(
    receiver: &mut tokio::sync::broadcast::Receiver<Event>,
    submission_id: SubmissionId,
) -> Vec<Event> {
    let mut collected = Vec::new();

    loop {
        match receiver.try_recv() {
            Ok(event) => {
                if event_submission_id(&event).is_some_and(|event_id| event_id == submission_id) {
                    collected.push(event);
                }
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => continue,
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => break,
            Err(tokio::sync::broadcast::error::TryRecvError::Closed) => break,
        }
    }

    collected
}

/// Handle to an actor-based agent running inside a `Runtime`.
/// Produced by `AgentBuilder.build_actor()`.
#[pyclass(name = "ActorAgentHandle")]
pub struct PyActorAgentHandle {
    send_fn: Arc<dyn Fn(Task) -> Result<(), String> + Send + Sync>,
}

#[pymethods]
impl PyActorAgentHandle {
    fn __repr__(&self) -> &str {
        "ActorAgentHandle(<running>)"
    }

    /// Send a task directly to the actor's mailbox. Returns a coroutine.
    pub fn send<'py>(&self, py: Python<'py>, prompt: String) -> PyResult<Bound<'py, PyAny>> {
        let send_fn = Arc::clone(&self.send_fn);
        crate::async_bridge::future_into_py(py, async move {
            let task = Task::new(prompt);
            send_fn(task).map_err(PyRuntimeError::new_err)?;
            Ok(Python::attach(|py| py.None()))
        })
    }
}

fn parse_image_mime(mime: &str) -> Option<ImageMime> {
    match mime.to_ascii_lowercase().as_str() {
        "jpeg" | "jpg" | "image/jpeg" => Some(ImageMime::JPEG),
        "png" | "image/png" => Some(ImageMime::PNG),
        "gif" | "image/gif" => Some(ImageMime::GIF),
        "webp" | "image/webp" => Some(ImageMime::WEBP),
        _ => None,
    }
}

fn py_task_to_rust_task(task_obj: &Bound<'_, PyAny>) -> PyResult<Task> {
    if let Ok(prompt) = task_obj.extract::<String>() {
        return Ok(Task::new(prompt));
    }

    let dict = task_obj.extract::<Bound<'_, PyDict>>().map_err(|_| {
        PyRuntimeError::new_err("task must be either str or dict-like Task payload")
    })?;

    let prompt_any = dict
        .get_item("prompt")?
        .ok_or_else(|| PyRuntimeError::new_err("task.prompt is required"))?;
    let prompt = prompt_any
        .extract::<String>()
        .map_err(|_| PyRuntimeError::new_err("task.prompt must be a string"))?;

    let mut task = Task::new(prompt);

    if let Some(system_prompt_any) = dict.get_item("system_prompt")?
        && !system_prompt_any.is_none()
    {
        let system_prompt = system_prompt_any
            .extract::<String>()
            .map_err(|_| PyRuntimeError::new_err("task.system_prompt must be a string"))?;
        task = task.with_system_prompt(system_prompt);
    }

    if let Some(image_any) = dict.get_item("image")?
        && !image_any.is_none()
    {
        let image = image_any
            .extract::<Bound<'_, PyDict>>()
            .map_err(|_| PyRuntimeError::new_err("task.image must be a dict"))?;
        let mime_any = image
            .get_item("mime")?
            .ok_or_else(|| PyRuntimeError::new_err("task.image.mime is required"))?;
        let data_any = image
            .get_item("data")?
            .ok_or_else(|| PyRuntimeError::new_err("task.image.data is required"))?;

        let mime_str = mime_any
            .extract::<String>()
            .map_err(|_| PyRuntimeError::new_err("task.image.mime must be a string"))?;
        let mime = parse_image_mime(&mime_str).ok_or_else(|| {
            PyRuntimeError::new_err(
                "unsupported task.image.mime, expected one of: jpeg|jpg|png|gif|webp",
            )
        })?;
        let data = data_any
            .extract::<Vec<u8>>()
            .map_err(|_| PyRuntimeError::new_err("task.image.data must be bytes"))?;

        task.image = Some((mime, data));
    }

    Ok(task)
}

/// Async iterator over `agent.run_stream(...)` outputs.
#[pyclass(name = "RunStream")]
pub struct PyRunStream {
    stream: Arc<tokio::sync::Mutex<AgentOutputStream>>,
    hook_errors: HookErrorState,
}

#[pymethods]
impl PyRunStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = Arc::clone(&self.stream);
        let hook_errors = self.hook_errors.clone();

        crate::async_bridge::future_into_py(py, async move {
            let next = {
                let mut guard = stream.lock().await;
                guard.next().await
            };

            match next {
                Some(Ok(output)) => Python::attach(|py| {
                    if output.done {
                        hook_error_to_pyerr(&hook_errors)?;
                    }
                    let val = serde_json::to_value(&output)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    json_value_to_py(py, &val)
                }),
                Some(Err(err)) => {
                    hook_error_to_pyerr(&hook_errors)?;
                    Err(PyRuntimeError::new_err(err.to_string()))
                }
                None => {
                    hook_error_to_pyerr(&hook_errors)?;
                    Err(pyo3::exceptions::PyStopAsyncIteration::new_err(
                        "stream ended",
                    ))
                }
            }
        })
    }
}
