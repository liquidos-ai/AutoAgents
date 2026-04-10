use crate::agent::executor::parse_executor_spec;
use crate::agent::py_agent::{
    AgentOutputStream, BuildActorResult, BuildDirectResult, HookErrorState, PyAgentDef,
    PyAgentOutput, PyExecutorBuildable, PyRunnable,
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

async fn resolve_built_agent(
    fut: BuildDirectResult,
    hook_errors: HookErrorState,
) -> PyResult<PyAgentHandle> {
    let (agent, event_stream) = fut.await?;
    hook_error_to_pyerr(&hook_errors)?;
    let events = Arc::new(PySharedEventStream::new(event_stream));
    Ok(PyAgentHandle {
        agent,
        events,
        hook_errors,
    })
}

async fn resolve_actor_handle(
    fut: BuildActorResult,
    hook_errors: HookErrorState,
) -> PyResult<PyActorAgentHandle> {
    let send_fn = fut.await?;
    hook_error_to_pyerr(&hook_errors)?;
    Ok(PyActorAgentHandle { send_fn })
}

async fn run_agent_task(
    agent: Arc<dyn PyRunnable>,
    events: Arc<PySharedEventStream>,
    hook_errors: HookErrorState,
    task: Task,
) -> PyResult<(PyAgentOutput, Vec<Event>)> {
    let submission_id = task.submission_id;
    hook_errors.clear();
    let mut event_rx = events.subscribe_receiver();

    let output = match agent.run(task).await {
        Ok(output) => output,
        Err(error) => {
            hook_error_to_pyerr(&hook_errors)?;
            return Err(PyRuntimeError::new_err(error));
        }
    };
    hook_error_to_pyerr(&hook_errors)?;
    events.flush().await;
    let collected_events = collect_run_events(&mut event_rx, submission_id);
    hook_error_to_pyerr(&hook_errors)?;

    Ok((output, collected_events))
}

async fn build_run_stream(
    agent: Arc<dyn PyRunnable>,
    hook_errors: HookErrorState,
    task: Task,
) -> PyResult<PyRunStream> {
    hook_errors.clear();
    let stream = match agent.run_stream(task).await {
        Ok(stream) => stream,
        Err(error) => {
            hook_error_to_pyerr(&hook_errors)?;
            return Err(PyRuntimeError::new_err(error));
        }
    };
    hook_error_to_pyerr(&hook_errors)?;

    Ok(PyRunStream {
        stream: Arc::new(tokio::sync::Mutex::new(stream)),
        hook_errors,
    })
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
            let handle = resolve_built_agent(fut, hook_errors).await?;
            Python::attach(|py: Python<'_>| -> PyResult<Py<PyAny>> {
                let result = handle.into_pyobject(py).map(|b| b.into_any().unbind())?;
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
        let runtime = self.actor_runtime()?;
        agent_def.task_locals = Some(pyo3_async_runtimes::tokio::get_current_locals(py)?);
        let fut = self
            .executor
            .build_actor(agent_def, llm, memory, runtime, self.topics.clone());

        crate::async_bridge::future_into_py(py, async move {
            let handle = resolve_actor_handle(fut, hook_errors).await?;
            Python::attach(|py: Python<'_>| -> PyResult<Py<PyAny>> {
                handle.into_pyobject(py).map(|b| b.into_any().unbind())
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

    fn actor_runtime(&self) -> PyResult<Arc<dyn Runtime>> {
        self.runtime
            .clone()
            .ok_or_else(|| PyRuntimeError::new_err("runtime is required for build_actor()"))
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
        let agent = Arc::clone(&self.agent);
        let events = Arc::clone(&self.events);
        let hook_errors = self.hook_errors.clone();

        crate::async_bridge::future_into_py(py, async move {
            let (output, collected_events) =
                run_agent_task(agent, events, hook_errors.clone(), task).await?;

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
            let stream = build_run_stream(agent, hook_errors, task).await?;
            Python::attach(|py: Python<'_>| -> PyResult<Py<PyAny>> {
                stream.into_pyobject(py).map(|b| b.into_any().unbind())
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
        | Event::CodeExecutionStarted { sub_id, .. }
        | Event::CodeExecutionConsole { sub_id, .. }
        | Event::CodeExecutionCompleted { sub_id, .. }
        | Event::CodeExecutionFailed { sub_id, .. }
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

impl PyActorAgentHandle {
    fn send_task(&self, task: Task) -> PyResult<()> {
        (self.send_fn)(task).map_err(PyRuntimeError::new_err)
    }
}

#[pymethods]
impl PyActorAgentHandle {
    fn __repr__(&self) -> &str {
        "ActorAgentHandle(<running>)"
    }

    /// Send a task directly to the actor's mailbox. Returns a coroutine.
    pub fn send<'py>(&self, py: Python<'py>, prompt: String) -> PyResult<Bound<'py, PyAny>> {
        let handle = PyActorAgentHandle {
            send_fn: Arc::clone(&self.send_fn),
        };
        crate::async_bridge::future_into_py(py, async move {
            let task = Task::new(prompt);
            handle.send_task(task)?;
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

impl PyRunStream {
    async fn next_output(&self) -> PyResult<Option<PyAgentOutput>> {
        let next = {
            let mut guard = self.stream.lock().await;
            guard.next().await
        };

        match next {
            Some(Ok(output)) => {
                if output.done {
                    hook_error_to_pyerr(&self.hook_errors)?;
                }
                Ok(Some(output))
            }
            Some(Err(err)) => {
                hook_error_to_pyerr(&self.hook_errors)?;
                Err(PyRuntimeError::new_err(err.to_string()))
            }
            None => {
                hook_error_to_pyerr(&self.hook_errors)?;
                Ok(None)
            }
        }
    }
}

#[pymethods]
impl PyRunStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let run_stream = PyRunStream {
            stream: Arc::clone(&self.stream),
            hook_errors: self.hook_errors.clone(),
        };

        crate::async_bridge::future_into_py(py, async move {
            match run_stream.next_output().await? {
                Some(output) => Python::attach(|py| {
                    let val = serde_json::to_value(&output)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    json_value_to_py(py, &val)
                }),
                None => Err(pyo3::exceptions::PyStopAsyncIteration::new_err(
                    "stream ended",
                )),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::py_agent::{ActorSendFn, BuildActorResult, BuildDirectResult};
    use autoagents_core::agent::memory::MemoryProvider;
    use autoagents_core::agent::memory::SlidingWindowMemory;
    use autoagents_core::error::Error as CoreError;
    use autoagents_core::runtime::SingleThreadedRuntime;
    use autoagents_core::utils::BoxEventStream;
    use autoagents_llm::chat::{
        ChatMessage, ChatProvider, ChatResponse, StructuredOutputFormat, Tool,
    };
    use autoagents_llm::completion::{CompletionProvider, CompletionRequest, CompletionResponse};
    use autoagents_llm::embedding::EmbeddingProvider;
    use autoagents_llm::error::LLMError;
    use autoagents_llm::models::ModelsProvider;
    use autoagents_llm::{HasConfig, NoConfig, ToolCall, async_trait};
    use autoagents_protocol::{ActorID, Event, SubmissionId};
    use futures::stream;
    use serde_json::json;
    use std::fmt;
    use std::future::Future;

    fn init_runtime_bridge() {
        Python::initialize();
        let runtime = crate::runtime::get_runtime().expect("shared runtime should initialize");
        let _ = pyo3_async_runtimes::tokio::init_with_runtime(runtime);
    }

    fn block_on_test<T>(future: impl Future<Output = T>) -> T {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime should build")
            .block_on(future)
    }

    #[derive(Debug)]
    struct MockChatResponse;

    impl fmt::Display for MockChatResponse {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("mock")
        }
    }

    impl ChatResponse for MockChatResponse {
        fn text(&self) -> Option<String> {
            Some("mock".to_string())
        }

        fn tool_calls(&self) -> Option<Vec<ToolCall>> {
            None
        }
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
            Ok(Box::new(MockChatResponse))
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
            Ok(input.into_iter().map(|_| vec![1.0]).collect())
        }
    }

    #[async_trait]
    impl ModelsProvider for MockLLMProvider {}

    impl autoagents_llm::LLMProvider for MockLLMProvider {}

    impl HasConfig for MockLLMProvider {
        type Config = NoConfig;
    }

    struct StubExecutor;

    impl PyExecutorBuildable for StubExecutor {
        fn build_direct(
            &self,
            _agent_def: PyAgentDef,
            _llm: Arc<dyn autoagents_llm::LLMProvider>,
            _memory: Box<dyn MemoryProvider>,
        ) -> BuildDirectResult {
            Box::pin(async { unreachable!("build_direct is not used in these tests") })
        }

        fn build_actor(
            &self,
            _agent_def: PyAgentDef,
            _llm: Arc<dyn autoagents_llm::LLMProvider>,
            _memory: Box<dyn MemoryProvider>,
            _runtime: Arc<dyn Runtime>,
            _topics: Vec<String>,
        ) -> BuildActorResult {
            Box::pin(async { unreachable!("build_actor is not used in these tests") })
        }
    }

    fn sample_agent_def() -> PyAgentDef {
        PyAgentDef {
            name: "planner".to_string(),
            description: "plans work".to_string(),
            tools: Vec::new(),
            output_schema: Some(json!({"type": "object"})),
            hooks: None,
            task_locals: None,
            hook_errors: HookErrorState::default(),
        }
    }

    #[derive(Clone)]
    struct StubRunnable {
        run_output: PyAgentOutput,
        run_error: Option<String>,
        stream_outputs: Vec<Result<PyAgentOutput, String>>,
        stream_error: Option<String>,
    }

    #[async_trait]
    impl PyRunnable for StubRunnable {
        async fn run(&self, _task: Task) -> Result<PyAgentOutput, String> {
            if let Some(error) = &self.run_error {
                return Err(error.clone());
            }
            Ok(self.run_output.clone())
        }

        async fn run_stream(&self, _task: Task) -> Result<AgentOutputStream, String> {
            if let Some(error) = &self.stream_error {
                return Err(error.clone());
            }
            let items = self
                .stream_outputs
                .clone()
                .into_iter()
                .map(|item| item.map_err(CoreError::CustomError));
            Ok(Box::pin(stream::iter(items)))
        }
    }

    struct BuildableStub {
        runnable: Arc<StubRunnable>,
        actor_error: Option<String>,
        direct_hook_error: Option<String>,
        actor_hook_error: Option<String>,
    }

    impl PyExecutorBuildable for BuildableStub {
        fn build_direct(
            &self,
            agent_def: PyAgentDef,
            _llm: Arc<dyn autoagents_llm::LLMProvider>,
            _memory: Box<dyn MemoryProvider>,
        ) -> BuildDirectResult {
            let runnable = Arc::clone(&self.runnable) as Arc<dyn PyRunnable>;
            let hook_error = self.direct_hook_error.clone();
            Box::pin(async move {
                if let Some(error) = hook_error {
                    agent_def.hook_errors.record(error);
                }
                let event_stream: BoxEventStream<Event> = Box::pin(stream::empty());
                Ok((runnable, event_stream))
            })
        }

        fn build_actor(
            &self,
            agent_def: PyAgentDef,
            _llm: Arc<dyn autoagents_llm::LLMProvider>,
            _memory: Box<dyn MemoryProvider>,
            _runtime: Arc<dyn Runtime>,
            _topics: Vec<String>,
        ) -> BuildActorResult {
            let actor_error = self.actor_error.clone();
            let hook_error = self.actor_hook_error.clone();
            Box::pin(async move {
                if let Some(error) = hook_error {
                    agent_def.hook_errors.record(error);
                }
                let send_fn: ActorSendFn = match actor_error {
                    Some(error) => Arc::new(move |_task: Task| Err(error.clone())),
                    None => Arc::new(|_task: Task| Ok(())),
                };
                Ok(send_fn)
            })
        }
    }

    fn configured_builder(
        executor: Box<dyn PyExecutorBuildable>,
        include_runtime: bool,
    ) -> PyAgentBuilder {
        PyAgentBuilder {
            executor,
            agent_def: sample_agent_def(),
            llm: Some(Arc::new(MockLLMProvider)),
            memory: Some(Box::new(SlidingWindowMemory::new(4))),
            runtime: include_runtime.then(|| SingleThreadedRuntime::new(None) as Arc<dyn Runtime>),
            topics: vec!["tasks".to_string()],
        }
    }

    #[test]
    fn builder_setters_and_inputs_clone_expected_state() {
        init_runtime_bridge();
        Python::attach(|py| {
            let builder = Py::new(
                py,
                PyAgentBuilder {
                    executor: Box::new(StubExecutor),
                    agent_def: sample_agent_def(),
                    llm: None,
                    memory: None,
                    runtime: None,
                    topics: Vec::new(),
                },
            )
            .expect("builder should create");

            let provider = Py::new(
                py,
                crate::llm::builder::PyLLMProvider::new(Arc::new(MockLLMProvider)),
            )
            .expect("provider should create");
            let memory =
                Py::new(py, crate::memory::sliding_window_memory(4)).expect("memory should create");
            let runtime =
                Py::new(py, PySingleThreadedRuntime::new()).expect("runtime should create");

            {
                let mut builder_ref = builder.borrow_mut(py);
                builder_ref = PyAgentBuilder::llm(builder_ref, provider.bind(py).as_any())
                    .expect("llm should set");
                builder_ref = PyAgentBuilder::memory(builder_ref, memory.borrow(py));
                builder_ref = PyAgentBuilder::runtime(builder_ref, runtime.borrow(py));
                builder_ref = PyAgentBuilder::subscribe(builder_ref, "tasks".to_string());
                let _builder_ref = PyAgentBuilder::subscribe(builder_ref, "alerts".to_string());
            }

            let builder_ref = builder.borrow(py);
            assert!(builder_ref.llm.is_some());
            assert!(builder_ref.memory.is_some());
            assert!(builder_ref.runtime.is_some());
            assert_eq!(
                builder_ref.topics,
                vec!["tasks".to_string(), "alerts".to_string()]
            );

            let (_llm, memory, agent_def) =
                builder_ref.build_inputs().expect("inputs should build");
            assert_eq!(memory.size(), 0);
            assert_eq!(agent_def.name, "planner");
            assert_eq!(agent_def.description, "plans work");
            assert_eq!(agent_def.output_schema, Some(json!({"type": "object"})));

            let empty_builder = PyAgentBuilder {
                executor: Box::new(StubExecutor),
                agent_def: sample_agent_def(),
                llm: None,
                memory: None,
                runtime: None,
                topics: Vec::new(),
            };
            let err = match empty_builder.build_inputs() {
                Ok(_) => panic!("missing llm should fail"),
                Err(err) => err,
            };
            assert!(err.to_string().contains("LLM provider is required"));
        });
    }

    #[test]
    fn task_and_event_helpers_cover_expected_branches() {
        init_runtime_bridge();
        assert!(matches!(parse_image_mime("jpeg"), Some(ImageMime::JPEG)));
        assert!(matches!(
            parse_image_mime("image/png"),
            Some(ImageMime::PNG)
        ));
        assert!(parse_image_mime("svg").is_none());

        let errors = HookErrorState::default();
        assert!(hook_error_to_pyerr(&errors).is_ok());
        errors.record("hook exploded");
        let err = hook_error_to_pyerr(&errors).expect_err("stored hook error should surface");
        assert!(err.to_string().contains("hook exploded"));

        Python::attach(|py| {
            let string_task = "inspect".into_pyobject(py).expect("string should convert");
            let task =
                py_task_to_rust_task(string_task.as_any()).expect("string task should parse");
            assert_eq!(task.prompt, "inspect");

            let dict_task = json_value_to_py(
                py,
                &json!({
                    "prompt": "image task",
                    "system_prompt": "system",
                    "image": {
                        "mime": "png",
                        "data": [137, 80, 78, 71]
                    }
                }),
            )
            .expect("dict should convert");
            let task = py_task_to_rust_task(dict_task.bind(py)).expect("dict task should parse");
            assert_eq!(task.prompt, "image task");
            assert_eq!(task.system_prompt.as_deref(), Some("system"));
            assert!(matches!(task.image, Some((ImageMime::PNG, _))));

            let err = py_task_to_rust_task(
                json_value_to_py(
                    py,
                    &json!({"prompt": "bad", "image": {"mime": "svg", "data": [1]}}),
                )
                .expect("dict should convert")
                .bind(py),
            )
            .expect_err("unsupported mime should fail");
            assert!(err.to_string().contains("unsupported task.image.mime"));
        });

        let sub_id = SubmissionId::from_u128(1);
        let other_sub_id = SubmissionId::from_u128(2);
        let actor_id = ActorID::from_u128(3);
        let (tx, mut rx) = tokio::sync::broadcast::channel(8);
        tx.send(Event::TaskStarted {
            sub_id,
            actor_id,
            actor_name: "planner".to_string(),
            task_description: "run".to_string(),
        })
        .expect("event should send");
        tx.send(Event::TaskComplete {
            sub_id: other_sub_id,
            actor_id,
            actor_name: "planner".to_string(),
            result: "done".to_string(),
        })
        .expect("event should send");
        tx.send(Event::NewTask {
            actor_id,
            task: Task::new("new"),
        })
        .expect("event should send");

        let collected = collect_run_events(&mut rx, sub_id);
        assert_eq!(collected.len(), 1);
        assert!(matches!(
            event_submission_id(&collected[0]),
            Some(id) if id == sub_id
        ));
        assert!(
            event_submission_id(&Event::NewTask {
                actor_id,
                task: Task::new("new"),
            })
            .is_none()
        );
    }

    #[test]
    fn run_stream_and_actor_handle_helpers_cover_success_and_error_paths() {
        let run_stream = PyRunStream {
            stream: Arc::new(tokio::sync::Mutex::new(Box::pin(stream::iter(vec![
                Ok(PyAgentOutput {
                    response: "partial".to_string(),
                    tool_calls: Vec::new(),
                    executions: Vec::new(),
                    done: false,
                }),
                Ok(PyAgentOutput {
                    response: "final".to_string(),
                    tool_calls: Vec::new(),
                    executions: Vec::new(),
                    done: true,
                }),
            ])))),
            hook_errors: HookErrorState::default(),
        };

        let first = block_on_test(run_stream.next_output())
            .expect("first output should succeed")
            .expect("stream should yield first item");
        assert_eq!(first.response, "partial");
        assert!(!first.done);

        let second = block_on_test(run_stream.next_output())
            .expect("second output should succeed")
            .expect("stream should yield second item");
        assert_eq!(second.response, "final");
        assert!(second.done);

        assert!(
            block_on_test(run_stream.next_output())
                .expect("completed stream should not error")
                .is_none()
        );

        let err_stream = PyRunStream {
            stream: Arc::new(tokio::sync::Mutex::new(Box::pin(stream::iter(vec![Err(
                CoreError::CustomError("boom".to_string()),
            )])))),
            hook_errors: HookErrorState::default(),
        };
        let err = block_on_test(err_stream.next_output()).expect_err("stream error should surface");
        assert!(err.to_string().contains("boom"));

        let actor = PyActorAgentHandle {
            send_fn: Arc::new(|task: Task| {
                if task.prompt == "ok" {
                    Ok(())
                } else {
                    Err("unexpected task".to_string())
                }
            }),
        };
        assert_eq!(actor.__repr__(), "ActorAgentHandle(<running>)");
        actor
            .send_task(Task::new("ok"))
            .expect("send_task should accept valid prompts");

        let err_actor = PyActorAgentHandle {
            send_fn: Arc::new(|_task: Task| Err("mailbox closed".to_string())),
        };
        let err = err_actor
            .send_task(Task::new("bad"))
            .expect_err("send errors should surface");
        assert!(err.to_string().contains("mailbox closed"));
    }

    #[test]
    fn builder_helpers_cover_direct_build_run_and_stream_paths() {
        let runnable = Arc::new(StubRunnable {
            run_output: PyAgentOutput {
                response: "done".to_string(),
                tool_calls: Vec::new(),
                executions: Vec::new(),
                done: true,
            },
            run_error: None,
            stream_outputs: vec![
                Ok(PyAgentOutput {
                    response: "partial".to_string(),
                    tool_calls: Vec::new(),
                    executions: Vec::new(),
                    done: false,
                }),
                Ok(PyAgentOutput {
                    response: "final".to_string(),
                    tool_calls: Vec::new(),
                    executions: Vec::new(),
                    done: true,
                }),
            ],
            stream_error: None,
        });
        let builder = configured_builder(
            Box::new(BuildableStub {
                runnable,
                actor_error: None,
                direct_hook_error: None,
                actor_hook_error: None,
            }),
            true,
        );

        let (llm, memory, agent_def) = builder.build_inputs().expect("inputs should build");
        let hook_errors = agent_def.hook_errors.clone();
        let handle = block_on_test(resolve_built_agent(
            builder.executor.build_direct(agent_def, llm, memory),
            hook_errors.clone(),
        ))
        .expect("direct agent should build");
        assert_eq!(handle.__repr__(), "AgentHandle(<built>)");
        let _event_stream = handle.event_stream();

        let (output, events) = block_on_test(run_agent_task(
            Arc::clone(&handle.agent),
            Arc::clone(&handle.events),
            handle.hook_errors.clone(),
            Task::new("inspect"),
        ))
        .expect("run should succeed");
        assert_eq!(output.response, "done");
        assert!(output.done);
        assert!(events.is_empty());

        let run_stream = block_on_test(build_run_stream(
            Arc::clone(&handle.agent),
            handle.hook_errors.clone(),
            Task::new("inspect"),
        ))
        .expect("run stream should build");
        let first = block_on_test(run_stream.next_output())
            .expect("first chunk should succeed")
            .expect("stream should yield first chunk");
        assert_eq!(first.response, "partial");
        let second = block_on_test(run_stream.next_output())
            .expect("second chunk should succeed")
            .expect("stream should yield second chunk");
        assert_eq!(second.response, "final");
        assert!(
            block_on_test(run_stream.next_output())
                .expect("completed stream should not error")
                .is_none()
        );

        let failing_run_builder = configured_builder(
            Box::new(BuildableStub {
                runnable: Arc::new(StubRunnable {
                    run_output: PyAgentOutput {
                        response: String::default(),
                        tool_calls: Vec::new(),
                        executions: Vec::new(),
                        done: true,
                    },
                    run_error: Some("run failed".to_string()),
                    stream_outputs: Vec::new(),
                    stream_error: None,
                }),
                actor_error: None,
                direct_hook_error: None,
                actor_hook_error: None,
            }),
            true,
        );
        let (llm, memory, agent_def) = failing_run_builder
            .build_inputs()
            .expect("failing run builder inputs should build");
        let hook_errors = agent_def.hook_errors.clone();
        let failing_handle = block_on_test(resolve_built_agent(
            failing_run_builder
                .executor
                .build_direct(agent_def, llm, memory),
            hook_errors,
        ))
        .expect("failing run builder should still build handle");
        let err = block_on_test(run_agent_task(
            Arc::clone(&failing_handle.agent),
            Arc::clone(&failing_handle.events),
            failing_handle.hook_errors.clone(),
            Task::new("broken"),
        ))
        .expect_err("run errors should surface");
        assert!(err.to_string().contains("run failed"));

        let failing_stream_builder = configured_builder(
            Box::new(BuildableStub {
                runnable: Arc::new(StubRunnable {
                    run_output: PyAgentOutput {
                        response: "unused".to_string(),
                        tool_calls: Vec::new(),
                        executions: Vec::new(),
                        done: true,
                    },
                    run_error: None,
                    stream_outputs: Vec::new(),
                    stream_error: Some("stream unavailable".to_string()),
                }),
                actor_error: None,
                direct_hook_error: None,
                actor_hook_error: None,
            }),
            true,
        );
        let (llm, memory, agent_def) = failing_stream_builder
            .build_inputs()
            .expect("failing stream builder inputs should build");
        let hook_errors = agent_def.hook_errors.clone();
        let failing_stream_handle = block_on_test(resolve_built_agent(
            failing_stream_builder
                .executor
                .build_direct(agent_def, llm, memory),
            hook_errors,
        ))
        .expect("failing stream builder should still build handle");
        let err = match block_on_test(build_run_stream(
            Arc::clone(&failing_stream_handle.agent),
            failing_stream_handle.hook_errors.clone(),
            Task::new("broken"),
        )) {
            Ok(_) => panic!("run_stream errors should surface"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("stream unavailable"));
    }

    #[test]
    fn builder_helpers_cover_actor_build_and_hook_errors() {
        Python::initialize();
        let runnable = Arc::new(StubRunnable {
            run_output: PyAgentOutput {
                response: "unused".to_string(),
                tool_calls: Vec::new(),
                executions: Vec::new(),
                done: true,
            },
            run_error: None,
            stream_outputs: Vec::new(),
            stream_error: None,
        });

        let missing_runtime = configured_builder(
            Box::new(BuildableStub {
                runnable: Arc::clone(&runnable),
                actor_error: None,
                direct_hook_error: None,
                actor_hook_error: None,
            }),
            false,
        );
        let err = match missing_runtime.actor_runtime() {
            Ok(_) => panic!("missing runtime should fail"),
            Err(err) => err,
        }
        .to_string();
        assert!(err.contains("runtime is required"));

        let actor_builder = configured_builder(
            Box::new(BuildableStub {
                runnable: Arc::clone(&runnable),
                actor_error: None,
                direct_hook_error: None,
                actor_hook_error: None,
            }),
            true,
        );
        let (llm, memory, agent_def) = actor_builder
            .build_inputs()
            .expect("actor builder inputs should build");
        let hook_errors = agent_def.hook_errors.clone();
        let actor_handle = block_on_test(resolve_actor_handle(
            actor_builder.executor.build_actor(
                agent_def,
                llm,
                memory,
                actor_builder.runtime.clone().expect("runtime should exist"),
                actor_builder.topics.clone(),
            ),
            hook_errors,
        ))
        .expect("actor handle should build");
        assert_eq!(actor_handle.__repr__(), "ActorAgentHandle(<running>)");
        actor_handle
            .send_task(Task::new("queued"))
            .expect("send should succeed");

        let actor_error_builder = configured_builder(
            Box::new(BuildableStub {
                runnable: Arc::clone(&runnable),
                actor_error: Some("mailbox closed".to_string()),
                direct_hook_error: None,
                actor_hook_error: None,
            }),
            true,
        );
        let (llm, memory, agent_def) = actor_error_builder
            .build_inputs()
            .expect("actor error builder inputs should build");
        let hook_errors = agent_def.hook_errors.clone();
        let actor_handle = block_on_test(resolve_actor_handle(
            actor_error_builder.executor.build_actor(
                agent_def,
                llm,
                memory,
                actor_error_builder
                    .runtime
                    .clone()
                    .expect("runtime should exist"),
                actor_error_builder.topics.clone(),
            ),
            hook_errors,
        ))
        .expect("actor handle should still build");
        let err = actor_handle
            .send_task(Task::new("queued"))
            .expect_err("send errors should surface");
        assert!(err.to_string().contains("mailbox closed"));

        let hook_error_builder = configured_builder(
            Box::new(BuildableStub {
                runnable,
                actor_error: None,
                direct_hook_error: Some("direct hook failed".to_string()),
                actor_hook_error: Some("actor hook failed".to_string()),
            }),
            true,
        );

        let (llm, memory, agent_def) = hook_error_builder
            .build_inputs()
            .expect("hook error builder inputs should build");
        let direct_hook_errors = agent_def.hook_errors.clone();
        let direct_err = match block_on_test(resolve_built_agent(
            hook_error_builder
                .executor
                .build_direct(agent_def, llm, memory),
            direct_hook_errors,
        )) {
            Ok(_) => panic!("direct build hook errors should surface"),
            Err(err) => err,
        };
        assert!(direct_err.to_string().contains("direct hook failed"));

        let (llm, memory, agent_def) = hook_error_builder
            .build_inputs()
            .expect("hook error builder inputs should build again");
        let actor_hook_errors = agent_def.hook_errors.clone();
        let actor_err = match block_on_test(resolve_actor_handle(
            hook_error_builder.executor.build_actor(
                agent_def,
                llm,
                memory,
                hook_error_builder
                    .runtime
                    .clone()
                    .expect("runtime should exist"),
                hook_error_builder.topics.clone(),
            ),
            actor_hook_errors,
        )) {
            Ok(_) => panic!("actor build hook errors should surface"),
            Err(err) => err,
        };
        assert!(actor_err.to_string().contains("actor hook failed"));
    }
}
