use crate::agent::py_agent::{
    ActorSendFn, BuildActorResult, BuildDirectResult, PyAgentDef, PyAgentOutput,
    PyExecutorBuildable, PyRunnable, call_hook_method_async, call_hook_method_sync, context_to_py,
    task_to_py,
};
use crate::convert::py_any_to_json_value;
use crate::tool::PyTool;
use autoagents_core::actor::Topic;
use autoagents_core::agent::error::RunnableAgentError;
use autoagents_core::agent::memory::MemoryProvider;
use autoagents_core::agent::prebuilt::executor::{BasicAgent, ReActAgent};
use autoagents_core::agent::task::Task;
use autoagents_core::agent::{
    ActorAgent, ActorAgentHandle, AgentBuilder, AgentDeriveT, AgentExecutor, AgentHooks, Context,
    DirectAgent, ExecutorConfig, HookOutcome,
};
use autoagents_core::runtime::Runtime;
use autoagents_core::tool::{ToolCallResult, ToolT, shared_tools_to_boxes};
use autoagents_llm::LLMProvider;
use autoagents_protocol::Event;
use futures::Stream;
use pyo3::exceptions::{PyRuntimeError, PyStopAsyncIteration};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use pyo3_async_runtimes::TaskLocals;
use serde_json::Value;
use std::pin::Pin;
use std::sync::Arc;

const CUSTOM_EXECUTOR_TURN_INDEX: usize = 1;

#[derive(Debug)]
pub struct PyCustomExecutorError(pub String);

impl std::fmt::Display for PyCustomExecutorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PyCustomExecutorError {}

impl PyCustomExecutorError {
    fn runtime(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl From<PyCustomExecutorError> for RunnableAgentError {
    fn from(value: PyCustomExecutorError) -> Self {
        RunnableAgentError::ExecutorError(value.0)
    }
}

async fn send_custom_event(tx_event: &Option<tokio::sync::mpsc::Sender<Event>>, event: Event) {
    if let Some(tx) = tx_event {
        let _ = tx.send(event).await;
    }
}

async fn emit_custom_task_started(task: &Task, context: &Context) {
    let tx_event = context.tx().ok();
    send_custom_event(
        &tx_event,
        Event::TaskStarted {
            sub_id: task.submission_id,
            actor_id: context.config().id,
            actor_name: context.config().name.clone(),
            task_description: task.prompt.clone(),
        },
    )
    .await;
}

async fn emit_custom_turn_started(task: &Task, context: &Context, max_turns: usize) {
    let tx_event = context.tx().ok();
    send_custom_event(
        &tx_event,
        Event::TurnStarted {
            sub_id: task.submission_id,
            actor_id: context.config().id,
            turn_number: CUSTOM_EXECUTOR_TURN_INDEX,
            max_turns,
        },
    )
    .await;
}

async fn emit_custom_turn_completed(task: &Task, context: &Context) {
    let tx_event = context.tx().ok();
    send_custom_event(
        &tx_event,
        Event::TurnCompleted {
            sub_id: task.submission_id,
            actor_id: context.config().id,
            turn_number: CUSTOM_EXECUTOR_TURN_INDEX,
            final_turn: true,
        },
    )
    .await;
}

fn task_completed_value(output: &PyAgentOutput) -> Result<Value, PyCustomExecutorError> {
    serde_json::to_value(output)
        .map_err(|e| PyCustomExecutorError::runtime(format!("serialize executor output: {e}")))
}

async fn emit_custom_task_completed(
    task: &Task,
    context: &Context,
    output: &PyAgentOutput,
) -> Result<(), PyCustomExecutorError> {
    let tx_event = context.tx().ok();
    let output_value = task_completed_value(output)?;
    let result = serde_json::to_string_pretty(&output_value)
        .map_err(|e| PyCustomExecutorError::runtime(format!("serialize executor output: {e}")))?;
    send_custom_event(
        &tx_event,
        Event::TaskComplete {
            sub_id: task.submission_id,
            actor_id: context.config().id,
            actor_name: context.config().name.clone(),
            result,
        },
    )
    .await;
    Ok(())
}

async fn emit_custom_stream_complete(task: &Task, context: &Context) {
    let tx_event = context.tx().ok();
    send_custom_event(
        &tx_event,
        Event::StreamComplete {
            sub_id: task.submission_id,
        },
    )
    .await;
}

async fn emit_custom_task_error(task: &Task, context: &Context, error: &PyCustomExecutorError) {
    let tx_event = context.tx().ok();
    send_custom_event(
        &tx_event,
        Event::TaskError {
            sub_id: task.submission_id,
            actor_id: context.config().id,
            error: error.to_string(),
        },
    )
    .await;
}

struct PyInjectedExecutor {
    agent_def: PyAgentDef,
    executor_impl: Py<PyAny>,
    max_turns: usize,
    task_locals: Option<TaskLocals>,
}

impl Clone for PyInjectedExecutor {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            agent_def: self.agent_def.clone(),
            executor_impl: self.executor_impl.clone_ref(py),
            max_turns: self.max_turns,
            task_locals: self.task_locals.clone(),
        })
    }
}

impl std::fmt::Debug for PyInjectedExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyInjectedExecutor")
            .field("name", &self.agent_def.name)
            .field("max_turns", &self.max_turns)
            .finish()
    }
}

#[async_trait::async_trait]
impl AgentDeriveT for PyInjectedExecutor {
    type Output = PyAgentOutput;

    fn description(&self) -> &str {
        &self.agent_def.description
    }

    fn output_schema(&self) -> Option<Value> {
        self.agent_def.output_schema.clone()
    }

    fn name(&self) -> &str {
        &self.agent_def.name
    }

    fn tools(&self) -> Vec<Box<dyn ToolT>> {
        shared_tools_to_boxes(&self.agent_def.tools)
    }
}

#[async_trait::async_trait]
impl AgentHooks for PyInjectedExecutor {
    async fn on_agent_create(&self) {
        self.agent_def.on_agent_create().await;
    }

    async fn on_run_start(&self, task: &Task, ctx: &Context) -> HookOutcome {
        self.agent_def.on_run_start(task, ctx).await
    }

    async fn on_run_complete(&self, task: &Task, result: &Self::Output, ctx: &Context) {
        self.agent_def.on_run_complete(task, result, ctx).await;
    }

    async fn on_turn_start(&self, turn_index: usize, ctx: &Context) {
        self.agent_def.on_turn_start(turn_index, ctx).await;
    }

    async fn on_turn_complete(&self, turn_index: usize, ctx: &Context) {
        self.agent_def.on_turn_complete(turn_index, ctx).await;
    }

    async fn on_tool_call(
        &self,
        tool_call: &autoagents_llm::ToolCall,
        ctx: &Context,
    ) -> HookOutcome {
        self.agent_def.on_tool_call(tool_call, ctx).await
    }

    async fn on_tool_start(&self, tool_call: &autoagents_llm::ToolCall, ctx: &Context) {
        self.agent_def.on_tool_start(tool_call, ctx).await;
    }

    async fn on_tool_result(
        &self,
        tool_call: &autoagents_llm::ToolCall,
        result: &ToolCallResult,
        ctx: &Context,
    ) {
        self.agent_def.on_tool_result(tool_call, result, ctx).await;
    }

    async fn on_tool_error(&self, tool_call: &autoagents_llm::ToolCall, err: Value, ctx: &Context) {
        self.agent_def.on_tool_error(tool_call, err, ctx).await;
    }

    async fn on_agent_shutdown(&self) {
        self.agent_def.on_agent_shutdown().await;
    }
}

#[async_trait::async_trait]
impl AgentExecutor for PyInjectedExecutor {
    type Output = PyAgentOutput;
    type Error = PyCustomExecutorError;

    fn config(&self) -> ExecutorConfig {
        ExecutorConfig {
            max_turns: self.max_turns.max(1),
        }
    }

    async fn execute(
        &self,
        task: &Task,
        context: Arc<Context>,
    ) -> Result<Self::Output, Self::Error> {
        emit_custom_task_started(task, &context).await;
        emit_custom_turn_started(task, &context, self.max_turns).await;
        self.agent_def
            .on_turn_start(CUSTOM_EXECUTOR_TURN_INDEX, &context)
            .await;

        let result = call_executor_output(
            &self.executor_impl,
            self.task_locals.as_ref(),
            "execute",
            task,
            &context,
        )
        .await
        .map_err(|e| {
            PyCustomExecutorError::runtime(format!("custom executor execute() failed: {e}"))
        });

        match result {
            Ok(output) => {
                self.agent_def
                    .on_turn_complete(CUSTOM_EXECUTOR_TURN_INDEX, &context)
                    .await;
                emit_custom_turn_completed(task, &context).await;
                emit_custom_task_completed(task, &context, &output).await?;
                Ok(output)
            }
            Err(error) => {
                emit_custom_task_error(task, &context, &error).await;
                Err(error)
            }
        }
    }

    async fn execute_stream(
        &self,
        task: &Task,
        context: Arc<Context>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Self::Output, Self::Error>> + Send>>, Self::Error>
    {
        let task_locals = self.task_locals.clone();
        let max_turns = self.max_turns;
        let task = task.clone();
        let context = Arc::clone(&context);
        let agent_def = self.agent_def.clone();

        emit_custom_task_started(&task, &context).await;
        emit_custom_turn_started(&task, &context, max_turns).await;
        agent_def
            .on_turn_start(CUSTOM_EXECUTOR_TURN_INDEX, &context)
            .await;

        let iter_obj = call_python_executor_method(
            &self.executor_impl,
            task_locals.as_ref(),
            "execute_stream",
            &task,
            &context,
        )
        .await
        .map_err(|e| PyCustomExecutorError::runtime(format!("execute_stream() call failed: {e}")));

        let iter_obj = match iter_obj {
            Ok(iter_obj) => iter_obj,
            Err(error) => {
                emit_custom_task_error(&task, &context, &error).await;
                return Err(error);
            }
        };

        let stream = futures::stream::try_unfold(
            (
                iter_obj,
                task_locals,
                task,
                context,
                agent_def,
                None::<PyAgentOutput>,
                false,
            ),
            |(iter_obj, task_locals, task, context, agent_def, last_output, completed)| async move {
                if completed {
                    return Ok(None);
                }

                match next_stream_output(&iter_obj, task_locals.as_ref()).await {
                    Ok(Some(item)) => {
                        let stream_complete = item.done;
                        let next_last_output = Some(item.clone());

                        if stream_complete {
                            agent_def
                                .on_turn_complete(CUSTOM_EXECUTOR_TURN_INDEX, &context)
                                .await;
                            emit_custom_turn_completed(&task, &context).await;
                            emit_custom_stream_complete(&task, &context).await;
                            emit_custom_task_completed(&task, &context, &item).await?;
                        }

                        Ok(Some((
                            item,
                            (
                                iter_obj,
                                task_locals,
                                task,
                                context,
                                agent_def,
                                next_last_output,
                                stream_complete,
                            ),
                        )))
                    }
                    Ok(None) => {
                        let final_output = last_output.unwrap_or(PyAgentOutput {
                            response: String::new(),
                            tool_calls: Vec::new(),
                            done: true,
                        });
                        let final_output = PyAgentOutput {
                            done: true,
                            ..final_output
                        };

                        agent_def
                            .on_turn_complete(CUSTOM_EXECUTOR_TURN_INDEX, &context)
                            .await;
                        emit_custom_turn_completed(&task, &context).await;
                        emit_custom_stream_complete(&task, &context).await;
                        emit_custom_task_completed(&task, &context, &final_output).await?;
                        Ok(None)
                    }
                    Err(err) => {
                        let error = PyCustomExecutorError::runtime(err);
                        emit_custom_task_error(&task, &context, &error).await;
                        Err(error)
                    }
                }
            },
        );

        Ok(Box::pin(stream))
    }
}

fn parse_tool_calls(value: Option<&Value>) -> Vec<ToolCallResult> {
    let Some(Value::Array(items)) = value else {
        return Vec::new();
    };

    items
        .iter()
        .filter_map(|item| serde_json::from_value::<ToolCallResult>(item.clone()).ok())
        .collect()
}

fn parse_executor_output(value: &Value) -> Result<PyAgentOutput, String> {
    let obj = value
        .as_object()
        .ok_or_else(|| "executor output must be a JSON object".to_string())?;

    let response = obj
        .get("response")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "executor output missing string field 'response'".to_string())?
        .to_string();

    let done = obj.get("done").and_then(|v| v.as_bool()).unwrap_or(true);
    let tool_calls = parse_tool_calls(obj.get("tool_calls"));

    Ok(PyAgentOutput {
        response,
        tool_calls,
        done,
    })
}

async fn call_python_executor_method(
    executor_impl: &Py<PyAny>,
    task_locals: Option<&TaskLocals>,
    method: &str,
    task: &Task,
    ctx: &Context,
) -> Result<Py<PyAny>, String> {
    let task_cloned = task.clone();
    call_hook_method_async(executor_impl, task_locals, method, |py, obj| {
        let py_task = task_to_py(&task_cloned, py)?;
        let py_ctx = context_to_py(ctx, py)?;
        obj.call_method1(method, (py_task.bind(py), py_ctx.bind(py)))
    })
    .await?
    .ok_or_else(|| format!("method '{method}' is not implemented"))
}

async fn call_executor_output(
    executor_impl: &Py<PyAny>,
    task_locals: Option<&TaskLocals>,
    method: &str,
    task: &Task,
    ctx: &Context,
) -> Result<PyAgentOutput, String> {
    let value = call_python_executor_method(executor_impl, task_locals, method, task, ctx).await?;
    Python::attach(|_py| {
        let json = py_any_to_json_value(value.bind(_py)).map_err(|e| e.to_string())?;
        parse_executor_output(&json)
    })
}

async fn next_stream_output(
    iter_obj: &Py<PyAny>,
    task_locals: Option<&TaskLocals>,
) -> Result<Option<PyAgentOutput>, String> {
    let anext_obj = Python::attach(|py| -> Result<Py<PyAny>, String> {
        let it = iter_obj.bind(py);
        it.call_method0("__anext__")
            .map(|v| v.unbind())
            .map_err(|e| e.to_string())
    })?;

    let task_locals = task_locals.cloned();
    let future = Python::attach(|py| {
        crate::async_bridge::into_future(anext_obj.bind(py).clone(), task_locals.as_ref())
    })
    .map_err(|e| e.to_string())?;

    let value = match future.await {
        Ok(v) => v,
        Err(e) => {
            let is_stop = Python::attach(|py| e.is_instance_of::<PyStopAsyncIteration>(py));
            if is_stop {
                return Ok(None);
            }
            return Err(e.to_string());
        }
    };

    Python::attach(|py| {
        let json = py_any_to_json_value(value.bind(py)).map_err(|e| e.to_string())?;
        parse_executor_output(&json).map(Some)
    })
}

fn resolve_max_turns(executor_impl: &Py<PyAny>, override_value: Option<usize>) -> PyResult<usize> {
    if let Some(v) = override_value {
        return Ok(v.max(1));
    }

    let config_obj = call_hook_method_sync(executor_impl, "config", |_py, obj| {
        obj.call_method0("config")
    })
    .map_err(PyRuntimeError::new_err)?
    .ok_or_else(|| {
        PyRuntimeError::new_err("custom executor must implement config() and return max_turns")
    })?;

    Python::attach(|py| {
        let cfg = config_obj.bind(py);
        if let Ok(d) = cfg.cast::<PyDict>()
            && let Ok(Some(item)) = d.get_item("max_turns")
            && let Ok(v) = item.extract::<usize>()
        {
            if v == 0 {
                return Err(PyRuntimeError::new_err(
                    "custom executor config.max_turns must be > 0",
                ));
            }
            return Ok(v);
        }

        let v = cfg
            .getattr("max_turns")
            .and_then(|v| v.extract::<usize>())
            .map_err(|_| {
                PyRuntimeError::new_err(
                    "custom executor config() must provide an integer max_turns",
                )
            })?;
        if v == 0 {
            return Err(PyRuntimeError::new_err(
                "custom executor config.max_turns must be > 0",
            ));
        }
        Ok(v)
    })
}

fn clone_tools(tools: &[PyRef<'_, PyTool>]) -> Vec<Arc<dyn ToolT>> {
    tools
        .iter()
        .map(|tool| Arc::new((**tool).clone()) as Arc<dyn ToolT>)
        .collect()
}

fn parse_schema_json(schema_json: String) -> PyResult<Value> {
    serde_json::from_str(&schema_json)
        .map_err(|e| PyRuntimeError::new_err(format!("invalid schema json: {e}")))
}

fn with_topics<T>(
    mut builder: AgentBuilder<T, ActorAgent>,
    topics: Vec<String>,
) -> AgentBuilder<T, ActorAgent>
where
    T: AgentDeriveT + AgentExecutor + AgentHooks,
    Value: From<<T as AgentExecutor>::Output>,
    <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output>,
    <T as AgentExecutor>::Error: Into<RunnableAgentError>,
{
    for topic in topics {
        builder = builder.subscribe(Topic::<Task>::new(topic));
    }
    builder
}

fn build_direct_executor<T>(
    executor: T,
    llm: Arc<dyn LLMProvider>,
    memory: Box<dyn MemoryProvider>,
) -> BuildDirectResult
where
    T: AgentDeriveT<Output = PyAgentOutput> + AgentExecutor + AgentHooks + Send + Sync + 'static,
    PyAgentOutput: From<<T as AgentExecutor>::Output>,
    <T as AgentExecutor>::Error: Into<RunnableAgentError>,
    <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output>,
{
    Box::pin(async move {
        let mut handle = AgentBuilder::<_, DirectAgent>::new(executor)
            .llm(llm)
            .memory(memory)
            .build()
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let event_stream = std::mem::replace(&mut handle.rx, Box::pin(futures::stream::empty()));
        let agent: Arc<dyn PyRunnable> = Arc::new(handle.agent);
        Ok((agent, event_stream))
    })
}

fn build_actor_executor<T>(
    executor: T,
    llm: Arc<dyn LLMProvider>,
    memory: Box<dyn MemoryProvider>,
    runtime: Arc<dyn Runtime>,
    topics: Vec<String>,
) -> BuildActorResult
where
    T: AgentDeriveT<Output = PyAgentOutput> + AgentExecutor + AgentHooks + Send + Sync + 'static,
    PyAgentOutput: From<<T as AgentExecutor>::Output>,
    <T as AgentExecutor>::Error: Into<RunnableAgentError>,
    <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output>,
    Value: From<<T as AgentExecutor>::Output>,
{
    Box::pin(async move {
        let builder = AgentBuilder::<_, ActorAgent>::new(executor)
            .llm(llm)
            .memory(memory)
            .runtime(runtime);
        let handle: ActorAgentHandle<T> = with_topics(builder, topics)
            .build()
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let actor_ref = handle.actor_ref;
        let send_fn: ActorSendFn =
            Arc::new(move |task: Task| actor_ref.send_message(task).map_err(|e| e.to_string()));
        Ok(send_fn)
    })
}

pub(crate) fn parse_executor_spec(
    agent: &Bound<'_, PyAny>,
) -> PyResult<(Box<dyn PyExecutorBuildable>, PyAgentDef)> {
    let kind = extract_string_attr(agent, "_binding_kind")?;
    let name = extract_string_attr(agent, "_name")?;
    let description = extract_string_attr(agent, "_description")?;
    let tools = extract_tools(agent)?;
    let output_schema = extract_output_schema(agent)?;
    let hooks = extract_optional_object_attr(agent, "_hooks")?;

    let agent_def = PyAgentDef {
        name: name.clone(),
        description: description.clone(),
        tools: tools.clone(),
        output_schema: output_schema.clone(),
        hooks: Python::attach(|py| hooks.as_ref().map(|obj| obj.clone_ref(py))),
        task_locals: None,
    };

    let executor: Box<dyn PyExecutorBuildable> = match kind.as_str() {
        "react" => Box::new(PyReActAgent {
            name,
            description,
            tools,
            output_schema,
            max_turns: extract_usize_attr(agent, "_max_turns")?.max(1),
            hooks,
        }),
        "basic" => Box::new(PyBasicAgent {
            name,
            description,
            tools,
            output_schema,
            hooks,
        }),
        "custom" => Box::new(PyCustomExecutor {
            name,
            description,
            tools,
            output_schema,
            hooks,
            executor_impl: extract_object_attr(agent, "_executor_impl")?,
            max_turns_override: Some(extract_usize_attr(agent, "_max_turns")?.max(1)),
        }),
        other => {
            return Err(PyRuntimeError::new_err(format!(
                "unsupported executor kind '{other}'"
            )));
        }
    };

    Ok((executor, agent_def))
}

fn extract_tools(agent: &Bound<'_, PyAny>) -> PyResult<Vec<Arc<dyn ToolT>>> {
    let tools_obj = agent.getattr("_tools")?;
    let tools_list = tools_obj
        .cast::<PyList>()
        .map_err(|_| PyRuntimeError::new_err("executor._tools must be a list of Tool instances"))?;
    let tools = tools_list
        .iter()
        .map(|tool| {
            tool.extract::<PyRef<'_, PyTool>>()
                .map_err(|_| PyRuntimeError::new_err("executor._tools must contain Tool instances"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(clone_tools(&tools))
}

fn extract_output_schema(agent: &Bound<'_, PyAny>) -> PyResult<Option<Value>> {
    let Some(schema_obj) = get_attr(agent, "_output_schema_json")? else {
        return Ok(None);
    };
    if schema_obj.is_none() {
        return Ok(None);
    }
    schema_obj
        .extract::<String>()
        .map_err(|_| PyRuntimeError::new_err("executor._output_schema_json must be a string"))
        .and_then(parse_schema_json)
        .map(Some)
}

fn extract_string_attr(agent: &Bound<'_, PyAny>, attr: &str) -> PyResult<String> {
    agent
        .getattr(attr)?
        .extract::<String>()
        .map_err(|_| PyRuntimeError::new_err(format!("executor.{attr} must be a string")))
}

fn extract_usize_attr(agent: &Bound<'_, PyAny>, attr: &str) -> PyResult<usize> {
    agent
        .getattr(attr)?
        .extract::<usize>()
        .map_err(|_| PyRuntimeError::new_err(format!("executor.{attr} must be an integer")))
}

fn extract_object_attr(agent: &Bound<'_, PyAny>, attr: &str) -> PyResult<Py<PyAny>> {
    let value = agent.getattr(attr)?;
    if value.is_none() {
        return Err(PyRuntimeError::new_err(format!(
            "executor.{attr} must not be None"
        )));
    }
    Ok(value.unbind())
}

fn extract_optional_object_attr(
    agent: &Bound<'_, PyAny>,
    attr: &str,
) -> PyResult<Option<Py<PyAny>>> {
    let Some(value) = get_attr(agent, attr)? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(value.unbind()))
}

fn get_attr<'py>(agent: &Bound<'py, PyAny>, attr: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    match agent.getattr(attr) {
        Ok(value) => Ok(Some(value)),
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyAttributeError>(agent.py()) => {
            Ok(None)
        }
        Err(err) => Err(err),
    }
}

/// Python-facing ReAct (reasoning-acting) executor.
///
/// Mirrors `ReActAgent::with_max_turns(agent_def, max_turns)` in Rust.
/// Configure the agent definition here; pass to `AgentBuilder` to add LLM
/// and memory before building.
///
/// Example:
/// ```python
/// agent = ReActAgent("planner", "Plans tasks").tools([tool]).max_turns(5)
/// handle = await AgentBuilder(agent).llm(llm).build()
/// ```
#[pyclass(name = "ReActAgent", skip_from_py_object)]
pub struct PyReActAgent {
    pub name: String,
    pub description: String,
    pub tools: Vec<Arc<dyn ToolT>>,
    pub output_schema: Option<Value>,
    pub max_turns: usize,
    pub hooks: Option<Py<PyAny>>,
}

impl Clone for PyReActAgent {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            name: self.name.clone(),
            description: self.description.clone(),
            tools: self.tools.clone(),
            output_schema: self.output_schema.clone(),
            max_turns: self.max_turns,
            hooks: self.hooks.as_ref().map(|h| h.clone_ref(py)),
        })
    }
}

#[pymethods]
impl PyReActAgent {
    #[new]
    pub fn new(name: String, description: String) -> Self {
        Self {
            name,
            description,
            tools: vec![],
            output_schema: None,
            max_turns: 10,
            hooks: None,
        }
    }

    pub fn tools<'a>(
        mut slf: PyRefMut<'a, Self>,
        tools: Vec<PyRef<'_, PyTool>>,
    ) -> PyRefMut<'a, Self> {
        slf.tools = clone_tools(&tools);
        slf
    }

    pub fn max_turns<'a>(mut slf: PyRefMut<'a, Self>, turns: usize) -> PyRefMut<'a, Self> {
        slf.max_turns = turns;
        slf
    }

    pub fn output_schema<'a>(
        mut slf: PyRefMut<'a, Self>,
        schema_json: String,
    ) -> PyResult<PyRefMut<'a, Self>> {
        slf.output_schema = Some(parse_schema_json(schema_json)?);
        Ok(slf)
    }

    pub fn hooks<'a>(mut slf: PyRefMut<'a, Self>, hooks: &Bound<'_, PyAny>) -> PyRefMut<'a, Self> {
        slf.hooks = Some(hooks.clone().unbind());
        slf
    }

    fn __repr__(&self) -> String {
        format!(
            "ReActAgent(name='{}', max_turns={})",
            self.name, self.max_turns
        )
    }
}

/// Python-facing Basic (single-pass) executor.
///
/// Mirrors `BasicAgent::new(agent_def)` in Rust.
/// Configure the agent definition here; pass to `AgentBuilder` to add LLM
/// and memory before building.
///
/// Example:
/// ```python
/// agent = BasicAgent("summariser", "Summarises text").tools([tool])
/// handle = await AgentBuilder(agent).llm(llm).build()
/// ```
#[pyclass(name = "BasicAgent", skip_from_py_object)]
pub struct PyBasicAgent {
    pub name: String,
    pub description: String,
    pub tools: Vec<Arc<dyn ToolT>>,
    pub output_schema: Option<Value>,
    pub hooks: Option<Py<PyAny>>,
}

impl Clone for PyBasicAgent {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            name: self.name.clone(),
            description: self.description.clone(),
            tools: self.tools.clone(),
            output_schema: self.output_schema.clone(),
            hooks: self.hooks.as_ref().map(|h| h.clone_ref(py)),
        })
    }
}

#[pymethods]
impl PyBasicAgent {
    #[new]
    pub fn new(name: String, description: String) -> Self {
        Self {
            name,
            description,
            tools: vec![],
            output_schema: None,
            hooks: None,
        }
    }

    pub fn tools<'a>(
        mut slf: PyRefMut<'a, Self>,
        tools: Vec<PyRef<'_, PyTool>>,
    ) -> PyRefMut<'a, Self> {
        slf.tools = clone_tools(&tools);
        slf
    }

    pub fn output_schema<'a>(
        mut slf: PyRefMut<'a, Self>,
        schema_json: String,
    ) -> PyResult<PyRefMut<'a, Self>> {
        slf.output_schema = Some(parse_schema_json(schema_json)?);
        Ok(slf)
    }

    pub fn hooks<'a>(mut slf: PyRefMut<'a, Self>, hooks: &Bound<'_, PyAny>) -> PyRefMut<'a, Self> {
        slf.hooks = Some(hooks.clone().unbind());
        slf
    }

    fn __repr__(&self) -> String {
        format!("BasicAgent(name='{}')", self.name)
    }
}

/// Python-facing fully custom executor.
///
/// This mirrors Rust's `T: AgentExecutor` pattern by accepting a Python object
/// that implements `config()`, `execute(task, ctx)`, and
/// `execute_stream(task, ctx)`.
#[pyclass(name = "CustomExecutor", skip_from_py_object)]
pub struct PyCustomExecutor {
    pub name: String,
    pub description: String,
    pub tools: Vec<Arc<dyn ToolT>>,
    pub output_schema: Option<Value>,
    pub hooks: Option<Py<PyAny>>,
    pub executor_impl: Py<PyAny>,
    pub max_turns_override: Option<usize>,
}

impl Clone for PyCustomExecutor {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            name: self.name.clone(),
            description: self.description.clone(),
            tools: self.tools.clone(),
            output_schema: self.output_schema.clone(),
            hooks: self.hooks.as_ref().map(|h| h.clone_ref(py)),
            executor_impl: self.executor_impl.clone_ref(py),
            max_turns_override: self.max_turns_override,
        })
    }
}

#[pymethods]
impl PyCustomExecutor {
    #[new]
    pub fn new(name: String, description: String, executor: &Bound<'_, PyAny>) -> Self {
        Self {
            name,
            description,
            tools: Vec::new(),
            output_schema: None,
            hooks: None,
            executor_impl: executor.clone().unbind(),
            max_turns_override: None,
        }
    }

    pub fn tools<'a>(
        mut slf: PyRefMut<'a, Self>,
        tools: Vec<PyRef<'_, PyTool>>,
    ) -> PyRefMut<'a, Self> {
        slf.tools = clone_tools(&tools);
        slf
    }

    pub fn output_schema<'a>(
        mut slf: PyRefMut<'a, Self>,
        schema_json: String,
    ) -> PyResult<PyRefMut<'a, Self>> {
        slf.output_schema = Some(parse_schema_json(schema_json)?);
        Ok(slf)
    }

    pub fn hooks<'a>(mut slf: PyRefMut<'a, Self>, hooks: &Bound<'_, PyAny>) -> PyRefMut<'a, Self> {
        slf.hooks = Some(hooks.clone().unbind());
        slf
    }

    pub fn max_turns<'a>(mut slf: PyRefMut<'a, Self>, turns: usize) -> PyRefMut<'a, Self> {
        slf.max_turns_override = Some(turns.max(1));
        slf
    }

    fn __repr__(&self) -> String {
        format!("CustomExecutor(name='{}')", self.name)
    }
}

// ── PyExecutorBuildable impls ─────────────────────────────────────────────────

impl PyExecutorBuildable for PyReActAgent {
    fn build_direct(
        &self,
        agent_def: PyAgentDef,
        llm: Arc<dyn LLMProvider>,
        memory: Box<dyn MemoryProvider>,
    ) -> BuildDirectResult {
        let max_turns = self.max_turns;
        build_direct_executor(
            ReActAgent::with_max_turns(agent_def, max_turns),
            llm,
            memory,
        )
    }

    fn build_actor(
        &self,
        agent_def: PyAgentDef,
        llm: Arc<dyn LLMProvider>,
        memory: Box<dyn MemoryProvider>,
        runtime: Arc<dyn Runtime>,
        topics: Vec<String>,
    ) -> BuildActorResult {
        let max_turns = self.max_turns;
        build_actor_executor(
            ReActAgent::with_max_turns(agent_def, max_turns),
            llm,
            memory,
            runtime,
            topics,
        )
    }
}

impl PyExecutorBuildable for PyBasicAgent {
    fn build_direct(
        &self,
        agent_def: PyAgentDef,
        llm: Arc<dyn LLMProvider>,
        memory: Box<dyn MemoryProvider>,
    ) -> BuildDirectResult {
        build_direct_executor(BasicAgent::new(agent_def), llm, memory)
    }

    fn build_actor(
        &self,
        agent_def: PyAgentDef,
        llm: Arc<dyn LLMProvider>,
        memory: Box<dyn MemoryProvider>,
        runtime: Arc<dyn Runtime>,
        topics: Vec<String>,
    ) -> BuildActorResult {
        build_actor_executor(BasicAgent::new(agent_def), llm, memory, runtime, topics)
    }
}

impl PyExecutorBuildable for PyCustomExecutor {
    fn build_direct(
        &self,
        agent_def: PyAgentDef,
        llm: Arc<dyn LLMProvider>,
        memory: Box<dyn MemoryProvider>,
    ) -> BuildDirectResult {
        let executor_impl = Python::attach(|py| self.executor_impl.clone_ref(py));
        let max_turns = match resolve_max_turns(&executor_impl, self.max_turns_override) {
            Ok(value) => value,
            Err(err) => return Box::pin(async move { Err(err) }),
        };

        build_direct_executor(
            PyInjectedExecutor {
                task_locals: agent_def.task_locals.clone(),
                agent_def,
                executor_impl,
                max_turns,
            },
            llm,
            memory,
        )
    }

    fn build_actor(
        &self,
        agent_def: PyAgentDef,
        llm: Arc<dyn LLMProvider>,
        memory: Box<dyn MemoryProvider>,
        runtime: Arc<dyn Runtime>,
        topics: Vec<String>,
    ) -> BuildActorResult {
        let executor_impl = Python::attach(|py| self.executor_impl.clone_ref(py));
        let max_turns = match resolve_max_turns(&executor_impl, self.max_turns_override) {
            Ok(value) => value,
            Err(err) => return Box::pin(async move { Err(err) }),
        };

        build_actor_executor(
            PyInjectedExecutor {
                task_locals: agent_def.task_locals.clone(),
                agent_def,
                executor_impl,
                max_turns,
            },
            llm,
            memory,
            runtime,
            topics,
        )
    }
}
