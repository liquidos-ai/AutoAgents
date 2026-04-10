use crate::agent::py_agent::{
    ActorSendFn, BuildActorResult, BuildDirectResult, HookErrorState, PyAgentDef, PyAgentOutput,
    PyExecutorBuildable, PyRunnable, call_hook_method_async, call_hook_method_sync, context_to_py,
    task_to_py,
};
use crate::convert::py_any_to_json_value;
use crate::tool::PyTool;
use autoagents_core::actor::Topic;
use autoagents_core::agent::error::RunnableAgentError;
use autoagents_core::agent::memory::MemoryProvider;
use autoagents_core::agent::prebuilt::executor::{
    BasicAgent, CodeActAgent, CodeActSandboxLimits, ReActAgent,
};
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
                        let final_output = last_output.unwrap_or_else(default_stream_output);
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

fn default_stream_output() -> PyAgentOutput {
    PyAgentOutput {
        response: String::default(),
        tool_calls: Vec::default(),
        executions: Vec::default(),
        done: true,
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

fn parse_executions(value: Option<&Value>) -> Vec<Value> {
    let Some(Value::Array(items)) = value else {
        return Vec::new();
    };

    items.clone()
}

fn aggregate_execution_tool_calls(executions: &[Value]) -> Vec<ToolCallResult> {
    executions
        .iter()
        .filter_map(|execution| execution.get("tool_calls").and_then(Value::as_array))
        .flat_map(|tool_calls| tool_calls.iter())
        .filter_map(|tool_call| serde_json::from_value::<ToolCallResult>(tool_call.clone()).ok())
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
    let executions = parse_executions(obj.get("executions"));
    let mut tool_calls = parse_tool_calls(obj.get("tool_calls"));
    if tool_calls.is_empty() {
        tool_calls = aggregate_execution_tool_calls(&executions);
    }

    Ok(PyAgentOutput {
        response,
        tool_calls,
        executions,
        done,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoagents_core::agent::AgentConfig;
    use autoagents_llm::async_trait;
    use autoagents_llm::chat::{ChatProvider, ChatResponse, StructuredOutputFormat, Tool};
    use autoagents_llm::completion::{CompletionProvider, CompletionRequest, CompletionResponse};
    use autoagents_llm::embedding::EmbeddingProvider;
    use autoagents_llm::error::LLMError;
    use autoagents_llm::models::ModelsProvider;
    use autoagents_protocol::Event;
    use futures::stream;
    use pyo3::types::{PyDict, PyList, PyModule};
    use serde_json::json;
    use std::ffi::CString;
    use std::future::Future;
    use tokio::sync::mpsc;

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

    fn module_from_code<'py>(
        py: Python<'py>,
        code: &str,
        filename: &str,
        module_name: &str,
    ) -> PyResult<Bound<'py, PyModule>> {
        PyModule::from_code(
            py,
            &CString::new(code).expect("python source should be a valid CString"),
            &CString::new(filename).expect("filename should be a valid CString"),
            &CString::new(module_name).expect("module name should be a valid CString"),
        )
    }

    fn sample_tool(py: Python<'_>) -> PyResult<Py<PyAny>> {
        let tool = PyTool::new(
            "lookup".to_string(),
            "Look up a record".to_string(),
            "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}},\"required\":[\"query\"]}".to_string(),
            py.None(),
            Some(
                "{\"type\":\"object\",\"properties\":{\"status\":{\"type\":\"string\"}}}"
                    .to_string(),
            ),
        )?;
        Ok(Py::new(py, tool)?.into_any())
    }

    fn runtime_executor_module(py: Python<'_>) -> PyResult<Py<PyModule>> {
        module_from_code(
            py,
            "class DoneStream:\n\
             \tdef __init__(self):\n\
             \t\tself.items = [\n\
             \t\t\t{\"response\": \"partial\", \"done\": False},\n\
             \t\t\t{\"response\": \"final\", \"done\": True, \"tool_calls\": [{\"tool_name\": \"lookup\", \"success\": True, \"arguments\": {\"query\": \"rust\"}, \"result\": {\"matches\": 1}}]},\n\
             \t\t]\n\
             \n\
             \tasync def __anext__(self):\n\
             \t\tif self.items:\n\
             \t\t\treturn self.items.pop(0)\n\
             \t\traise StopAsyncIteration\n\
             \n\
             class ClosingStream:\n\
             \tdef __init__(self):\n\
             \t\tself.items = [\n\
             \t\t\t{\"response\": \"partial\", \"done\": False},\n\
             \t\t]\n\
             \n\
             \tasync def __anext__(self):\n\
             \t\tif self.items:\n\
             \t\t\treturn self.items.pop(0)\n\
             \t\traise StopAsyncIteration\n\
             \n\
             class SyncExecutor:\n\
             \tdef config(self):\n\
             \t\treturn {\"max_turns\": 2}\n\
             \n\
             \tdef execute(self, task, ctx):\n\
             \t\treturn {\n\
             \t\t\t\"response\": f\"exec:{task['prompt']}\",\n\
             \t\t\t\"executions\": [{\n\
             \t\t\t\t\"execution_id\": \"exec_1\",\n\
             \t\t\t\t\"source\": \"return 1;\",\n\
             \t\t\t\t\"console\": [],\n\
             \t\t\t\t\"tool_calls\": [{\"tool_name\": \"lookup\", \"success\": True, \"arguments\": {\"query\": task['prompt']}, \"result\": {\"matches\": 1}}],\n\
             \t\t\t\t\"result\": 1,\n\
             \t\t\t\t\"success\": True,\n\
             \t\t\t\t\"error\": None,\n\
             \t\t\t\t\"duration_ms\": 5,\n\
             \t\t\t}],\n\
             \t\t\t\"done\": False,\n\
             \t\t}\n\
             \n\
             \tdef execute_stream(self, task, ctx):\n\
             \t\treturn DoneStream()\n\
             \n\
             class ClosingExecutor(SyncExecutor):\n\
             \tdef execute_stream(self, task, ctx):\n\
             \t\treturn ClosingStream()\n\
             \n\
             class MissingExecuteExecutor:\n\
             \tdef config(self):\n\
             \t\treturn {\"max_turns\": 2}\n\
             \n\
             class MissingStreamExecutor:\n\
             \tdef config(self):\n\
             \t\treturn {\"max_turns\": 2}\n\
             \n\
             \tdef execute(self, task, ctx):\n\
             \t\treturn {\"response\": task['prompt'], \"done\": True}\n",
            "autoagents_py/tests/custom_executor_runtime.py",
            "autoagents_custom_executor_runtime",
        )
        .map(|module| module.unbind())
    }

    fn instantiate_executor(module: &Py<PyModule>, name: &str) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            module
                .bind(py)
                .getattr(name)?
                .call0()
                .map(|value| value.unbind())
        })
    }

    fn register_python_execution_module(py: Python<'_>) -> PyResult<()> {
        let package = PyModule::new(py, "autoagents_py")?;
        let execution = module_from_code(
            py,
            "class ExecutionLLM:\n\
             \tdef __init__(self, inner):\n\
             \t\tself.inner = inner\n\
             \n\
             class ExecutionMemory:\n\
             \tdef __init__(self, inner):\n\
             \t\tself.inner = inner\n",
            "autoagents_py/tests/execution_module.py",
            "autoagents_py.execution",
        )?;
        package.add_submodule(&execution)?;

        let sys = py.import("sys")?;
        let modules = sys.getattr("modules")?.cast_into::<PyDict>()?;
        modules.set_item("autoagents_py", &package)?;
        modules.set_item("autoagents_py.execution", &execution)?;
        Ok(())
    }

    fn sample_tool_result() -> Value {
        json!({
            "tool_name": "lookup",
            "success": true,
            "arguments": {"query": "rust"},
            "result": {"matches": 1}
        })
    }

    fn sample_execution() -> Value {
        json!({
            "execution_id": "exec_1",
            "source": "return 1;",
            "console": [],
            "tool_calls": [sample_tool_result()],
            "result": 1,
            "success": true,
            "error": null,
            "duration_ms": 5
        })
    }

    fn sample_agent_def(task_locals: Option<TaskLocals>) -> PyAgentDef {
        PyAgentDef {
            name: "custom-executor".to_string(),
            description: "Executes custom tasks".to_string(),
            tools: Vec::new(),
            output_schema: None,
            hooks: None,
            task_locals,
            hook_errors: HookErrorState::default(),
        }
    }

    fn make_context(tx: Option<mpsc::Sender<Event>>) -> Arc<Context> {
        Arc::new(
            Context::new(Arc::new(MockLLMProvider), tx).with_config(AgentConfig::new(
                "executor".to_string(),
                "Handles tasks".to_string(),
            )),
        )
    }

    fn drain_events(rx: &mut mpsc::Receiver<Event>) -> Vec<Event> {
        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }
        events
    }

    #[derive(Debug)]
    struct MockChatResponse;

    impl std::fmt::Display for MockChatResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("mock")
        }
    }

    impl ChatResponse for MockChatResponse {
        fn text(&self) -> Option<String> {
            Some("mock".to_string())
        }

        fn tool_calls(&self) -> Option<Vec<autoagents_llm::ToolCall>> {
            None
        }
    }

    struct MockLLMProvider;

    #[async_trait]
    impl ChatProvider for MockLLMProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[autoagents_llm::chat::ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            Ok(Box::new(MockChatResponse))
        }

        async fn chat_stream(
            &self,
            _messages: &[autoagents_llm::chat::ChatMessage],
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<
            std::pin::Pin<Box<dyn futures::Stream<Item = Result<String, LLMError>> + Send>>,
            LLMError,
        > {
            Ok(Box::pin(stream::iter(vec![Ok("mock".to_string())])))
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
                text: "completion".to_string(),
            })
        }
    }

    #[async_trait]
    impl EmbeddingProvider for MockLLMProvider {
        async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
            Ok(input.into_iter().map(|_| vec![0.1, 0.2]).collect())
        }
    }

    #[async_trait]
    impl ModelsProvider for MockLLMProvider {}

    impl LLMProvider for MockLLMProvider {}

    #[test]
    fn parse_executor_output_aggregates_codeact_tool_calls() {
        let value = json!({"response": "done", "done": true, "executions": [sample_execution()]});

        let output = parse_executor_output(&value).expect("output should parse");
        assert_eq!(output.tool_calls.len(), 1);
        assert_eq!(output.tool_calls[0].tool_name, "lookup");
        assert_eq!(output.executions.len(), 1);
    }

    #[test]
    fn executor_output_helpers_cover_defaults_and_errors() {
        let default_output = default_stream_output();
        assert!(default_output.done);
        assert!(default_output.response.is_empty());
        assert!(default_output.tool_calls.is_empty());
        assert!(default_output.executions.is_empty());

        let parsed_calls = parse_tool_calls(Some(&json!([
            sample_tool_result(),
            {"tool_name": 7}
        ])));
        assert_eq!(parsed_calls.len(), 1);
        assert!(parse_tool_calls(Some(&json!("not-an-array"))).is_empty());
        assert!(parse_tool_calls(None).is_empty());

        let parsed_executions = parse_executions(Some(&json!([sample_execution()])));
        assert_eq!(parsed_executions.len(), 1);
        assert!(parse_executions(Some(&json!({"oops": true}))).is_empty());
        assert_eq!(
            aggregate_execution_tool_calls(&parsed_executions)[0].tool_name,
            "lookup"
        );

        let output = parse_executor_output(&json!({
            "response": "finished",
            "executions": [sample_execution()]
        }))
        .expect("executor output should parse");
        assert!(output.done, "done defaults to true");
        assert_eq!(output.tool_calls.len(), 1);

        assert!(
            parse_executor_output(&json!(["not-an-object"]))
                .expect_err("arrays should be rejected")
                .contains("JSON object")
        );
        assert!(
            parse_executor_output(&json!({"done": true}))
                .expect_err("response is required")
                .contains("missing string field 'response'")
        );

        let schema = parse_schema_json("{\"type\":\"object\"}".to_string())
            .expect("valid schema json should parse");
        assert_eq!(schema["type"], json!("object"));
        assert!(
            parse_schema_json("{".to_string())
                .expect_err("invalid schema json should fail")
                .to_string()
                .contains("invalid schema json")
        );

        let serialized = task_completed_value(&PyAgentOutput {
            response: "done".to_string(),
            tool_calls: Vec::new(),
            executions: vec![json!({"id": "exec_1"})],
            done: true,
        })
        .expect("py agent output should serialize");
        assert_eq!(serialized["executions"][0]["id"], json!("exec_1"));

        validate_codeact_sandbox_limits(&CodeActSandboxLimits::default())
            .expect("default limits should validate");
        let zero_timeout = CodeActSandboxLimits {
            timeout_ms: 0,
            ..CodeActSandboxLimits::default()
        };
        assert!(
            validate_codeact_sandbox_limits(&zero_timeout)
                .expect_err("zero limits should fail")
                .to_string()
                .contains("timeout_ms must be > 0")
        );
        let inverted_limits = CodeActSandboxLimits {
            max_tool_calls_per_execution: 1,
            max_concurrent_tool_calls: 2,
            ..CodeActSandboxLimits::default()
        };
        assert!(
            validate_codeact_sandbox_limits(&inverted_limits)
                .expect_err("concurrency must not exceed total tool calls")
                .to_string()
                .contains("max_concurrent_tool_calls must be <=")
        );
    }

    #[test]
    fn extractor_helpers_and_resolve_max_turns_cover_variants() {
        Python::initialize();
        Python::attach(|py| {
            let module = module_from_code(
                py,
                "class AttrConfig:\n\
                 \tdef __init__(self, max_turns):\n\
                 \t\tself.max_turns = max_turns\n\
                 \n\
                 class DictConfigExecutor:\n\
                 \tdef config(self):\n\
                 \t\treturn {\"max_turns\": 4}\n\
                 \n\
                 class AttrConfigExecutor:\n\
                 \tdef config(self):\n\
                 \t\treturn AttrConfig(5)\n\
                 \n\
                 class ZeroConfigExecutor:\n\
                 \tdef config(self):\n\
                 \t\treturn {\"max_turns\": 0}\n\
                 \n\
                 class InvalidConfigExecutor:\n\
                 \tdef config(self):\n\
                 \t\treturn {\"max_turns\": \"bad\"}\n\
                 \n\
                 class MissingConfigExecutor:\n\
                 \tpass\n\
                 \n\
                 class Holder:\n\
                 \tdef __init__(self):\n\
                 \t\tself._name = \"planner\"\n\
                 \t\tself._description = \"Plans work\"\n\
                 \t\tself._binding_kind = \"react\"\n\
                 \t\tself._max_turns = 3\n\
                 \t\tself._output_schema_json = '{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}}}'\n\
                 \t\tself._hooks = object()\n\
                 \t\tself._executor_impl = object()\n\
                 \t\tself._sandbox_limits = None\n\
                 \n\
                 class BadStringHolder:\n\
                 \tdef __init__(self):\n\
                 \t\tself._name = 7\n\
                 \n\
                 class BadIntHolder:\n\
                 \tdef __init__(self):\n\
                 \t\tself._max_turns = \"bad\"\n\
                 \n\
                 class NoneHolder:\n\
                 \tdef __init__(self):\n\
                 \t\tself._executor_impl = None\n\
                 \t\tself._hooks = None\n\
                 \t\tself._output_schema_json = None\n\
                 \n\
                 class InvalidSchemaHolder:\n\
                 \tdef __init__(self):\n\
                 \t\tself._output_schema_json = '{'\n\
                 \n\
                 class BadSchemaTypeHolder:\n\
                 \tdef __init__(self):\n\
                 \t\tself._output_schema_json = 9\n\
                 \n\
                 class SandboxHolder:\n\
                 \tdef __init__(self):\n\
                 \t\tself._sandbox_limits = {\n\
                 \t\t\t\"timeout_ms\": 10,\n\
                 \t\t\t\"memory_limit_bytes\": 1024,\n\
                 \t\t\t\"max_source_bytes\": 1024,\n\
                 \t\t\t\"max_console_bytes\": 1024,\n\
                 \t\t\t\"max_tool_calls_per_execution\": 4,\n\
                 \t\t\t\"max_concurrent_tool_calls\": 2,\n\
                 \t\t}\n\
                 \n\
                 class BadSandboxHolder:\n\
                 \tdef __init__(self):\n\
                 \t\tself._sandbox_limits = {\n\
                 \t\t\t\"timeout_ms\": 0,\n\
                 \t\t\t\"memory_limit_bytes\": 1024,\n\
                 \t\t\t\"max_source_bytes\": 1024,\n\
                 \t\t\t\"max_console_bytes\": 1024,\n\
                 \t\t\t\"max_tool_calls_per_execution\": 4,\n\
                 \t\t\t\"max_concurrent_tool_calls\": 2,\n\
                 \t\t}\n\
                 \n\
                 class InvertedSandboxHolder:\n\
                 \tdef __init__(self):\n\
                 \t\tself._sandbox_limits = {\n\
                 \t\t\t\"timeout_ms\": 10,\n\
                 \t\t\t\"memory_limit_bytes\": 1024,\n\
                 \t\t\t\"max_source_bytes\": 1024,\n\
                 \t\t\t\"max_console_bytes\": 1024,\n\
                 \t\t\t\"max_tool_calls_per_execution\": 1,\n\
                 \t\t\t\"max_concurrent_tool_calls\": 2,\n\
                 \t\t}\n",
                "autoagents_py/tests/executor_extractors.py",
                "autoagents_executor_extractors",
            )
            .expect("python module should compile");

            let holder = module
                .getattr("Holder")
                .expect("holder class should exist")
                .call0()
                .expect("holder should instantiate");
            let bad_string = module
                .getattr("BadStringHolder")
                .expect("bad string holder should exist")
                .call0()
                .expect("bad string holder should instantiate");
            let bad_int = module
                .getattr("BadIntHolder")
                .expect("bad int holder should exist")
                .call0()
                .expect("bad int holder should instantiate");
            let none_holder = module
                .getattr("NoneHolder")
                .expect("none holder should exist")
                .call0()
                .expect("none holder should instantiate");
            let invalid_schema = module
                .getattr("InvalidSchemaHolder")
                .expect("invalid schema holder should exist")
                .call0()
                .expect("invalid schema holder should instantiate");
            let bad_schema_type = module
                .getattr("BadSchemaTypeHolder")
                .expect("bad schema type holder should exist")
                .call0()
                .expect("bad schema type holder should instantiate");
            let sandbox_holder = module
                .getattr("SandboxHolder")
                .expect("sandbox holder should exist")
                .call0()
                .expect("sandbox holder should instantiate");
            let bad_sandbox_holder = module
                .getattr("BadSandboxHolder")
                .expect("bad sandbox holder should exist")
                .call0()
                .expect("bad sandbox holder should instantiate");
            let inverted_sandbox_holder = module
                .getattr("InvertedSandboxHolder")
                .expect("inverted sandbox holder should exist")
                .call0()
                .expect("inverted sandbox holder should instantiate");

            assert_eq!(
                extract_string_attr(holder.as_any(), "_name").unwrap(),
                "planner"
            );
            assert_eq!(
                extract_usize_attr(holder.as_any(), "_max_turns").unwrap(),
                3
            );
            assert!(get_attr(holder.as_any(), "_missing").unwrap().is_none());
            assert!(
                extract_string_attr(bad_string.as_any(), "_name")
                    .expect_err("invalid string attr should fail")
                    .to_string()
                    .contains("must be a string")
            );
            assert!(
                extract_usize_attr(bad_int.as_any(), "_max_turns")
                    .expect_err("invalid usize attr should fail")
                    .to_string()
                    .contains("must be an integer")
            );

            assert!(extract_object_attr(holder.as_any(), "_hooks").is_ok());
            assert!(
                extract_object_attr(none_holder.as_any(), "_executor_impl")
                    .expect_err("None object attrs should fail")
                    .to_string()
                    .contains("must not be None")
            );
            assert!(
                extract_optional_object_attr(holder.as_any(), "_hooks")
                    .unwrap()
                    .is_some()
            );
            assert!(
                extract_optional_object_attr(none_holder.as_any(), "_hooks")
                    .unwrap()
                    .is_none()
            );
            assert!(
                extract_optional_object_attr(holder.as_any(), "_missing")
                    .unwrap()
                    .is_none()
            );

            let schema = extract_output_schema(holder.as_any())
                .expect("schema should parse")
                .expect("schema should exist");
            assert_eq!(schema["type"], json!("object"));
            assert!(
                extract_output_schema(none_holder.as_any())
                    .unwrap()
                    .is_none()
            );
            assert!(
                extract_output_schema(invalid_schema.as_any())
                    .expect_err("invalid schema json should fail")
                    .to_string()
                    .contains("invalid schema json")
            );
            assert!(
                extract_output_schema(bad_schema_type.as_any())
                    .expect_err("schema type must be string")
                    .to_string()
                    .contains("must be a string")
            );

            let holder_obj = holder.as_any().clone().unbind();
            assert_eq!(resolve_max_turns(&holder_obj, Some(0)).unwrap(), 1);
            let dict_executor = module
                .getattr("DictConfigExecutor")
                .expect("dict executor should exist")
                .call0()
                .expect("dict executor should instantiate")
                .unbind();
            let attr_executor = module
                .getattr("AttrConfigExecutor")
                .expect("attr executor should exist")
                .call0()
                .expect("attr executor should instantiate")
                .unbind();
            let zero_executor = module
                .getattr("ZeroConfigExecutor")
                .expect("zero executor should exist")
                .call0()
                .expect("zero executor should instantiate")
                .unbind();
            let invalid_executor = module
                .getattr("InvalidConfigExecutor")
                .expect("invalid executor should exist")
                .call0()
                .expect("invalid executor should instantiate")
                .unbind();
            let missing_executor = module
                .getattr("MissingConfigExecutor")
                .expect("missing executor should exist")
                .call0()
                .expect("missing executor should instantiate")
                .unbind();

            assert_eq!(resolve_max_turns(&dict_executor, None).unwrap(), 4);
            assert_eq!(resolve_max_turns(&attr_executor, None).unwrap(), 5);
            assert!(
                resolve_max_turns(&zero_executor, None)
                    .expect_err("zero max_turns should fail")
                    .to_string()
                    .contains("must be > 0")
            );
            assert!(
                resolve_max_turns(&invalid_executor, None)
                    .expect_err("invalid max_turns should fail")
                    .to_string()
                    .contains("provide an integer max_turns")
            );
            assert!(
                resolve_max_turns(&missing_executor, None)
                    .expect_err("missing config should fail")
                    .to_string()
                    .contains("must implement config()")
            );

            let limits = extract_codeact_sandbox_limits(sandbox_holder.as_any())
                .expect("valid sandbox limits should parse");
            assert_eq!(limits.max_concurrent_tool_calls, 2);
            assert!(
                extract_codeact_sandbox_limits(bad_sandbox_holder.as_any())
                    .expect_err("zero sandbox limits should fail")
                    .to_string()
                    .contains("timeout_ms must be > 0")
            );
            assert!(
                extract_codeact_sandbox_limits(inverted_sandbox_holder.as_any())
                    .expect_err("inverted sandbox limits should fail")
                    .to_string()
                    .contains("max_concurrent_tool_calls must be <=")
            );
        });
    }

    #[test]
    fn builder_methods_and_parse_executor_spec_cover_supported_kinds() {
        init_runtime_bridge();
        Python::attach(|py| -> PyResult<()> {
            let tool = sample_tool(py)?;
            let hooks = PyModule::new(py, "_executor_hooks")?.into_any().unbind();
            let module = module_from_code(
                py,
                "class CustomImpl:\n\
                 \tdef config(self):\n\
                 \t\treturn {\"max_turns\": 2}\n\
                 \n\
                 \tdef execute(self, task, ctx):\n\
                 \t\treturn {\"response\": task[\"prompt\"], \"done\": True}\n\
                 \n\
                 \tdef execute_stream(self, task, ctx):\n\
                 \t\treturn self\n\
                 \n\
                 \tdef __anext__(self):\n\
                 \t\traise StopAsyncIteration\n",
                "autoagents_py/tests/executor_specs.py",
                "autoagents_executor_specs",
            )?;
            let custom_impl = module.getattr("CustomImpl")?.call0()?.unbind();

            let react = Py::new(
                py,
                PyReActAgent::new("react".to_string(), "desc".to_string()),
            )?;
            {
                let tool_ref = tool.bind(py).extract::<PyRef<'_, PyTool>>()?;
                let slf = react.bind(py).borrow_mut();
                let slf = PyReActAgent::tools(slf, vec![tool_ref]);
                let slf = PyReActAgent::max_turns(slf, 7);
                let slf = PyReActAgent::output_schema(
                    slf,
                    "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}}}"
                        .to_string(),
                )?;
                let _ = PyReActAgent::hooks(slf, hooks.bind(py));
            }
            assert_eq!(
                react.bind(py).borrow().__repr__(),
                "ReActAgent(name='react', max_turns=7)"
            );

            let basic = Py::new(
                py,
                PyBasicAgent::new("basic".to_string(), "desc".to_string()),
            )?;
            {
                let tool_ref = tool.bind(py).extract::<PyRef<'_, PyTool>>()?;
                let slf = basic.bind(py).borrow_mut();
                let slf = PyBasicAgent::tools(slf, vec![tool_ref]);
                let slf = PyBasicAgent::output_schema(
                    slf,
                    "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}}}"
                        .to_string(),
                )?;
                let _ = PyBasicAgent::hooks(slf, hooks.bind(py));
            }
            assert_eq!(
                basic.bind(py).borrow().__repr__(),
                "BasicAgent(name='basic')"
            );

            let custom = Py::new(
                py,
                PyCustomExecutor::new(
                    "custom".to_string(),
                    "desc".to_string(),
                    custom_impl.bind(py),
                ),
            )?;
            {
                let tool_ref = tool.bind(py).extract::<PyRef<'_, PyTool>>()?;
                let slf = custom.bind(py).borrow_mut();
                let slf = PyCustomExecutor::tools(slf, vec![tool_ref]);
                let slf = PyCustomExecutor::output_schema(
                    slf,
                    "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}}}"
                        .to_string(),
                )?;
                let slf = PyCustomExecutor::hooks(slf, hooks.bind(py));
                let _ = PyCustomExecutor::max_turns(slf, 0);
            }
            assert_eq!(
                custom.bind(py).borrow().__repr__(),
                "CustomExecutor(name='custom')"
            );
            assert_eq!(custom.bind(py).borrow().max_turns_override, Some(1));

            let namespace = py.import("types")?.getattr("SimpleNamespace")?;
            let tools = PyList::empty(py);
            tools.append(tool.bind(py))?;

            let make_spec = |kind: &str,
                             executor_impl: Option<&Py<PyAny>>,
                             sandbox_limits: Option<Value>|
             -> PyResult<Py<PyAny>> {
                let spec = namespace.call0()?;
                spec.setattr("_binding_kind", kind)?;
                spec.setattr("_name", format!("{kind}-agent"))?;
                spec.setattr("_description", "executor description")?;
                spec.setattr("_tools", tools.clone())?;
                spec.setattr(
                    "_output_schema_json",
                    "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}}}",
                )?;
                spec.setattr("_hooks", py.None())?;
                spec.setattr("_max_turns", 0)?;
                if let Some(value) = executor_impl {
                    spec.setattr("_executor_impl", value.bind(py))?;
                } else {
                    spec.setattr("_executor_impl", py.None())?;
                }
                if let Some(limits) = sandbox_limits {
                    let value = crate::convert::json_value_to_py(py, &limits)?;
                    spec.setattr("_sandbox_limits", value.bind(py))?;
                } else {
                    spec.setattr("_sandbox_limits", py.None())?;
                }
                Ok(spec.unbind())
            };

            let valid_sandbox = json!({
                "timeout_ms": 10,
                "memory_limit_bytes": 1024,
                "max_source_bytes": 1024,
                "max_console_bytes": 1024,
                "max_tool_calls_per_execution": 4,
                "max_concurrent_tool_calls": 2
            });

            for kind in ["react", "basic"] {
                let spec = make_spec(kind, None, None)?;
                let (_executor, agent_def) = parse_executor_spec(spec.bind(py))?;
                assert_eq!(agent_def.name, format!("{kind}-agent"));
                assert_eq!(agent_def.tools.len(), 1);
                assert!(agent_def.output_schema.is_some());
            }

            let codeact_spec = make_spec("codeact", None, Some(valid_sandbox.clone()))?;
            let (_executor, codeact_def) = parse_executor_spec(codeact_spec.bind(py))?;
            assert_eq!(codeact_def.name, "codeact-agent");

            let custom_spec = make_spec("custom", Some(&custom_impl), None)?;
            let (_executor, custom_def) = parse_executor_spec(custom_spec.bind(py))?;
            assert_eq!(custom_def.name, "custom-agent");

            let unsupported = make_spec("mystery", None, None)?;
            match parse_executor_spec(unsupported.bind(py)) {
                Ok(_) => panic!("unsupported executor kinds should fail"),
                Err(err) => assert!(err.to_string().contains("unsupported executor kind")),
            }

            let invalid_codeact = make_spec(
                "codeact",
                None,
                Some(json!({
                    "timeout_ms": 0,
                    "memory_limit_bytes": 1024,
                    "max_source_bytes": 1024,
                    "max_console_bytes": 1024,
                    "max_tool_calls_per_execution": 4,
                    "max_concurrent_tool_calls": 2
                })),
            )?;
            match parse_executor_spec(invalid_codeact.bind(py)) {
                Ok(_) => panic!("invalid codeact sandbox limits should fail"),
                Err(err) => assert!(err.to_string().contains("timeout_ms must be > 0")),
            }

            Ok(())
        })
        .expect("python specs should be parsed");
    }

    #[test]
    fn custom_executor_helpers_and_injected_executor_cover_sync_run_paths() {
        Python::initialize();
        Python::attach(register_python_execution_module).expect("execution module should register");
        let module =
            Python::attach(runtime_executor_module).expect("runtime executor module should load");

        let task = Task::new("ship it");
        let sync_executor =
            instantiate_executor(&module, "SyncExecutor").expect("sync executor should exist");
        let call_output = block_on_test(async {
            tokio::time::timeout(
                std::time::Duration::from_secs(2),
                call_executor_output(
                    &sync_executor,
                    None,
                    "execute",
                    &task,
                    make_context(None).as_ref(),
                ),
            )
            .await
        })
        .expect("executor output should not hang")
        .expect("executor output should parse");
        assert_eq!(call_output.response, "exec:ship it");
        assert_eq!(call_output.tool_calls.len(), 1);

        let missing_execute = instantiate_executor(&module, "MissingExecuteExecutor")
            .expect("missing execute executor should exist");
        assert!(
            block_on_test(call_python_executor_method(
                &missing_execute,
                None,
                "execute",
                &task,
                make_context(None).as_ref(),
            ))
            .expect_err("missing execute should fail")
            .contains("method 'execute' is not implemented")
        );

        let (tx, mut rx) = mpsc::channel(8);
        let injected = PyInjectedExecutor {
            agent_def: sample_agent_def(None),
            executor_impl: Python::attach(|py| sync_executor.clone_ref(py)),
            max_turns: 3,
            task_locals: None,
        };
        let output = block_on_test(async {
            tokio::time::timeout(
                std::time::Duration::from_secs(2),
                injected.execute(&task, make_context(Some(tx))),
            )
            .await
        })
        .expect("custom execute should not hang")
        .expect("custom execute should succeed");
        assert_eq!(output.response, "exec:ship it");
        assert_eq!(output.tool_calls[0].tool_name, "lookup");

        let events = drain_events(&mut rx);
        assert!(matches!(events[0], Event::TaskStarted { .. }));
        assert!(matches!(events[1], Event::TurnStarted { .. }));
        assert!(matches!(events[2], Event::TurnCompleted { .. }));
        assert!(matches!(events[3], Event::TaskComplete { .. }));

        let (tx, mut rx) = mpsc::channel(8);
        let missing = PyInjectedExecutor {
            agent_def: sample_agent_def(None),
            executor_impl: Python::attach(|py| missing_execute.clone_ref(py)),
            max_turns: 3,
            task_locals: None,
        };
        let err = block_on_test(async {
            tokio::time::timeout(
                std::time::Duration::from_secs(2),
                missing.execute(&task, make_context(Some(tx))),
            )
            .await
        })
        .expect("missing execute should not hang")
        .expect_err("missing execute should error");
        assert!(err.to_string().contains("custom executor execute() failed"));
        let error_events = drain_events(&mut rx);
        assert!(matches!(error_events[2], Event::TaskError { .. }));

        let (tx, mut rx) = mpsc::channel(8);
        let missing_stream_executor = PyInjectedExecutor {
            agent_def: sample_agent_def(None),
            executor_impl: instantiate_executor(&module, "MissingStreamExecutor")
                .expect("missing stream executor should exist"),
            max_turns: 3,
            task_locals: None,
        };
        let err = match block_on_test(async {
            tokio::time::timeout(
                std::time::Duration::from_secs(2),
                missing_stream_executor.execute_stream(&task, make_context(Some(tx))),
            )
            .await
        })
        .expect("missing stream execute should not hang")
        {
            Ok(_) => panic!("missing stream method should fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("execute_stream() call failed"));
        let stream_error_events = drain_events(&mut rx);
        assert!(matches!(stream_error_events[2], Event::TaskError { .. }));
    }
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
        hook_errors: HookErrorState::default(),
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
        "codeact" => Box::new(PyCodeActAgent {
            name,
            description,
            tools,
            output_schema,
            max_turns: extract_usize_attr(agent, "_max_turns")?.max(1),
            hooks,
            sandbox_limits: extract_codeact_sandbox_limits(agent)?,
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

fn extract_codeact_sandbox_limits(agent: &Bound<'_, PyAny>) -> PyResult<CodeActSandboxLimits> {
    let Some(value) = get_attr(agent, "_sandbox_limits")? else {
        return Ok(CodeActSandboxLimits::default());
    };
    if value.is_none() {
        return Ok(CodeActSandboxLimits::default());
    }

    let json = py_any_to_json_value(&value).map_err(|e| {
        PyRuntimeError::new_err(format!(
            "executor._sandbox_limits must be a JSON object: {e}"
        ))
    })?;
    let limits: CodeActSandboxLimits = serde_json::from_value(json).map_err(|e| {
        PyRuntimeError::new_err(format!("invalid executor._sandbox_limits value: {e}"))
    })?;
    validate_codeact_sandbox_limits(&limits)?;
    Ok(limits)
}

fn validate_codeact_sandbox_limits(limits: &CodeActSandboxLimits) -> PyResult<()> {
    for (name, value) in [
        ("timeout_ms", limits.timeout_ms as u128),
        ("memory_limit_bytes", limits.memory_limit_bytes as u128),
        ("max_source_bytes", limits.max_source_bytes as u128),
        ("max_console_bytes", limits.max_console_bytes as u128),
        (
            "max_tool_calls_per_execution",
            limits.max_tool_calls_per_execution as u128,
        ),
        (
            "max_concurrent_tool_calls",
            limits.max_concurrent_tool_calls as u128,
        ),
    ] {
        if value == 0 {
            return Err(PyRuntimeError::new_err(format!(
                "executor._sandbox_limits.{name} must be > 0"
            )));
        }
    }

    if limits.max_concurrent_tool_calls > limits.max_tool_calls_per_execution {
        return Err(PyRuntimeError::new_err(
            "executor._sandbox_limits.max_concurrent_tool_calls must be <= max_tool_calls_per_execution",
        ));
    }

    Ok(())
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

pub struct PyCodeActAgent {
    pub name: String,
    pub description: String,
    pub tools: Vec<Arc<dyn ToolT>>,
    pub output_schema: Option<Value>,
    pub max_turns: usize,
    pub hooks: Option<Py<PyAny>>,
    pub sandbox_limits: CodeActSandboxLimits,
}

impl Clone for PyCodeActAgent {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            name: self.name.clone(),
            description: self.description.clone(),
            tools: self.tools.clone(),
            output_schema: self.output_schema.clone(),
            max_turns: self.max_turns,
            hooks: self.hooks.as_ref().map(|h| h.clone_ref(py)),
            sandbox_limits: self.sandbox_limits.clone(),
        })
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

impl PyExecutorBuildable for PyCodeActAgent {
    fn build_direct(
        &self,
        agent_def: PyAgentDef,
        llm: Arc<dyn LLMProvider>,
        memory: Box<dyn MemoryProvider>,
    ) -> BuildDirectResult {
        let max_turns = self.max_turns;
        let sandbox_limits = self.sandbox_limits.clone();
        build_direct_executor(
            CodeActAgent::with_max_turns(agent_def, max_turns).with_sandbox_limits(sandbox_limits),
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
        let sandbox_limits = self.sandbox_limits.clone();
        build_actor_executor(
            CodeActAgent::with_max_turns(agent_def, max_turns).with_sandbox_limits(sandbox_limits),
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
