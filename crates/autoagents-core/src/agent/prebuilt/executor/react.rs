use crate::agent::executor::AgentExecutor;
use crate::agent::executor::event_helper::EventHelper;
use crate::agent::executor::turn_engine::{
    TurnDelta, TurnEngine, TurnEngineConfig, TurnEngineError, record_task_state,
};
use crate::agent::task::Task;
use crate::agent::{AgentDeriveT, Context, ExecutorConfig};
use crate::channel::channel;
use crate::tool::{ToolCallResult, ToolT};
use crate::utils::{receiver_into_stream, spawn_future};
use async_trait::async_trait;
use autoagents_llm::ToolCall;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::Arc;
use thiserror::Error;

#[cfg(not(target_arch = "wasm32"))]
pub use tokio::sync::mpsc::error::SendError;

#[cfg(target_arch = "wasm32")]
type SendError = futures::channel::mpsc::SendError;

use crate::agent::hooks::{AgentHooks, HookOutcome};
use autoagents_protocol::Event;

/// Output of the ReAct-style agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReActAgentOutput {
    pub response: String,
    pub tool_calls: Vec<ToolCallResult>,
    pub done: bool,
}

impl From<ReActAgentOutput> for Value {
    fn from(output: ReActAgentOutput) -> Self {
        serde_json::to_value(output).unwrap_or(Value::Null)
    }
}
impl From<ReActAgentOutput> for String {
    fn from(output: ReActAgentOutput) -> Self {
        output.response
    }
}

impl ReActAgentOutput {
    /// Try to parse the response string as structured JSON of type `T`.
    /// Returns `serde_json::Error` if parsing fails.
    pub fn try_parse<T: for<'de> serde::Deserialize<'de>>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_str::<T>(&self.response)
    }

    /// Parse the response string as structured JSON of type `T`, or map the raw
    /// text into `T` using the provided fallback function if parsing fails.
    /// This is useful in examples to avoid repeating parsing boilerplate.
    pub fn parse_or_map<T, F>(&self, fallback: F) -> T
    where
        T: for<'de> serde::Deserialize<'de>,
        F: FnOnce(&str) -> T,
    {
        self.try_parse::<T>()
            .unwrap_or_else(|_| fallback(&self.response))
    }
}

impl ReActAgentOutput {
    /// Extract the agent output from the ReAct response
    #[allow(clippy::result_large_err)]
    pub fn extract_agent_output<T>(val: Value) -> Result<T, ReActExecutorError>
    where
        T: for<'de> serde::Deserialize<'de>,
    {
        let react_output: Self = serde_json::from_value(val)
            .map_err(|e| ReActExecutorError::AgentOutputError(e.to_string()))?;
        serde_json::from_str(&react_output.response)
            .map_err(|e| ReActExecutorError::AgentOutputError(e.to_string()))
    }
}

#[derive(Error, Debug)]
pub enum ReActExecutorError {
    #[error("LLM error: {0}")]
    LLMError(String),

    #[error("Maximum turns exceeded: {max_turns}")]
    MaxTurnsExceeded { max_turns: usize },

    #[error("Other error: {0}")]
    Other(String),

    #[cfg(not(target_arch = "wasm32"))]
    #[error("Event error: {0}")]
    EventError(#[from] SendError<Event>),

    #[cfg(target_arch = "wasm32")]
    #[error("Event error: {0}")]
    EventError(#[from] SendError),

    #[error("Extracting Agent Output Error: {0}")]
    AgentOutputError(String),
}

impl From<TurnEngineError> for ReActExecutorError {
    fn from(error: TurnEngineError) -> Self {
        match error {
            TurnEngineError::LLMError(err) => ReActExecutorError::LLMError(err),
            TurnEngineError::Aborted => {
                ReActExecutorError::Other("Run aborted by hook".to_string())
            }
            TurnEngineError::Other(err) => ReActExecutorError::Other(err),
        }
    }
}

/// Wrapper type for the multi-turn ReAct executor with tool calling support.
///
/// Use `ReActAgent<T>` when your agent needs to perform tool calls, manage
/// multiple turns, and optionally stream content and tool-call deltas.
#[derive(Debug)]
pub struct ReActAgent<T: AgentDeriveT> {
    inner: Arc<T>,
}

impl<T: AgentDeriveT> Clone for ReActAgent<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T: AgentDeriveT> ReActAgent<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

impl<T: AgentDeriveT> Deref for ReActAgent<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Implement AgentDeriveT for the wrapper by delegating to the inner type
#[async_trait]
impl<T: AgentDeriveT> AgentDeriveT for ReActAgent<T> {
    type Output = <T as AgentDeriveT>::Output;

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn output_schema(&self) -> Option<Value> {
        self.inner.output_schema()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn tools(&self) -> Vec<Box<dyn ToolT>> {
        self.inner.tools()
    }
}

#[async_trait]
impl<T> AgentHooks for ReActAgent<T>
where
    T: AgentDeriveT + AgentHooks + Send + Sync + 'static,
{
    async fn on_agent_create(&self) {
        self.inner.on_agent_create().await
    }

    async fn on_run_start(&self, task: &Task, ctx: &Context) -> HookOutcome {
        self.inner.on_run_start(task, ctx).await
    }

    async fn on_run_complete(&self, task: &Task, result: &Self::Output, ctx: &Context) {
        self.inner.on_run_complete(task, result, ctx).await
    }

    async fn on_turn_start(&self, turn_index: usize, ctx: &Context) {
        self.inner.on_turn_start(turn_index, ctx).await
    }

    async fn on_turn_complete(&self, turn_index: usize, ctx: &Context) {
        self.inner.on_turn_complete(turn_index, ctx).await
    }

    async fn on_tool_call(&self, tool_call: &ToolCall, ctx: &Context) -> HookOutcome {
        self.inner.on_tool_call(tool_call, ctx).await
    }

    async fn on_tool_start(&self, tool_call: &ToolCall, ctx: &Context) {
        self.inner.on_tool_start(tool_call, ctx).await
    }

    async fn on_tool_result(&self, tool_call: &ToolCall, result: &ToolCallResult, ctx: &Context) {
        self.inner.on_tool_result(tool_call, result, ctx).await
    }

    async fn on_tool_error(&self, tool_call: &ToolCall, err: Value, ctx: &Context) {
        self.inner.on_tool_error(tool_call, err, ctx).await
    }
    async fn on_agent_shutdown(&self) {
        self.inner.on_agent_shutdown().await
    }
}

/// Implementation of AgentExecutor for the ReActExecutorWrapper
#[async_trait]
impl<T: AgentDeriveT + AgentHooks> AgentExecutor for ReActAgent<T> {
    type Output = ReActAgentOutput;
    type Error = ReActExecutorError;

    fn config(&self) -> ExecutorConfig {
        ExecutorConfig { max_turns: 10 }
    }

    async fn execute(
        &self,
        task: &Task,
        context: Arc<Context>,
    ) -> Result<Self::Output, Self::Error> {
        if self.on_run_start(task, &context).await == HookOutcome::Abort {
            return Err(ReActExecutorError::Other("Run aborted by hook".to_string()));
        }

        record_task_state(&context, task);

        let tx_event = context.tx().ok();
        EventHelper::send_task_started(
            &tx_event,
            task.submission_id,
            context.config().id,
            context.config().name.clone(),
            task.prompt.clone(),
        )
        .await;

        let engine = TurnEngine::new(TurnEngineConfig::react(self.config().max_turns));
        let mut turn_state = engine.turn_state(&context);
        let max_turns = self.config().max_turns;
        let mut accumulated_tool_calls = Vec::new();
        let mut final_response = String::new();

        for turn_index in 0..max_turns {
            let result = engine
                .run_turn(self, task, &context, &mut turn_state, turn_index, max_turns)
                .await?;

            match result {
                crate::agent::executor::TurnResult::Complete(output) => {
                    final_response = output.response.clone();
                    EventHelper::send_task_completed(
                        &tx_event,
                        task.submission_id,
                        context.config().id,
                        context.config().name.clone(),
                        final_response.clone(),
                    )
                    .await;

                    accumulated_tool_calls.extend(output.tool_calls);

                    return Ok(ReActAgentOutput {
                        response: final_response,
                        done: true,
                        tool_calls: accumulated_tool_calls,
                    });
                }
                crate::agent::executor::TurnResult::Continue(Some(output)) => {
                    if !output.response.is_empty() {
                        final_response = output.response;
                    }
                    accumulated_tool_calls.extend(output.tool_calls);
                }
                crate::agent::executor::TurnResult::Continue(None) => {}
            }
        }

        if !final_response.is_empty() || !accumulated_tool_calls.is_empty() {
            EventHelper::send_task_completed(
                &tx_event,
                task.submission_id,
                context.config().id,
                context.config().name.clone(),
                final_response.clone(),
            )
            .await;

            return Ok(ReActAgentOutput {
                response: final_response,
                done: true,
                tool_calls: accumulated_tool_calls,
            });
        }

        Err(ReActExecutorError::MaxTurnsExceeded { max_turns })
    }

    async fn execute_stream(
        &self,
        task: &Task,
        context: Arc<Context>,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<ReActAgentOutput, Self::Error>> + Send>>,
        Self::Error,
    > {
        if self.on_run_start(task, &context).await == HookOutcome::Abort {
            return Err(ReActExecutorError::Other("Run aborted by hook".to_string()));
        }

        record_task_state(&context, task);

        let tx_event = context.tx().ok();
        EventHelper::send_task_started(
            &tx_event,
            task.submission_id,
            context.config().id,
            context.config().name.clone(),
            task.prompt.clone(),
        )
        .await;

        let engine = TurnEngine::new(TurnEngineConfig::react(self.config().max_turns));
        let mut turn_state = engine.turn_state(&context);
        let max_turns = self.config().max_turns;
        let context_clone = context.clone();
        let task = task.clone();
        let executor = self.clone();

        let (tx, rx) = channel::<Result<ReActAgentOutput, ReActExecutorError>>(100);

        spawn_future(async move {
            let mut accumulated_tool_calls = Vec::new();
            let mut final_response = String::new();

            for turn_index in 0..max_turns {
                let turn_stream = engine
                    .run_turn_stream(
                        executor.clone(),
                        &task,
                        context_clone.clone(),
                        &mut turn_state,
                        turn_index,
                        max_turns,
                    )
                    .await;

                let mut turn_result = None;

                match turn_stream {
                    Ok(mut stream) => {
                        use futures::StreamExt;
                        while let Some(delta_result) = stream.next().await {
                            match delta_result {
                                Ok(TurnDelta::Text(content)) => {
                                    let _ = tx
                                        .send(Ok(ReActAgentOutput {
                                            response: content,
                                            tool_calls: Vec::new(),
                                            done: false,
                                        }))
                                        .await;
                                }
                                Ok(TurnDelta::ToolResults(tool_results)) => {
                                    accumulated_tool_calls.extend(tool_results);
                                    let _ = tx
                                        .send(Ok(ReActAgentOutput {
                                            response: String::new(),
                                            tool_calls: accumulated_tool_calls.clone(),
                                            done: false,
                                        }))
                                        .await;
                                }
                                Ok(TurnDelta::Done(result)) => {
                                    turn_result = Some(result);
                                    break;
                                }
                                Err(err) => {
                                    let _ = tx.send(Err(err.into())).await;
                                    return;
                                }
                            }
                        }
                    }
                    Err(err) => {
                        let _ = tx.send(Err(err.into())).await;
                        return;
                    }
                }

                let Some(result) = turn_result else {
                    let _ = tx
                        .send(Err(ReActExecutorError::Other(
                            "Stream ended without final result".to_string(),
                        )))
                        .await;
                    return;
                };

                match result {
                    crate::agent::executor::TurnResult::Complete(output) => {
                        final_response = output.response.clone();
                        accumulated_tool_calls.extend(output.tool_calls);
                        break;
                    }
                    crate::agent::executor::TurnResult::Continue(Some(output)) => {
                        if !output.response.is_empty() {
                            final_response = output.response;
                        }
                        accumulated_tool_calls.extend(output.tool_calls);
                    }
                    crate::agent::executor::TurnResult::Continue(None) => {}
                }
            }

            let tx_event = context_clone.tx().ok();
            EventHelper::send_stream_complete(&tx_event, task.submission_id).await;
            let _ = tx
                .send(Ok(ReActAgentOutput {
                    response: final_response.clone(),
                    done: true,
                    tool_calls: accumulated_tool_calls.clone(),
                }))
                .await;

            if !final_response.is_empty() || !accumulated_tool_calls.is_empty() {
                EventHelper::send_task_completed(
                    &tx_event,
                    task.submission_id,
                    context_clone.config().id,
                    context_clone.config().name.clone(),
                    final_response,
                )
                .await;
            }
        });

        Ok(receiver_into_stream(rx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::agent::MockAgentImpl;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestAgentOutput {
        value: i32,
        message: String,
    }

    #[test]
    fn test_extract_agent_output_success() {
        let agent_output = TestAgentOutput {
            value: 42,
            message: "Hello, world!".to_string(),
        };

        let react_output = ReActAgentOutput {
            response: serde_json::to_string(&agent_output).unwrap(),
            done: true,
            tool_calls: vec![],
        };

        let react_value = serde_json::to_value(react_output).unwrap();
        let extracted: TestAgentOutput =
            ReActAgentOutput::extract_agent_output(react_value).unwrap();
        assert_eq!(extracted, agent_output);
    }

    #[test]
    fn test_extract_agent_output_invalid_react() {
        let result = ReActAgentOutput::extract_agent_output::<TestAgentOutput>(
            serde_json::json!({"not": "react"}),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_react_agent_output_try_parse_success() {
        let output = ReActAgentOutput {
            response: r#"{"value":1,"message":"hi"}"#.to_string(),
            tool_calls: vec![],
            done: true,
        };
        let parsed: TestAgentOutput = output.try_parse().unwrap();
        assert_eq!(parsed.value, 1);
    }

    #[test]
    fn test_react_agent_output_try_parse_failure() {
        let output = ReActAgentOutput {
            response: "not json".to_string(),
            tool_calls: vec![],
            done: true,
        };
        assert!(output.try_parse::<TestAgentOutput>().is_err());
    }

    #[test]
    fn test_react_agent_output_parse_or_map() {
        let output = ReActAgentOutput {
            response: "plain text".to_string(),
            tool_calls: vec![],
            done: true,
        };
        let result: String = output.parse_or_map(|s| s.to_uppercase());
        assert_eq!(result, "PLAIN TEXT");
    }

    #[test]
    fn test_react_agent_output_into_value() {
        let output = ReActAgentOutput {
            response: "resp".to_string(),
            tool_calls: vec![],
            done: true,
        };
        let val: Value = output.into();
        assert!(val.is_object());
        assert_eq!(val["response"], "resp");
    }

    #[test]
    fn test_react_agent_output_into_string() {
        let output = ReActAgentOutput {
            response: "resp".to_string(),
            tool_calls: vec![],
            done: true,
        };
        let s: String = output.into();
        assert_eq!(s, "resp");
    }

    #[test]
    fn test_error_from_turn_engine_llm() {
        let err: ReActExecutorError = TurnEngineError::LLMError("llm err".to_string()).into();
        assert!(matches!(err, ReActExecutorError::LLMError(_)));
    }

    #[test]
    fn test_error_from_turn_engine_aborted() {
        let err: ReActExecutorError = TurnEngineError::Aborted.into();
        assert!(matches!(err, ReActExecutorError::Other(_)));
    }

    #[test]
    fn test_error_from_turn_engine_other() {
        let err: ReActExecutorError = TurnEngineError::Other("other".to_string()).into();
        assert!(matches!(err, ReActExecutorError::Other(_)));
    }

    #[test]
    fn test_react_agent_creation_and_deref() {
        let mock = MockAgentImpl::new("react_test", "desc");
        let agent = ReActAgent::new(mock);
        assert_eq!(agent.name(), "react_test");
        assert_eq!(agent.description(), "desc");
    }

    #[test]
    fn test_react_agent_clone() {
        let mock = MockAgentImpl::new("clone_test", "desc");
        let agent = ReActAgent::new(mock);
        let cloned = agent.clone();
        assert_eq!(cloned.name(), "clone_test");
    }

    #[test]
    fn test_react_agent_config() {
        let mock = MockAgentImpl::new("cfg_test", "desc");
        let agent = ReActAgent::new(mock);
        assert_eq!(agent.config().max_turns, 10);
    }

    #[tokio::test]
    async fn test_react_agent_execute() {
        use crate::agent::{AgentConfig, Context};
        use autoagents_protocol::ActorID;
        use autoagents_test_utils::llm::MockLLMProvider;

        let mock = MockAgentImpl::new("exec_test", "desc");
        let agent = ReActAgent::new(mock);
        let llm = std::sync::Arc::new(MockLLMProvider {});
        let config = AgentConfig {
            id: ActorID::new_v4(),
            name: "exec_test".to_string(),
            description: "desc".to_string(),
            output_schema: None,
        };
        let context = Arc::new(Context::new(llm, None).with_config(config));
        let task = crate::agent::task::Task::new("test");

        let result = agent.execute(&task, context).await;
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.done);
        assert_eq!(output.response, "Mock response");
    }
}
