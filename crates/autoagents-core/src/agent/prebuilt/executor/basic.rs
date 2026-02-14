use crate::agent::executor::event_helper::EventHelper;
use crate::agent::executor::turn_engine::{
    TurnDelta, TurnEngine, TurnEngineConfig, TurnEngineError, TurnEngineOutput, record_task_state,
};
use crate::agent::hooks::HookOutcome;
use crate::agent::task::Task;
use crate::agent::{AgentDeriveT, AgentExecutor, AgentHooks, Context, ExecutorConfig};
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

/// Output of the Basic executor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicAgentOutput {
    pub response: String,
    pub done: bool,
}

impl From<BasicAgentOutput> for Value {
    fn from(output: BasicAgentOutput) -> Self {
        serde_json::to_value(output).unwrap_or(Value::Null)
    }
}
impl From<BasicAgentOutput> for String {
    fn from(output: BasicAgentOutput) -> Self {
        output.response
    }
}

impl BasicAgentOutput {
    /// Try to parse the response string as structured JSON of type `T`.
    /// Returns `serde_json::Error` if parsing fails.
    pub fn try_parse<T: for<'de> serde::Deserialize<'de>>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_str::<T>(&self.response)
    }

    /// Parse the response string as structured JSON of type `T`, or map the raw
    /// text into `T` using the provided fallback function if parsing fails.
    pub fn parse_or_map<T, F>(&self, fallback: F) -> T
    where
        T: for<'de> serde::Deserialize<'de>,
        F: FnOnce(&str) -> T,
    {
        self.try_parse::<T>()
            .unwrap_or_else(|_| fallback(&self.response))
    }
}

/// Error type for Basic executor
#[derive(Debug, thiserror::Error)]
pub enum BasicExecutorError {
    #[error("LLM error: {0}")]
    LLMError(String),

    #[error("Other error: {0}")]
    Other(String),
}

impl From<TurnEngineError> for BasicExecutorError {
    fn from(error: TurnEngineError) -> Self {
        match error {
            TurnEngineError::LLMError(err) => BasicExecutorError::LLMError(err),
            TurnEngineError::Aborted => {
                BasicExecutorError::Other("Run aborted by hook".to_string())
            }
            TurnEngineError::Other(err) => BasicExecutorError::Other(err),
        }
    }
}

/// Wrapper type for the single-turn Basic executor.
///
/// Use `BasicAgent<T>` when you want a single request/response interaction
/// with optional streaming but without tool calling or multi-turn loops.
#[derive(Debug)]
pub struct BasicAgent<T: AgentDeriveT> {
    inner: Arc<T>,
}

impl<T: AgentDeriveT> Clone for BasicAgent<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T: AgentDeriveT> BasicAgent<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

impl<T: AgentDeriveT> Deref for BasicAgent<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Implement AgentDeriveT for the wrapper by delegating to the inner type
#[async_trait]
impl<T: AgentDeriveT> AgentDeriveT for BasicAgent<T> {
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
impl<T> AgentHooks for BasicAgent<T>
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

/// Implementation of AgentExecutor for the BasicExecutorWrapper
#[async_trait]
impl<T: AgentDeriveT + AgentHooks> AgentExecutor for BasicAgent<T> {
    type Output = BasicAgentOutput;
    type Error = BasicExecutorError;

    fn config(&self) -> ExecutorConfig {
        ExecutorConfig { max_turns: 1 }
    }

    async fn execute(
        &self,
        task: &Task,
        context: Arc<Context>,
    ) -> Result<Self::Output, Self::Error> {
        if self.on_run_start(task, &context).await == HookOutcome::Abort {
            return Err(BasicExecutorError::Other("Run aborted by hook".to_string()));
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

        let engine = TurnEngine::new(TurnEngineConfig::basic(self.config().max_turns));
        let mut turn_state = engine.turn_state(&context);
        let turn_result = engine
            .run_turn(
                self,
                task,
                &context,
                &mut turn_state,
                0,
                self.config().max_turns,
            )
            .await?;

        let output = extract_turn_output(turn_result);

        EventHelper::send_task_completed(
            &tx_event,
            task.submission_id,
            context.config().id,
            context.config().name.clone(),
            output.response.clone(),
        )
        .await;

        Ok(BasicAgentOutput {
            response: output.response,
            done: true,
        })
    }

    async fn execute_stream(
        &self,
        task: &Task,
        context: Arc<Context>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Self::Output, Self::Error>> + Send>>, Self::Error>
    {
        if self.on_run_start(task, &context).await == HookOutcome::Abort {
            return Err(BasicExecutorError::Other("Run aborted by hook".to_string()));
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

        let engine = TurnEngine::new(TurnEngineConfig::basic(self.config().max_turns));
        let mut turn_state = engine.turn_state(&context);
        let context_clone = context.clone();
        let task = task.clone();
        let executor = self.clone();

        let (tx, rx) = channel::<Result<BasicAgentOutput, BasicExecutorError>>(100);

        spawn_future(async move {
            let turn_stream = engine
                .run_turn_stream(
                    executor,
                    &task,
                    context_clone.clone(),
                    &mut turn_state,
                    0,
                    1,
                )
                .await;

            let mut final_response = String::default();

            match turn_stream {
                Ok(mut stream) => {
                    use futures::StreamExt;
                    while let Some(delta_result) = stream.next().await {
                        match delta_result {
                            Ok(TurnDelta::Text(content)) => {
                                let _ = tx
                                    .send(Ok(BasicAgentOutput {
                                        response: content,
                                        done: false,
                                    }))
                                    .await;
                            }
                            Ok(TurnDelta::ToolResults(_)) => {}
                            Ok(TurnDelta::Done(result)) => {
                                let output = extract_turn_output(result);
                                final_response = output.response.clone();
                                let _ = tx
                                    .send(Ok(BasicAgentOutput {
                                        response: output.response,
                                        done: true,
                                    }))
                                    .await;
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

            let tx_event = context_clone.tx().ok();
            EventHelper::send_stream_complete(&tx_event, task.submission_id).await;
            EventHelper::send_task_completed(
                &tx_event,
                task.submission_id,
                context_clone.config().id,
                context_clone.config().name.clone(),
                final_response,
            )
            .await;
        });

        Ok(receiver_into_stream(rx))
    }
}

fn extract_turn_output(
    result: crate::agent::executor::TurnResult<TurnEngineOutput>,
) -> TurnEngineOutput {
    match result {
        crate::agent::executor::TurnResult::Complete(output) => output,
        crate::agent::executor::TurnResult::Continue(Some(output)) => output,
        crate::agent::executor::TurnResult::Continue(None) => TurnEngineOutput {
            response: String::default(),
            tool_calls: Vec::default(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentDeriveT;
    use crate::tests::agent::MockAgentImpl;
    use autoagents_test_utils::llm::MockLLMProvider;
    use std::sync::Arc;

    #[test]
    fn test_basic_agent_creation() {
        let mock_agent = MockAgentImpl::new("test_agent", "Test agent description");
        let basic_agent = BasicAgent::new(mock_agent);

        assert_eq!(basic_agent.name(), "test_agent");
        assert_eq!(basic_agent.description(), "Test agent description");
    }

    #[test]
    fn test_basic_agent_clone() {
        let mock_agent = MockAgentImpl::new("test_agent", "Test agent description");
        let basic_agent = BasicAgent::new(mock_agent);
        let cloned_agent = basic_agent.clone();

        assert_eq!(cloned_agent.name(), "test_agent");
        assert_eq!(cloned_agent.description(), "Test agent description");
    }

    #[test]
    fn test_basic_agent_output_conversions() {
        let output = BasicAgentOutput {
            response: "Test response".to_string(),
            done: true,
        };

        // Test conversion to Value
        let value: Value = output.clone().into();
        assert!(value.is_object());

        // Test conversion to String
        let string: String = output.into();
        assert_eq!(string, "Test response");
    }

    #[tokio::test]
    async fn test_basic_agent_execute() {
        use crate::agent::task::Task;
        use crate::agent::{AgentConfig, Context};
        use autoagents_protocol::ActorID;

        let mock_agent = MockAgentImpl::new("test_agent", "Test agent description");
        let basic_agent = BasicAgent::new(mock_agent);

        let llm = Arc::new(MockLLMProvider {});
        let config = AgentConfig {
            id: ActorID::new_v4(),
            name: "test_agent".to_string(),
            description: "Test agent description".to_string(),
            output_schema: None,
        };

        let context = Context::new(llm, None).with_config(config);

        let context_arc = Arc::new(context);
        let task = Task::new("Test task");
        let result = basic_agent.execute(&task, context_arc).await;

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.response, "Mock response");
        assert!(output.done);
    }

    #[test]
    fn test_executor_config() {
        let mock_agent = MockAgentImpl::new("test_agent", "Test agent description");
        let basic_agent = BasicAgent::new(mock_agent);

        let config = basic_agent.config();
        assert_eq!(config.max_turns, 1);
    }

    #[test]
    fn test_basic_agent_output_try_parse_success() {
        let output = BasicAgentOutput {
            response: r#"{"name":"test","value":42}"#.to_string(),
            done: true,
        };
        #[derive(serde::Deserialize, PartialEq, Debug)]
        struct Data {
            name: String,
            value: i32,
        }
        let parsed: Data = output.try_parse().unwrap();
        assert_eq!(
            parsed,
            Data {
                name: "test".to_string(),
                value: 42
            }
        );
    }

    #[test]
    fn test_basic_agent_output_try_parse_failure() {
        let output = BasicAgentOutput {
            response: "not json".to_string(),
            done: true,
        };
        let result = output.try_parse::<serde_json::Value>();
        assert!(result.is_err());
    }

    #[test]
    fn test_basic_agent_output_parse_or_map_fallback() {
        let output = BasicAgentOutput {
            response: "plain text".to_string(),
            done: true,
        };
        let result: String = output.parse_or_map(|s| s.to_uppercase());
        assert_eq!(result, "PLAIN TEXT");
    }

    #[test]
    fn test_basic_agent_output_parse_or_map_success() {
        let output = BasicAgentOutput {
            response: r#""hello""#.to_string(),
            done: true,
        };
        let result: String = output.parse_or_map(|s| s.to_uppercase());
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_error_from_turn_engine_llm() {
        let err: BasicExecutorError = TurnEngineError::LLMError("bad".to_string()).into();
        assert!(matches!(err, BasicExecutorError::LLMError(_)));
        assert!(err.to_string().contains("bad"));
    }

    #[test]
    fn test_error_from_turn_engine_aborted() {
        let err: BasicExecutorError = TurnEngineError::Aborted.into();
        assert!(matches!(err, BasicExecutorError::Other(_)));
        assert!(err.to_string().contains("aborted"));
    }

    #[test]
    fn test_error_from_turn_engine_other() {
        let err: BasicExecutorError = TurnEngineError::Other("misc".to_string()).into();
        assert!(matches!(err, BasicExecutorError::Other(_)));
        assert!(err.to_string().contains("misc"));
    }

    #[test]
    fn test_extract_turn_output_complete() {
        let result = crate::agent::executor::TurnResult::Complete(
            crate::agent::executor::turn_engine::TurnEngineOutput {
                response: "done".to_string(),
                tool_calls: Vec::new(),
            },
        );
        let output = extract_turn_output(result);
        assert_eq!(output.response, "done");
    }

    #[test]
    fn test_extract_turn_output_continue_some() {
        let result = crate::agent::executor::TurnResult::Continue(Some(
            crate::agent::executor::turn_engine::TurnEngineOutput {
                response: "partial".to_string(),
                tool_calls: Vec::new(),
            },
        ));
        let output = extract_turn_output(result);
        assert_eq!(output.response, "partial");
    }

    #[test]
    fn test_extract_turn_output_continue_none() {
        let result = crate::agent::executor::TurnResult::Continue(None);
        let output = extract_turn_output(result);
        assert!(output.response.is_empty());
        assert!(output.tool_calls.is_empty());
    }
}
