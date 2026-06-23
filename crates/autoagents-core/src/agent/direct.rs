use crate::agent::base::AgentType;
use crate::agent::context::Context;
use crate::agent::error::{AgentBuildError, RunnableAgentError};
use crate::agent::executor::event_helper::EventHelper;
use crate::agent::task::Task;
use crate::agent::{AgentBuilder, AgentDeriveT, AgentExecutor, AgentHooks, BaseAgent, HookOutcome};
use crate::error::Error;
use autoagents_protocol::Event;
use serde_json::Value;
use std::sync::Arc;

use crate::agent::constants::DEFAULT_CHANNEL_BUFFER;

use crate::channel::{Receiver, Sender, channel};

#[cfg(not(target_arch = "wasm32"))]
use crate::event_fanout::EventFanout;
use crate::utils::{BoxEventStream, receiver_into_stream};
#[cfg(not(target_arch = "wasm32"))]
use futures_util::stream;

/// Marker type for direct (non-actor) agents.
///
/// Direct agents execute immediately within the caller's task without
/// requiring a runtime or event wiring. Use this for simple one-shot
/// invocations and unit tests.
#[derive(Clone, Copy)]
pub struct DirectAgent {}

impl AgentType for DirectAgent {
    fn type_name() -> &'static str {
        "direct_agent"
    }
}

/// Handle for a direct agent containing the agent instance and an event stream
/// receiver. Use `agent.run(...)` for one-shot calls or `agent.run_stream(...)`
/// to receive streaming outputs.
///
/// Terminal outcomes emit protocol events on [`Self::rx`]: `TaskComplete` on success
/// and `TaskError` on failure (hook abort, executor error, stream setup error,
/// in-stream item errors, and empty streams). For `run_stream()`, `TaskComplete`
/// is emitted when the returned output stream is fully drained; the last successful
/// item is used for the event payload.
pub struct DirectAgentHandle<T: AgentDeriveT + AgentExecutor + AgentHooks + Send + Sync> {
    pub agent: BaseAgent<T, DirectAgent>,
    pub rx: BoxEventStream<Event>,
    #[cfg(not(target_arch = "wasm32"))]
    fanout: Option<EventFanout>,
}

impl<T: AgentDeriveT + AgentExecutor + AgentHooks> DirectAgentHandle<T> {
    pub fn new(agent: BaseAgent<T, DirectAgent>, rx: BoxEventStream<Event>) -> Self {
        Self {
            agent,
            rx,
            #[cfg(not(target_arch = "wasm32"))]
            fanout: None,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn subscribe_events(&mut self) -> BoxEventStream<Event> {
        if let Some(fanout) = &self.fanout {
            return fanout.subscribe();
        }

        let stream = std::mem::replace(&mut self.rx, Box::pin(stream::empty::<Event>()));
        let fanout = EventFanout::new(stream, DEFAULT_CHANNEL_BUFFER);
        self.rx = fanout.subscribe();
        let stream = fanout.subscribe();
        self.fanout = Some(fanout);
        stream
    }
}

impl<T: AgentDeriveT + AgentExecutor + AgentHooks> AgentBuilder<T, DirectAgent> {
    /// Build the BaseAgent and return a wrapper
    #[allow(clippy::result_large_err)]
    pub async fn build(self) -> Result<DirectAgentHandle<T>, Error> {
        let llm = self.llm.ok_or(AgentBuildError::BuildFailure(
            "LLM provider is required".to_string(),
        ))?;
        let (tx, rx): (Sender<Event>, Receiver<Event>) = channel(DEFAULT_CHANNEL_BUFFER);
        let agent: BaseAgent<T, DirectAgent> =
            BaseAgent::<T, DirectAgent>::new(self.inner, llm, self.memory, tx, self.stream).await?;
        let stream = receiver_into_stream(rx);
        Ok(DirectAgentHandle::new(agent, stream))
    }
}

fn wrap_direct_stream_with_terminal_events<T>(
    agent: BaseAgent<T, DirectAgent>,
    stream: crate::utils::BoxRuntimeStream<
        Result<<T as AgentExecutor>::Output, <T as AgentExecutor>::Error>,
    >,
    task: Task,
    context: Arc<Context>,
    tx_event: Option<crate::channel::Sender<Event>>,
    submission_id: autoagents_protocol::SubmissionId,
    actor_id: autoagents_protocol::ActorID,
) -> crate::utils::BoxRuntimeStream<Result<<T as AgentDeriveT>::Output, Error>>
where
    T: AgentDeriveT + AgentExecutor + AgentHooks + Send + Sync,
    Value: From<<T as AgentExecutor>::Output>,
    <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output>,
    <T as AgentExecutor>::Output: Clone,
    <T as AgentExecutor>::Error: Into<RunnableAgentError>,
{
    use futures::StreamExt;

    Box::pin(futures::stream::unfold(
        (
            stream,
            false,
            None::<<T as AgentExecutor>::Output>,
            task,
            context,
        ),
        move |(mut stream, mut saw_error, last, task, context)| {
            let agent = agent.clone_shallow();
            let tx_event = tx_event.clone();
            async move {
                match stream.next().await {
                    Some(result) => {
                        match EventHelper::map_executor_stream_item(
                            &tx_event,
                            submission_id,
                            actor_id,
                            result,
                        )
                        .await
                        {
                            Ok(output) => {
                                let agent_out: <T as AgentDeriveT>::Output = output.clone().into();
                                Some((
                                    Ok(agent_out),
                                    (stream, saw_error, Some(output), task, context),
                                ))
                            }
                            Err(err) => {
                                saw_error = true;
                                Some((
                                    Err(Error::from(err)),
                                    (stream, saw_error, last, task, context),
                                ))
                            }
                        }
                    }
                    None => {
                        if !saw_error {
                            if let Some(executor_out) = last {
                                let _ = agent
                                    .finish_executor_run(
                                        &task,
                                        context.as_ref(),
                                        submission_id,
                                        executor_out,
                                    )
                                    .await;
                            } else {
                                let err = RunnableAgentError::ExecutorError(
                                    "Stream completed without output".to_string(),
                                );
                                #[cfg(not(target_arch = "wasm32"))]
                                EventHelper::send_task_error(
                                    &tx_event,
                                    submission_id,
                                    actor_id,
                                    err.to_string(),
                                )
                                .await;
                            }
                        }
                        None
                    }
                }
            }
        },
    ))
}

impl<T: AgentDeriveT + AgentExecutor + AgentHooks> BaseAgent<T, DirectAgent> {
    /// Execute the agent for a single task and return the final agent output.
    pub async fn run(&self, task: Task) -> Result<<T as AgentDeriveT>::Output, RunnableAgentError>
    where
        Value: From<<T as AgentExecutor>::Output>,
        <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output>,
        <T as AgentExecutor>::Output: Clone,
        <T as AgentExecutor>::Error: Into<RunnableAgentError>,
    {
        let submission_id = task.submission_id;
        let tx_event = self.tx.clone();
        let context = self.create_context();

        //Run Hook
        let hook_outcome = self.inner.on_run_start(&task, &context).await;
        match hook_outcome {
            HookOutcome::Abort => {
                return Err(
                    EventHelper::abort_run_from_hook(&tx_event, submission_id, self.id).await,
                );
            }
            HookOutcome::Continue => {}
        }

        // Execute the agent's logic using the executor
        match self.inner().execute(&task, context.clone()).await {
            Ok(output) => {
                self.finish_executor_run(&task, &context, submission_id, output)
                    .await
            }
            Err(e) => {
                let err: RunnableAgentError = e.into();
                #[cfg(not(target_arch = "wasm32"))]
                EventHelper::send_task_error(&tx_event, submission_id, self.id, err.to_string())
                    .await;
                Err(err)
            }
        }
    }

    /// Execute the agent with streaming enabled and receive a stream of
    /// partial outputs which culminate in a final chunk with `done=true`.
    pub async fn run_stream(
        &self,
        task: Task,
    ) -> Result<
        crate::utils::BoxRuntimeStream<Result<<T as AgentDeriveT>::Output, Error>>,
        RunnableAgentError,
    >
    where
        Value: From<<T as AgentExecutor>::Output>,
        <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output>,
        <T as AgentExecutor>::Output: Clone,
        <T as AgentExecutor>::Error: Into<RunnableAgentError>,
    {
        let submission_id = task.submission_id;
        let tx_event = self.tx.clone();
        let context = self.create_context();

        //Run Hook
        let hook_outcome = self.inner.on_run_start(&task, &context).await;
        match hook_outcome {
            HookOutcome::Abort => {
                return Err(
                    EventHelper::abort_run_from_hook(&tx_event, submission_id, self.id).await,
                );
            }
            HookOutcome::Continue => {}
        }

        // Execute the agent's streaming logic using the executor
        match self.inner().execute_stream(&task, context.clone()).await {
            Ok(stream) => Ok(wrap_direct_stream_with_terminal_events(
                self.clone_shallow(),
                stream,
                task,
                context,
                tx_event,
                submission_id,
                self.id,
            )),
            Err(e) => {
                let err: RunnableAgentError = e.into();
                #[cfg(not(target_arch = "wasm32"))]
                EventHelper::send_task_error(&tx_event, submission_id, self.id, err.to_string())
                    .await;
                Err(err)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::hooks::HookOutcome;
    use crate::agent::output::AgentOutputT;
    use crate::agent::prebuilt::executor::{
        BasicAgent as StableBasicAgent, BasicAgentOutput, ReActAgent as StableReActAgent,
        ReActAgentOutput,
    };
    use crate::agent::task::Task;
    use crate::agent::{Context, ExecutorConfig};
    use crate::tests::{
        ConfigurableLLMProvider, MockAgentImpl, MultiItemStreamAgent, TestAgentOutput, TestError,
    };
    use crate::tool::ToolT;
    use async_trait::async_trait;
    use futures::StreamExt;
    use serde::{Deserialize, Serialize};
    use serde_json::Value;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };

    #[tokio::test]
    async fn test_direct_agent_build_requires_llm() {
        let mock_agent = MockAgentImpl::new("direct", "direct agent");
        let err = match AgentBuilder::<_, DirectAgent>::new(mock_agent)
            .build()
            .await
        {
            Ok(_) => panic!("expected missing llm error"),
            Err(err) => err,
        };

        assert!(matches!(err, crate::error::Error::AgentBuildError(_)));
    }

    #[tokio::test]
    async fn test_direct_agent_run_success() {
        let mock_agent = MockAgentImpl::new("direct", "direct agent");
        let llm = Arc::new(ConfigurableLLMProvider::default());
        let mut handle = AgentBuilder::<_, DirectAgent>::new(mock_agent)
            .llm(llm)
            .build()
            .await
            .expect("build should succeed");

        let task = Task::new("hello");
        let result = handle.agent.run(task).await.expect("run should succeed");
        assert_eq!(result.result, "Processed: hello");

        let event = tokio::time::timeout(std::time::Duration::from_secs(1), handle.rx.next())
            .await
            .expect("timed out waiting for TaskComplete event")
            .expect("stream ended without event");
        match event {
            Event::TaskComplete { result, .. } => {
                let parsed: Value =
                    serde_json::from_str(&result).expect("TaskComplete result should be JSON");
                assert_eq!(parsed["result"], "Processed: hello");
            }
            other => panic!("expected TaskComplete, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_direct_agent_run_executor_error() {
        let mock_agent = MockAgentImpl::new("direct", "direct agent").with_failure(true);
        let llm = Arc::new(ConfigurableLLMProvider::default());
        let mut handle = AgentBuilder::<_, DirectAgent>::new(mock_agent)
            .llm(llm)
            .build()
            .await
            .expect("build should succeed");

        let task = Task::new("fail");
        let err = handle.agent.run(task).await.expect_err("expected error");
        assert!(matches!(err, RunnableAgentError::ExecutorError(_)));

        let event = tokio::time::timeout(std::time::Duration::from_secs(1), handle.rx.next())
            .await
            .expect("timed out waiting for TaskError event")
            .expect("stream ended without event");
        match event {
            Event::TaskError { error, .. } => {
                assert!(error.contains("Mock execution failed"));
            }
            other => panic!("expected TaskError, got {other:?}"),
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct HookCountOutput {
        result: String,
    }

    impl AgentOutputT for HookCountOutput {
        fn output_schema() -> &'static str {
            r#"{"type":"object","properties":{"result":{"type":"string"}},"required":["result"]}"#
        }

        fn structured_output_format() -> Value {
            serde_json::json!({
                "name": "HookCountOutput",
                "description": "Hook count output",
                "schema": {
                    "type": "object",
                    "properties": {
                        "result": {"type": "string"}
                    },
                    "required": ["result"]
                },
                "strict": true
            })
        }
    }

    impl From<BasicAgentOutput> for HookCountOutput {
        fn from(output: BasicAgentOutput) -> Self {
            Self {
                result: output.response,
            }
        }
    }

    impl From<ReActAgentOutput> for HookCountOutput {
        fn from(output: ReActAgentOutput) -> Self {
            Self {
                result: output.response,
            }
        }
    }

    #[derive(Debug, Clone)]
    struct CountingHookAgent {
        on_run_start_calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl AgentDeriveT for CountingHookAgent {
        type Output = HookCountOutput;

        fn description(&self) -> &'static str {
            "counting hook agent"
        }

        fn output_schema(&self) -> Option<Value> {
            Some(serde_json::json!({
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"]
            }))
        }

        fn name(&self) -> &'static str {
            "counting_hook_agent"
        }

        fn tools(&self) -> Vec<Box<dyn ToolT>> {
            vec![]
        }
    }

    #[async_trait]
    impl AgentHooks for CountingHookAgent {
        async fn on_run_start(&self, _task: &Task, _ctx: &Context) -> HookOutcome {
            self.on_run_start_calls.fetch_add(1, Ordering::SeqCst);
            HookOutcome::Continue
        }
    }

    #[tokio::test]
    async fn test_direct_basic_agent_run_calls_on_run_start_once() {
        let calls = Arc::new(AtomicUsize::new(0));
        let llm = Arc::new(ConfigurableLLMProvider::default());
        let handle =
            AgentBuilder::<_, DirectAgent>::new(StableBasicAgent::new(CountingHookAgent {
                on_run_start_calls: Arc::clone(&calls),
            }))
            .llm(llm)
            .build()
            .await
            .expect("build should succeed");

        let task = Task::new("hello");
        let result = handle.agent.run(task).await.expect("run should succeed");

        assert_eq!(result.result, "Mock response");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_direct_react_agent_run_calls_on_run_start_once() {
        let calls = Arc::new(AtomicUsize::new(0));
        let llm = Arc::new(ConfigurableLLMProvider::default());
        let handle =
            AgentBuilder::<_, DirectAgent>::new(StableReActAgent::new(CountingHookAgent {
                on_run_start_calls: Arc::clone(&calls),
            }))
            .llm(llm)
            .build()
            .await
            .expect("build should succeed");

        let task = Task::new("hello");
        let result = handle.agent.run(task).await.expect("run should succeed");

        assert_eq!(result.result, "Mock response");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[derive(Clone, Debug)]
    struct StreamAgent;

    #[async_trait]
    impl AgentDeriveT for StreamAgent {
        type Output = TestAgentOutput;

        fn description(&self) -> &'static str {
            "stream agent"
        }

        fn output_schema(&self) -> Option<Value> {
            Some(TestAgentOutput::structured_output_format())
        }

        fn name(&self) -> &'static str {
            "stream_agent"
        }

        fn tools(&self) -> Vec<Box<dyn ToolT>> {
            vec![]
        }
    }

    #[async_trait]
    impl AgentExecutor for StreamAgent {
        type Output = TestAgentOutput;
        type Error = TestError;

        fn config(&self) -> ExecutorConfig {
            ExecutorConfig::default()
        }

        async fn execute(
            &self,
            task: &Task,
            _context: Arc<Context>,
        ) -> Result<Self::Output, Self::Error> {
            Ok(TestAgentOutput {
                result: format!("Streamed: {}", task.prompt),
            })
        }
    }

    impl AgentHooks for StreamAgent {}

    #[tokio::test]
    async fn test_direct_agent_run_stream_default_executes_once() {
        let llm = Arc::new(ConfigurableLLMProvider::default());
        let mut handle = AgentBuilder::<_, DirectAgent>::new(StreamAgent)
            .llm(llm)
            .build()
            .await
            .expect("build should succeed");

        let task = Task::new("stream");
        let stream = handle
            .agent
            .run_stream(task)
            .await
            .expect("stream should succeed");
        let outputs: Vec<_> = stream.collect().await;
        assert_eq!(outputs.len(), 1);
        let output = outputs[0].as_ref().expect("expected Ok output");
        assert_eq!(output.result, "Streamed: stream");

        let event = tokio::time::timeout(std::time::Duration::from_secs(1), handle.rx.next())
            .await
            .expect("timed out waiting for TaskComplete event")
            .expect("stream ended without event");
        match event {
            Event::TaskComplete { result, .. } => {
                let parsed: Value =
                    serde_json::from_str(&result).expect("TaskComplete result should be JSON");
                assert_eq!(parsed["result"], "Streamed: stream");
            }
            other => panic!("expected TaskComplete, got {other:?}"),
        }
    }

    #[derive(Debug)]
    struct AbortAgent {
        executed: Arc<AtomicBool>,
    }

    #[async_trait]
    impl AgentDeriveT for AbortAgent {
        type Output = TestAgentOutput;

        fn description(&self) -> &'static str {
            "abort agent"
        }

        fn output_schema(&self) -> Option<Value> {
            Some(TestAgentOutput::structured_output_format())
        }

        fn name(&self) -> &'static str {
            "abort_agent"
        }

        fn tools(&self) -> Vec<Box<dyn ToolT>> {
            vec![]
        }
    }

    #[async_trait]
    impl AgentExecutor for AbortAgent {
        type Output = TestAgentOutput;
        type Error = TestError;

        fn config(&self) -> ExecutorConfig {
            ExecutorConfig::default()
        }

        async fn execute(
            &self,
            _task: &Task,
            _context: Arc<Context>,
        ) -> Result<Self::Output, Self::Error> {
            self.executed.store(true, Ordering::SeqCst);
            Ok(TestAgentOutput {
                result: "should-not-run".to_string(),
            })
        }
    }

    #[async_trait]
    impl AgentHooks for AbortAgent {
        async fn on_run_start(&self, _task: &Task, _ctx: &Context) -> HookOutcome {
            HookOutcome::Abort
        }
    }

    #[tokio::test]
    async fn test_direct_agent_run_aborts_before_execute() {
        let executed = Arc::new(AtomicBool::new(false));
        let agent = AbortAgent {
            executed: Arc::clone(&executed),
        };
        let llm = Arc::new(ConfigurableLLMProvider::default());
        let mut handle = AgentBuilder::<_, DirectAgent>::new(agent)
            .llm(llm)
            .build()
            .await
            .expect("build should succeed");

        let task = Task::new("abort");
        let err = handle.agent.run(task).await.expect_err("expected abort");
        assert!(matches!(err, RunnableAgentError::Abort));
        assert!(!executed.load(Ordering::SeqCst));

        let event = tokio::time::timeout(std::time::Duration::from_secs(1), handle.rx.next())
            .await
            .expect("timed out waiting for TaskError event")
            .expect("stream ended without event");
        match event {
            Event::TaskError { error, .. } => {
                assert!(error.contains("Abort"));
            }
            other => panic!("expected TaskError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_direct_agent_run_stream_aborts_before_execute_stream() {
        let executed = Arc::new(AtomicBool::new(false));
        let agent = AbortAgent {
            executed: Arc::clone(&executed),
        };
        let llm = Arc::new(ConfigurableLLMProvider::default());
        let mut handle = AgentBuilder::<_, DirectAgent>::new(agent)
            .llm(llm)
            .stream(true)
            .build()
            .await
            .expect("build should succeed");

        let task = Task::new("abort");
        let err = match handle.agent.run_stream(task).await {
            Err(err) => err,
            Ok(_) => panic!("expected abort"),
        };
        assert!(matches!(err, RunnableAgentError::Abort));
        assert!(!executed.load(Ordering::SeqCst));

        let event = tokio::time::timeout(std::time::Duration::from_secs(1), handle.rx.next())
            .await
            .expect("timed out waiting for TaskError event")
            .expect("stream ended without event");
        match event {
            Event::TaskError { error, .. } => {
                assert!(error.contains("Abort"));
            }
            other => panic!("expected TaskError, got {other:?}"),
        }
    }

    #[derive(Debug, Clone)]
    struct FailingStreamSetupAgent;

    #[async_trait]
    impl AgentDeriveT for FailingStreamSetupAgent {
        type Output = TestAgentOutput;

        fn description(&self) -> &'static str {
            "failing stream setup"
        }

        fn output_schema(&self) -> Option<Value> {
            Some(TestAgentOutput::structured_output_format())
        }

        fn name(&self) -> &'static str {
            "failing_stream_setup"
        }

        fn tools(&self) -> Vec<Box<dyn ToolT>> {
            vec![]
        }
    }

    #[async_trait]
    impl AgentExecutor for FailingStreamSetupAgent {
        type Output = TestAgentOutput;
        type Error = TestError;

        fn config(&self) -> ExecutorConfig {
            ExecutorConfig::default()
        }

        async fn execute(
            &self,
            _task: &Task,
            _context: Arc<Context>,
        ) -> Result<Self::Output, Self::Error> {
            Ok(TestAgentOutput {
                result: "unused".to_string(),
            })
        }

        async fn execute_stream(
            &self,
            _task: &Task,
            _context: Arc<Context>,
        ) -> Result<crate::utils::BoxRuntimeStream<Result<Self::Output, Self::Error>>, Self::Error>
        {
            Err(TestError::ExecutionFailed(
                "stream setup failed".to_string(),
            ))
        }
    }

    impl AgentHooks for FailingStreamSetupAgent {}

    #[tokio::test]
    async fn test_direct_agent_run_stream_setup_error_emits_task_error() {
        let llm = Arc::new(ConfigurableLLMProvider::default());
        let mut handle = AgentBuilder::<_, DirectAgent>::new(FailingStreamSetupAgent)
            .llm(llm)
            .stream(true)
            .build()
            .await
            .expect("build should succeed");

        let err = match handle.agent.run_stream(Task::new("fail setup")).await {
            Err(err) => err,
            Ok(_) => panic!("expected stream setup failure"),
        };
        assert!(matches!(err, RunnableAgentError::ExecutorError(_)));

        let event = tokio::time::timeout(std::time::Duration::from_secs(1), handle.rx.next())
            .await
            .expect("timed out waiting for TaskError event")
            .expect("stream ended without event");
        match event {
            Event::TaskError { error, .. } => {
                assert!(error.contains("stream setup failed"));
            }
            other => panic!("expected TaskError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_direct_agent_run_stream_item_error_emits_task_error() {
        let mock_agent = MockAgentImpl::new("direct", "direct agent").with_failure(true);
        let llm = Arc::new(ConfigurableLLMProvider::default());
        let mut handle = AgentBuilder::<_, DirectAgent>::new(mock_agent)
            .llm(llm)
            .stream(true)
            .build()
            .await
            .expect("build should succeed");

        let mut stream = handle
            .agent
            .run_stream(Task::new("fail"))
            .await
            .expect("default execute_stream should return Ok stream");

        let err = stream
            .next()
            .await
            .expect("stream should yield one item")
            .expect_err("expected stream item error");
        assert!(matches!(err, Error::RunnableAgentError(_)));

        let event = tokio::time::timeout(std::time::Duration::from_secs(1), handle.rx.next())
            .await
            .expect("timed out waiting for TaskError event")
            .expect("stream ended without event");
        match event {
            Event::TaskError { error, .. } => {
                assert!(error.contains("Mock execution failed"));
            }
            other => panic!("expected TaskError, got {other:?}"),
        }
    }

    #[derive(Debug, Clone)]
    struct StreamItemErrorAgent;

    #[async_trait]
    impl AgentDeriveT for StreamItemErrorAgent {
        type Output = TestAgentOutput;

        fn description(&self) -> &'static str {
            "stream item error agent"
        }

        fn output_schema(&self) -> Option<Value> {
            Some(TestAgentOutput::structured_output_format())
        }

        fn name(&self) -> &'static str {
            "stream_item_error"
        }

        fn tools(&self) -> Vec<Box<dyn ToolT>> {
            vec![]
        }
    }

    #[async_trait]
    impl AgentExecutor for StreamItemErrorAgent {
        type Output = TestAgentOutput;
        type Error = TestError;

        fn config(&self) -> ExecutorConfig {
            ExecutorConfig::default()
        }

        async fn execute(
            &self,
            _task: &Task,
            _context: Arc<Context>,
        ) -> Result<Self::Output, Self::Error> {
            Ok(TestAgentOutput {
                result: "unused".to_string(),
            })
        }

        async fn execute_stream(
            &self,
            _task: &Task,
            _context: Arc<Context>,
        ) -> Result<crate::utils::BoxRuntimeStream<Result<Self::Output, Self::Error>>, Self::Error>
        {
            Ok(Box::pin(futures::stream::iter([Err(
                TestError::ExecutionFailed("stream item failed".to_string()),
            )])))
        }
    }

    impl AgentHooks for StreamItemErrorAgent {}

    #[tokio::test]
    async fn test_direct_agent_run_stream_custom_item_error_emits_task_error() {
        let llm = Arc::new(ConfigurableLLMProvider::default());
        let mut handle = AgentBuilder::<_, DirectAgent>::new(StreamItemErrorAgent)
            .llm(llm)
            .stream(true)
            .build()
            .await
            .expect("build should succeed");

        let mut stream = handle
            .agent
            .run_stream(Task::new("stream error"))
            .await
            .expect("stream should start");

        let err = stream
            .next()
            .await
            .expect("stream should yield one item")
            .expect_err("expected stream item failure");
        assert!(matches!(err, Error::RunnableAgentError(_)));

        let event = tokio::time::timeout(std::time::Duration::from_secs(1), handle.rx.next())
            .await
            .expect("timed out waiting for TaskError event")
            .expect("stream ended without event");
        match event {
            Event::TaskError { error, .. } => {
                assert!(error.contains("stream item failed"));
            }
            other => panic!("expected TaskError, got {other:?}"),
        }
    }

    #[derive(Debug, Clone)]
    struct EmptyStreamAgent;

    #[async_trait]
    impl AgentDeriveT for EmptyStreamAgent {
        type Output = TestAgentOutput;

        fn description(&self) -> &'static str {
            "empty stream agent"
        }

        fn output_schema(&self) -> Option<Value> {
            Some(TestAgentOutput::structured_output_format())
        }

        fn name(&self) -> &'static str {
            "empty_stream_agent"
        }

        fn tools(&self) -> Vec<Box<dyn ToolT>> {
            vec![]
        }
    }

    #[async_trait]
    impl AgentExecutor for EmptyStreamAgent {
        type Output = TestAgentOutput;
        type Error = TestError;

        fn config(&self) -> ExecutorConfig {
            ExecutorConfig::default()
        }

        async fn execute(
            &self,
            _task: &Task,
            _context: Arc<Context>,
        ) -> Result<Self::Output, Self::Error> {
            Ok(TestAgentOutput {
                result: "unused".to_string(),
            })
        }

        async fn execute_stream(
            &self,
            _task: &Task,
            _context: Arc<Context>,
        ) -> Result<crate::utils::BoxRuntimeStream<Result<Self::Output, Self::Error>>, Self::Error>
        {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    impl AgentHooks for EmptyStreamAgent {}

    #[tokio::test]
    async fn test_direct_agent_run_stream_empty_stream_emits_task_error() {
        let llm = Arc::new(ConfigurableLLMProvider::default());
        let mut handle = AgentBuilder::<_, DirectAgent>::new(EmptyStreamAgent)
            .llm(llm)
            .stream(true)
            .build()
            .await
            .expect("build should succeed");

        let stream = handle
            .agent
            .run_stream(Task::new("empty"))
            .await
            .expect("stream should start");
        let outputs: Vec<_> = stream.collect().await;
        assert!(outputs.is_empty());

        let event = tokio::time::timeout(std::time::Duration::from_secs(1), handle.rx.next())
            .await
            .expect("timed out waiting for TaskError event")
            .expect("stream ended without event");
        match event {
            Event::TaskError { error, .. } => {
                assert!(error.contains("without output"));
            }
            other => panic!("expected TaskError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_direct_agent_run_stream_uses_last_item_for_task_complete() {
        let llm = Arc::new(ConfigurableLLMProvider::default());
        let mut handle = AgentBuilder::<_, DirectAgent>::new(MultiItemStreamAgent)
            .llm(llm)
            .stream(true)
            .build()
            .await
            .expect("build should succeed");

        let stream = handle
            .agent
            .run_stream(Task::new("chunked"))
            .await
            .expect("stream should start");
        let outputs: Vec<_> = stream.collect().await;
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[2].as_ref().expect("third item").result, "chunked-3");

        let event = tokio::time::timeout(std::time::Duration::from_secs(1), handle.rx.next())
            .await
            .expect("timed out waiting for TaskComplete event")
            .expect("stream ended without event");
        match event {
            Event::TaskComplete { result, .. } => {
                let parsed: Value =
                    serde_json::from_str(&result).expect("TaskComplete result should be JSON");
                assert_eq!(parsed["sequence"], 3);
                assert_eq!(parsed["response"], "chunked-3");
            }
            other => panic!("expected TaskComplete, got {other:?}"),
        }
    }

    #[derive(Debug, Clone)]
    struct OkThenErrStreamAgent;

    #[async_trait]
    impl AgentDeriveT for OkThenErrStreamAgent {
        type Output = TestAgentOutput;

        fn description(&self) -> &'static str {
            "ok then err stream agent"
        }

        fn output_schema(&self) -> Option<Value> {
            Some(TestAgentOutput::structured_output_format())
        }

        fn name(&self) -> &'static str {
            "ok_then_err_stream_agent"
        }

        fn tools(&self) -> Vec<Box<dyn ToolT>> {
            vec![]
        }
    }

    #[async_trait]
    impl AgentExecutor for OkThenErrStreamAgent {
        type Output = TestAgentOutput;
        type Error = TestError;

        fn config(&self) -> ExecutorConfig {
            ExecutorConfig::default()
        }

        async fn execute(
            &self,
            _task: &Task,
            _context: Arc<Context>,
        ) -> Result<Self::Output, Self::Error> {
            Ok(TestAgentOutput {
                result: "unused".to_string(),
            })
        }

        async fn execute_stream(
            &self,
            _task: &Task,
            _context: Arc<Context>,
        ) -> Result<crate::utils::BoxRuntimeStream<Result<Self::Output, Self::Error>>, Self::Error>
        {
            Ok(Box::pin(futures::stream::iter([
                Ok(TestAgentOutput {
                    result: "partial".to_string(),
                }),
                Err(TestError::ExecutionFailed("stream item failed".to_string())),
            ])))
        }
    }

    impl AgentHooks for OkThenErrStreamAgent {}

    #[tokio::test]
    async fn test_direct_agent_run_stream_ok_then_err_emits_task_error_not_task_complete() {
        let llm = Arc::new(ConfigurableLLMProvider::default());
        let mut handle = AgentBuilder::<_, DirectAgent>::new(OkThenErrStreamAgent)
            .llm(llm)
            .stream(true)
            .build()
            .await
            .expect("build should succeed");

        let mut stream = handle
            .agent
            .run_stream(Task::new("partial failure"))
            .await
            .expect("stream should start");

        let first = stream
            .next()
            .await
            .expect("stream should yield ok item")
            .expect("expected ok item");
        assert_eq!(first.result, "partial");

        let second = stream
            .next()
            .await
            .expect("stream should yield err item")
            .expect_err("expected err item");
        assert!(matches!(second, Error::RunnableAgentError(_)));

        assert!(
            stream.next().await.is_none(),
            "stream should end after error item"
        );

        let event = tokio::time::timeout(std::time::Duration::from_secs(1), handle.rx.next())
            .await
            .expect("timed out waiting for TaskError event")
            .expect("stream ended without event");
        match event {
            Event::TaskError { error, .. } => {
                assert!(error.contains("stream item failed"));
            }
            other => panic!("expected TaskError, got {other:?}"),
        }

        let no_terminal_success =
            tokio::time::timeout(std::time::Duration::from_millis(100), handle.rx.next()).await;
        assert!(
            no_terminal_success.is_err(),
            "Ok-then-Err stream should not emit TaskComplete after TaskError"
        );
    }
}
