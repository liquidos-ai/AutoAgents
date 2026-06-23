#[cfg(not(target_arch = "wasm32"))]
use crate::actor::Topic;
use crate::agent::base::AgentType;
use crate::agent::context::Context;
use crate::agent::error::{AgentBuildError, RunnableAgentError};
use crate::agent::executor::event_helper::EventHelper;
use crate::agent::hooks::AgentHooks;
use crate::agent::state::AgentState;
use crate::agent::task::Task;
use crate::agent::{AgentBuilder, AgentDeriveT, AgentExecutor, BaseAgent, HookOutcome};
use crate::channel::Sender;
use crate::error::Error;
#[cfg(not(target_arch = "wasm32"))]
use crate::runtime::TypedRuntime;
use async_trait::async_trait;
use autoagents_protocol::{Event, SubmissionId};
#[cfg(target_arch = "wasm32")]
use futures::SinkExt;
#[cfg(not(target_arch = "wasm32"))]
use ractor::Actor;
#[cfg(not(target_arch = "wasm32"))]
use ractor::{ActorProcessingErr, ActorRef};
use serde_json::Value;
use std::fmt::Debug;
use std::sync::Arc;

/// Marker type for actor-based agents.
///
/// Actor agents run inside a runtime, can subscribe to topics, receive
/// messages, and emit protocol `Event`s for streaming updates.
pub struct ActorAgent {}

impl AgentType for ActorAgent {
    fn type_name() -> &'static str {
        "protocol_agent"
    }
}

/// Handle for an actor-based agent that contains both the agent and the
/// address of its actor. Use `addr()` to send messages directly or publish
/// `Task`s to subscribed `Topic<Task>`.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
pub struct ActorAgentHandle<T: AgentDeriveT + AgentExecutor + AgentHooks + Send + Sync> {
    pub agent: Arc<BaseAgent<T, ActorAgent>>,
    pub actor_ref: ActorRef<Task>,
}

#[cfg(not(target_arch = "wasm32"))]
impl<T: AgentDeriveT + AgentExecutor + AgentHooks> ActorAgentHandle<T> {
    /// Get the actor reference (`ActorRef<Task>`) for direct messaging.
    pub fn addr(&self) -> ActorRef<Task> {
        self.actor_ref.clone()
    }

    /// Get a clone of the agent reference for querying metadata or invoking
    /// methods that require `Arc<BaseAgent<..>>`.
    pub fn agent(&self) -> Arc<BaseAgent<T, ActorAgent>> {
        self.agent.clone()
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<T: AgentDeriveT + AgentExecutor + AgentHooks> Debug for ActorAgentHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentHandle")
            .field("agent", &self.agent)
            .finish()
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub struct AgentActor<T: AgentDeriveT + AgentExecutor + AgentHooks>(
    pub Arc<BaseAgent<T, ActorAgent>>,
);

#[cfg(not(target_arch = "wasm32"))]
impl<T: AgentDeriveT + AgentExecutor + AgentHooks> AgentActor<T> {}

#[cfg(not(target_arch = "wasm32"))]
impl<T: AgentDeriveT + AgentExecutor + AgentHooks> AgentBuilder<T, ActorAgent>
where
    T: Send + Sync + 'static,
    serde_json::Value: From<<T as AgentExecutor>::Output>,
    <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output>,
    <T as AgentExecutor>::Error: Into<RunnableAgentError>,
{
    /// Build the BaseAgent and return a wrapper that includes the actor reference
    pub async fn build(self) -> Result<ActorAgentHandle<T>, Error> {
        let llm = self.llm.ok_or(AgentBuildError::BuildFailure(
            "LLM provider is required".to_string(),
        ))?;
        let runtime = self.runtime.ok_or(AgentBuildError::BuildFailure(
            "Runtime should be defined".into(),
        ))?;
        let tx = runtime.tx();

        let agent: Arc<BaseAgent<T, ActorAgent>> = Arc::new(
            BaseAgent::<T, ActorAgent>::new(self.inner, llm, self.memory, tx, self.stream).await?,
        );

        // Create agent actor
        let agent_actor = AgentActor(agent.clone());
        let actor_ref = Actor::spawn(Some(agent_actor.0.name().into()), agent_actor, ())
            .await
            .map_err(AgentBuildError::SpawnError)?
            .0;

        // Subscribe to topics
        for topic in self.subscribed_topics {
            runtime.subscribe(&topic, actor_ref.clone()).await?;
        }

        Ok(ActorAgentHandle { agent, actor_ref })
    }

    pub fn subscribe(mut self, topic: Topic<Task>) -> Self {
        self.subscribed_topics.push(topic);
        self
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<T: AgentDeriveT + AgentExecutor + AgentHooks> BaseAgent<T, ActorAgent> {
    pub fn tx(&self) -> Result<Sender<Event>, RunnableAgentError> {
        self.tx.clone().ok_or(RunnableAgentError::EmptyTx)
    }

    /// Convert executor output to a protocol event payload and finish the run
    /// lifecycle. Uses `Value: From<AgentExecutor::Output>` so `TaskComplete`
    /// matches the non-streaming path regardless of `AgentDeriveT::Output`.
    async fn finish_executor_run(
        &self,
        task: &Task,
        context: &Context,
        tx_event: &Option<Sender<Event>>,
        submission_id: SubmissionId,
        executor_out: <T as AgentExecutor>::Output,
    ) -> Result<<T as AgentDeriveT>::Output, RunnableAgentError>
    where
        Value: From<<T as AgentExecutor>::Output>,
        <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output>,
    {
        let value: Value = executor_out.clone().into();
        #[cfg(not(target_arch = "wasm32"))]
        if let Err(e) = EventHelper::send_task_completed_value(
            tx_event,
            submission_id,
            self.id,
            self.name().to_string(),
            &value,
        )
        .await
        {
            let err = RunnableAgentError::ExecutorError(e.to_string());
            EventHelper::send_task_error(tx_event, submission_id, self.id, err.to_string()).await;
            return Err(err);
        }

        let agent_out: <T as AgentDeriveT>::Output = executor_out.into();
        self.inner.on_run_complete(task, &agent_out, context).await;
        Ok(agent_out)
    }

    pub async fn run(
        self: Arc<Self>,
        task: Task,
    ) -> Result<<T as AgentDeriveT>::Output, RunnableAgentError>
    where
        Value: From<<T as AgentExecutor>::Output>,
        <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output>,
        <T as AgentExecutor>::Error: Into<RunnableAgentError>,
    {
        let submission_id = task.submission_id;
        let tx = self.tx().map_err(|_| RunnableAgentError::EmptyTx)?;
        let tx_event = Some(tx.clone());

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
                self.finish_executor_run(&task, &context, &tx_event, submission_id, output)
                    .await
            }
            Err(e) => {
                #[cfg(not(target_arch = "wasm32"))]
                EventHelper::send_task_error(&tx_event, submission_id, self.id, e.to_string())
                    .await;
                Err(e.into())
            }
        }
    }

    /// Return a live executor output stream without the full task lifecycle.
    ///
    /// **Event channel:** Does **not** emit terminal protocol events (`TaskComplete`,
    /// `TaskError`) on the agent event channel. In-stream failures appear only as `Err`
    /// items on the returned stream. Lifecycle hooks (`on_run_start`, `on_run_complete`) are
    /// also skipped.
    ///
    /// Use [`Self::run_stream_to_completion`] when dispatching through a runtime, waiting on
    /// `TaskComplete` / `TaskError`, or matching the pub/sub actor path (which calls
    /// `run_stream_to_completion` internally).
    ///
    /// Mid-run events (`StreamChunk`, tool-call events, etc.) may still be emitted by the
    /// executor while the returned stream is polled.
    pub async fn run_stream(
        self: Arc<Self>,
        task: Task,
    ) -> Result<
        crate::utils::BoxRuntimeStream<Result<<T as AgentDeriveT>::Output, RunnableAgentError>>,
        RunnableAgentError,
    >
    where
        <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output>,
        <T as AgentExecutor>::Error: Into<RunnableAgentError>,
    {
        let context = self.create_context();
        self.run_stream_with_context(task, context).await
    }

    async fn run_executor_stream_with_context(
        self: Arc<Self>,
        task: Task,
        context: Arc<Context>,
    ) -> Result<
        crate::utils::BoxRuntimeStream<Result<<T as AgentExecutor>::Output, RunnableAgentError>>,
        RunnableAgentError,
    >
    where
        <T as AgentExecutor>::Error: Into<RunnableAgentError>,
    {
        match self.inner().execute_stream(&task, context).await {
            Ok(stream) => {
                use futures::StreamExt;
                let transformed_stream =
                    stream.map(move |result| result.map_err(|error| error.into()));
                Ok(Box::pin(transformed_stream))
            }
            Err(error) => Err(error.into()),
        }
    }

    async fn run_stream_with_context(
        self: Arc<Self>,
        task: Task,
        context: Arc<Context>,
    ) -> Result<
        crate::utils::BoxRuntimeStream<Result<<T as AgentDeriveT>::Output, RunnableAgentError>>,
        RunnableAgentError,
    >
    where
        <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output>,
        <T as AgentExecutor>::Error: Into<RunnableAgentError>,
    {
        let stream = self.run_executor_stream_with_context(task, context).await?;
        use futures::StreamExt;
        let transformed_stream = stream.map(|result| result.map(|output| output.into()));
        Ok(Box::pin(transformed_stream))
    }

    /// Execute a streaming task to completion inside the actor, draining the
    /// stream and running lifecycle hooks before returning.
    ///
    /// This is the **event-aware** streaming entry point: emits `TaskError` on hook abort,
    /// stream setup failure, in-stream item errors, and empty streams; emits `TaskComplete`
    /// on success. Pub/sub [`AgentActor`](AgentActor) dispatch uses this method when
    /// `stream()` is enabled.
    ///
    /// When the executor stream yields multiple successful outputs, only the
    /// **last** item is used for `TaskComplete` and the returned agent output.
    /// Intermediate items are not emitted as terminal events.
    ///
    /// For incremental output without terminal events, see [`Self::run_stream`].
    pub async fn run_stream_to_completion(
        self: Arc<Self>,
        task: Task,
    ) -> Result<<T as AgentDeriveT>::Output, RunnableAgentError>
    where
        Value: From<<T as AgentExecutor>::Output>,
        <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output>,
        <T as AgentExecutor>::Error: Into<RunnableAgentError>,
    {
        let submission_id = task.submission_id;
        let tx = self.tx().map_err(|_| RunnableAgentError::EmptyTx)?;
        let tx_event = Some(tx.clone());
        let context = self.create_context();

        let hook_outcome = self.inner.on_run_start(&task, &context).await;
        match hook_outcome {
            HookOutcome::Abort => {
                return Err(
                    EventHelper::abort_run_from_hook(&tx_event, submission_id, self.id).await,
                );
            }
            HookOutcome::Continue => {}
        }

        let mut stream = match self
            .clone()
            .run_executor_stream_with_context(task.clone(), context.clone())
            .await
        {
            Ok(stream) => stream,
            Err(e) => {
                #[cfg(not(target_arch = "wasm32"))]
                EventHelper::send_task_error(&tx_event, submission_id, self.id, e.to_string())
                    .await;
                return Err(e);
            }
        };
        use futures::StreamExt;

        let mut last_executor_output = None;
        while let Some(result) = stream.next().await {
            match EventHelper::map_executor_stream_item(&tx_event, submission_id, self.id, result)
                .await
            {
                Ok(output) => last_executor_output = Some(output),
                Err(e) => return Err(e),
            }
        }

        let executor_out = match last_executor_output {
            Some(output) => output,
            None => {
                let err = RunnableAgentError::ExecutorError(
                    "Stream completed without output".to_string(),
                );
                #[cfg(not(target_arch = "wasm32"))]
                EventHelper::send_task_error(&tx_event, submission_id, self.id, err.to_string())
                    .await;
                return Err(err);
            }
        };

        self.finish_executor_run(&task, &context, &tx_event, submission_id, executor_out)
            .await
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
impl<T: AgentDeriveT + AgentExecutor + AgentHooks> Actor for AgentActor<T>
where
    T: Send + Sync + 'static,
    serde_json::Value: From<<T as AgentExecutor>::Output>,
    <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output>,
    <T as AgentExecutor>::Error: Into<RunnableAgentError>,
{
    type Msg = Task;
    type State = AgentState;
    type Arguments = ();

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        _args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        Ok(AgentState::new())
    }

    async fn post_stop(
        &self,
        _myself: ActorRef<Self::Msg>,
        _state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        //Run Hook
        self.0.inner().on_agent_shutdown().await;
        Ok(())
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        _state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        let agent = self.0.clone();

        // Run agent
        let result = if agent.stream() {
            agent.run_stream_to_completion(message).await
        } else {
            agent.run(message).await
        };

        // Task-level failures are surfaced on the event channel as `TaskError` (or
        // `TaskComplete` on success) by the run helpers above. Do not propagate them
        // to ractor — returning `Err` here terminates the actor and breaks long-running
        // pub/sub agents after a single bad task.
        let _ = result;
        Ok(())
    }
}

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod tests {
    use super::*;
    use crate::actor::{LocalTransport, Topic, Transport};
    use crate::agent::hooks::HookOutcome;
    use crate::agent::output::AgentOutputT;
    use crate::agent::{Context, ExecutorConfig};
    use crate::runtime::{Runtime, RuntimeError};
    use crate::tests::{
        DivergentStreamingAgent, MockAgentImpl, MockLLMProvider, MultiItemStreamAgent,
        TestAgentOutput, TestError,
    };
    use crate::utils::BoxEventStream;
    use async_trait::async_trait;
    use futures::{StreamExt, stream};
    use std::any::{Any, TypeId};
    use std::sync::Arc;
    use tokio::sync::{Mutex, mpsc};

    #[derive(Debug)]
    struct TestRuntime {
        subscribed: Arc<Mutex<Vec<(String, TypeId)>>>,
        tx: mpsc::Sender<Event>,
    }

    impl TestRuntime {
        fn new() -> Self {
            let (tx, _rx) = mpsc::channel(4);
            Self {
                subscribed: Arc::new(Mutex::new(Vec::new())),
                tx,
            }
        }
    }

    #[async_trait]
    impl Runtime for TestRuntime {
        fn id(&self) -> autoagents_protocol::RuntimeID {
            autoagents_protocol::RuntimeID::new_v4()
        }

        async fn subscribe_any(
            &self,
            topic_name: &str,
            topic_type: TypeId,
            _actor: Arc<dyn crate::actor::AnyActor>,
        ) -> Result<(), RuntimeError> {
            let mut subscribed = self.subscribed.lock().await;
            subscribed.push((topic_name.to_string(), topic_type));
            Ok(())
        }

        async fn publish_any(
            &self,
            _topic_name: &str,
            _topic_type: TypeId,
            _message: Arc<dyn Any + Send + Sync>,
        ) -> Result<(), RuntimeError> {
            Ok(())
        }

        fn tx(&self) -> mpsc::Sender<Event> {
            self.tx.clone()
        }

        async fn transport(&self) -> Arc<dyn Transport> {
            Arc::new(LocalTransport)
        }

        async fn take_event_receiver(&self) -> Option<BoxEventStream<Event>> {
            None
        }

        async fn subscribe_events(&self) -> BoxEventStream<Event> {
            Box::pin(stream::empty())
        }

        async fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Ok(())
        }

        async fn stop(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_actor_builder_requires_llm() {
        let mock = MockAgentImpl::new("agent", "desc");
        let runtime = Arc::new(TestRuntime::new());
        let err = AgentBuilder::<_, ActorAgent>::new(mock)
            .runtime(runtime)
            .build()
            .await
            .unwrap_err();
        assert!(matches!(err, Error::AgentBuildError(_)));
    }

    #[tokio::test]
    async fn test_actor_builder_requires_runtime() {
        let mock = MockAgentImpl::new("agent", "desc");
        let llm = Arc::new(MockLLMProvider);
        let err = AgentBuilder::<_, ActorAgent>::new(mock)
            .llm(llm)
            .build()
            .await
            .unwrap_err();
        assert!(matches!(err, Error::AgentBuildError(_)));
    }

    #[tokio::test]
    async fn test_actor_builder_subscribes_topics() {
        let mock = MockAgentImpl::new("agent", "desc");
        let llm = Arc::new(MockLLMProvider);
        let runtime = Arc::new(TestRuntime::new());
        let topic = Topic::<Task>::new("jobs");

        let _handle = AgentBuilder::<_, ActorAgent>::new(mock)
            .llm(llm)
            .runtime(runtime.clone())
            .subscribe(topic)
            .build()
            .await
            .expect("build should succeed");

        let subscribed = runtime.subscribed.lock().await;
        assert_eq!(subscribed.len(), 1);
        assert_eq!(subscribed[0].0, "jobs");
    }

    #[tokio::test]
    async fn test_actor_agent_tx_missing_returns_error() {
        let mock = MockAgentImpl::new("agent", "desc");
        let llm = Arc::new(MockLLMProvider);
        let (tx, _rx) = mpsc::channel(2);
        let mut agent = BaseAgent::<_, ActorAgent>::new(mock, llm, None, tx, false)
            .await
            .unwrap();
        agent.tx = None;
        let err = agent.tx().unwrap_err();
        assert!(matches!(err, RunnableAgentError::EmptyTx));
    }

    async fn streaming_actor_agent(stream: bool) -> Arc<BaseAgent<MockAgentImpl, ActorAgent>> {
        let mock = MockAgentImpl::new("stream_agent", "streaming test agent");
        let llm = Arc::new(MockLLMProvider);
        let (tx, _rx) = mpsc::channel(8);
        Arc::new(
            BaseAgent::<_, ActorAgent>::new(mock, llm, None, tx, stream)
                .await
                .expect("agent should build"),
        )
    }

    #[tokio::test]
    async fn test_actor_run_stream_returns_executor_output() {
        let agent = streaming_actor_agent(true).await;
        let task = Task::new("stream me");
        let stream = agent.run_stream(task).await.expect("stream should start");
        let outputs: Vec<_> = stream.collect().await;
        assert_eq!(outputs.len(), 1);
        let output = outputs[0].as_ref().expect("expected stream output");
        assert!(output.result.contains("stream me"));
    }

    #[tokio::test]
    async fn test_actor_run_stream_to_completion_returns_output() {
        let agent = streaming_actor_agent(true).await;
        let task = Task::new("complete me");
        let output = agent
            .run_stream_to_completion(task)
            .await
            .expect("stream should complete");
        assert!(output.result.contains("complete me"));
    }

    #[derive(Debug, Clone)]
    struct AbortStreamingAgent;

    #[async_trait]
    impl AgentDeriveT for AbortStreamingAgent {
        type Output = TestAgentOutput;

        fn description(&self) -> &'static str {
            "abort streaming agent"
        }

        fn output_schema(&self) -> Option<serde_json::Value> {
            Some(TestAgentOutput::structured_output_format())
        }

        fn name(&self) -> &'static str {
            "abort_streaming_agent"
        }

        fn tools(&self) -> Vec<Box<dyn crate::tool::ToolT>> {
            vec![]
        }
    }

    #[async_trait]
    impl AgentExecutor for AbortStreamingAgent {
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
    }

    #[async_trait]
    impl AgentHooks for AbortStreamingAgent {
        async fn on_run_start(&self, _task: &Task, _ctx: &Context) -> HookOutcome {
            HookOutcome::Abort
        }
    }

    #[tokio::test]
    async fn test_actor_run_aborts_on_hook_emits_task_error() {
        let llm = Arc::new(MockLLMProvider);
        let (tx, mut rx) = mpsc::channel(2);
        let agent = Arc::new(
            BaseAgent::<_, ActorAgent>::new(AbortStreamingAgent, llm, None, tx, false)
                .await
                .expect("agent should build"),
        );

        let err = agent
            .run(Task::new("abort"))
            .await
            .expect_err("expected hook abort");
        assert!(matches!(err, RunnableAgentError::Abort));

        let event = tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
            .await
            .expect("timed out waiting for TaskError event")
            .expect("channel closed without event");
        match event {
            Event::TaskError { error, .. } => {
                assert!(error.contains("Abort"));
            }
            other => panic!("expected TaskError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_actor_run_stream_to_completion_aborts_on_hook() {
        let llm = Arc::new(MockLLMProvider);
        let (tx, mut rx) = mpsc::channel(2);
        let agent = Arc::new(
            BaseAgent::<_, ActorAgent>::new(AbortStreamingAgent, llm, None, tx, true)
                .await
                .expect("agent should build"),
        );

        let err = agent
            .run_stream_to_completion(Task::new("abort"))
            .await
            .expect_err("expected hook abort");
        assert!(matches!(err, RunnableAgentError::Abort));

        let event = tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
            .await
            .expect("timed out waiting for TaskError event")
            .expect("channel closed without event");
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

        fn output_schema(&self) -> Option<serde_json::Value> {
            Some(TestAgentOutput::structured_output_format())
        }

        fn name(&self) -> &'static str {
            "failing_stream_setup"
        }

        fn tools(&self) -> Vec<Box<dyn crate::tool::ToolT>> {
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
    async fn test_actor_run_stream_to_completion_execute_stream_setup_error() {
        let llm = Arc::new(MockLLMProvider);
        let (tx, mut rx) = mpsc::channel(2);
        let agent = Arc::new(
            BaseAgent::<_, ActorAgent>::new(FailingStreamSetupAgent, llm, None, tx, true)
                .await
                .expect("agent should build"),
        );

        let err = agent
            .run_stream_to_completion(Task::new("fail setup"))
            .await
            .expect_err("expected stream setup failure");
        assert!(matches!(err, RunnableAgentError::ExecutorError(_)));

        let event = tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
            .await
            .expect("timed out waiting for TaskError event")
            .expect("channel closed without event");
        match event {
            Event::TaskError { error, .. } => {
                assert!(error.contains("stream setup failed"));
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

        fn output_schema(&self) -> Option<serde_json::Value> {
            Some(TestAgentOutput::structured_output_format())
        }

        fn name(&self) -> &'static str {
            "empty_stream_agent"
        }

        fn tools(&self) -> Vec<Box<dyn crate::tool::ToolT>> {
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
            Ok(Box::pin(stream::empty()))
        }
    }

    impl AgentHooks for EmptyStreamAgent {}

    #[tokio::test]
    async fn test_actor_run_stream_to_completion_empty_stream_error() {
        let llm = Arc::new(MockLLMProvider);
        let (tx, mut rx) = mpsc::channel(2);
        let agent = Arc::new(
            BaseAgent::<_, ActorAgent>::new(EmptyStreamAgent, llm, None, tx, true)
                .await
                .expect("agent should build"),
        );

        let err = agent
            .run_stream_to_completion(Task::new("empty"))
            .await
            .expect_err("expected empty stream failure");
        assert!(matches!(err, RunnableAgentError::ExecutorError(_)));
        assert!(err.to_string().contains("without output"));

        let event = tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
            .await
            .expect("timed out waiting for TaskError event")
            .expect("channel closed without event");
        match event {
            Event::TaskError { error, .. } => {
                assert!(error.contains("without output"));
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

        fn output_schema(&self) -> Option<serde_json::Value> {
            Some(TestAgentOutput::structured_output_format())
        }

        fn name(&self) -> &'static str {
            "stream_item_error_agent"
        }

        fn tools(&self) -> Vec<Box<dyn crate::tool::ToolT>> {
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
            Ok(Box::pin(stream::iter([Err(TestError::ExecutionFailed(
                "stream item failed".to_string(),
            ))])))
        }
    }

    impl AgentHooks for StreamItemErrorAgent {}

    #[tokio::test]
    async fn test_actor_run_stream_does_not_emit_task_error_on_item_failure() {
        let llm = Arc::new(MockLLMProvider);
        let (tx, mut rx) = mpsc::channel(2);
        let agent = Arc::new(
            BaseAgent::<_, ActorAgent>::new(StreamItemErrorAgent, llm, None, tx, true)
                .await
                .expect("agent should build"),
        );

        let mut stream = agent
            .run_stream(Task::new("stream error"))
            .await
            .expect("stream should start");
        let err = stream
            .next()
            .await
            .expect("stream should yield one item")
            .expect_err("expected stream item failure");
        assert!(err.to_string().contains("stream item failed"));

        let event = tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv()).await;
        match event {
            Ok(Some(Event::TaskError { .. })) => {
                panic!("run_stream must not emit TaskError; use run_stream_to_completion")
            }
            Ok(Some(_)) | Ok(None) => {}
            Err(_) => {}
        }
    }

    #[tokio::test]
    async fn test_actor_run_stream_to_completion_stream_item_error() {
        let llm = Arc::new(MockLLMProvider);
        let (tx, mut rx) = mpsc::channel(2);
        let agent = Arc::new(
            BaseAgent::<_, ActorAgent>::new(StreamItemErrorAgent, llm, None, tx, true)
                .await
                .expect("agent should build"),
        );

        let err = agent
            .run_stream_to_completion(Task::new("stream error"))
            .await
            .expect_err("expected stream item failure");
        assert!(matches!(err, RunnableAgentError::ExecutorError(_)));
        assert!(err.to_string().contains("stream item failed"));

        let event = tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
            .await
            .expect("timed out waiting for TaskError event")
            .expect("channel closed without event");
        match event {
            Event::TaskError { error, .. } => {
                assert!(error.contains("stream item failed"));
            }
            other => panic!("expected TaskError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_actor_run_stream_to_completion_missing_tx_error() {
        let llm = Arc::new(MockLLMProvider);
        let (tx, _rx) = mpsc::channel(2);
        let mut agent = BaseAgent::<_, ActorAgent>::new(
            MockAgentImpl::new("agent", "desc"),
            llm,
            None,
            tx,
            true,
        )
        .await
        .expect("agent should build");
        agent.tx = None;
        let agent = Arc::new(agent);

        let err = agent
            .run_stream_to_completion(Task::new("missing tx"))
            .await
            .expect_err("expected missing tx failure");
        assert!(matches!(err, RunnableAgentError::EmptyTx));
    }

    #[tokio::test]
    async fn test_actor_run_stream_to_completion_uses_last_stream_item() {
        let llm = Arc::new(MockLLMProvider);
        let (tx, mut rx) = mpsc::channel(2);
        let agent = Arc::new(
            BaseAgent::<_, ActorAgent>::new(MultiItemStreamAgent, llm, None, tx, true)
                .await
                .expect("agent should build"),
        );

        let output = agent
            .run_stream_to_completion(Task::new("chunked"))
            .await
            .expect("stream should complete");
        assert_eq!(output.result, "chunked-3");

        let event = tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
            .await
            .expect("timed out waiting for TaskComplete event")
            .expect("channel closed without event");
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

    #[tokio::test]
    async fn test_actor_run_stream_to_completion_task_complete_uses_executor_value_from() {
        let llm = Arc::new(MockLLMProvider);
        let (tx, mut rx) = mpsc::channel(2);
        let agent = Arc::new(
            BaseAgent::<_, ActorAgent>::new(DivergentStreamingAgent, llm, None, tx, true)
                .await
                .expect("agent should build"),
        );

        let output = agent
            .run_stream_to_completion(Task::new("stream payload"))
            .await
            .expect("stream should complete");
        assert_eq!(output.result, "stream payload");

        let event = tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
            .await
            .expect("timed out waiting for TaskComplete event")
            .expect("channel closed without event");
        match event {
            Event::TaskComplete { result, .. } => {
                let parsed: Value =
                    serde_json::from_str(&result).expect("TaskComplete result should be JSON");
                assert_eq!(parsed["executor_only"], 42);
                assert_eq!(parsed["response"], "stream payload");
            }
            other => panic!("expected TaskComplete, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_actor_run_and_stream_to_completion_emit_matching_task_complete() {
        let llm = Arc::new(MockLLMProvider);
        let (tx_run, mut rx_run) = mpsc::channel(2);
        let run_agent = Arc::new(
            BaseAgent::<_, ActorAgent>::new(
                DivergentStreamingAgent,
                llm.clone(),
                None,
                tx_run,
                false,
            )
            .await
            .expect("run agent should build"),
        );
        let (tx_stream, mut rx_stream) = mpsc::channel(2);
        let stream_agent = Arc::new(
            BaseAgent::<_, ActorAgent>::new(DivergentStreamingAgent, llm, None, tx_stream, true)
                .await
                .expect("stream agent should build"),
        );

        let task = Task::new("parity payload");
        run_agent
            .clone()
            .run(task.clone())
            .await
            .expect("run should succeed");
        stream_agent
            .run_stream_to_completion(task)
            .await
            .expect("stream should complete");

        let run_event = tokio::time::timeout(std::time::Duration::from_secs(1), rx_run.recv())
            .await
            .expect("timed out waiting for run TaskComplete")
            .expect("run channel closed without event");
        let stream_event =
            tokio::time::timeout(std::time::Duration::from_secs(1), rx_stream.recv())
                .await
                .expect("timed out waiting for stream TaskComplete")
                .expect("stream channel closed without event");

        match (run_event, stream_event) {
            (
                Event::TaskComplete {
                    result: run_result, ..
                },
                Event::TaskComplete {
                    result: stream_result,
                    ..
                },
            ) => {
                assert_eq!(
                    run_result, stream_result,
                    "run() and run_stream_to_completion() must serialize TaskComplete identically"
                );
            }
            (run_event, stream_event) => {
                panic!("expected TaskComplete events, got {run_event:?} and {stream_event:?}")
            }
        }
    }
}
