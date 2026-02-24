#[cfg(not(target_arch = "wasm32"))]
use crate::actor::Topic;
use crate::agent::base::AgentType;
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
use autoagents_protocol::Event;
#[cfg(target_arch = "wasm32")]
use futures::SinkExt;
use futures::Stream;
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

    pub async fn run(
        self: Arc<Self>,
        task: Task,
    ) -> Result<<T as AgentDeriveT>::Output, RunnableAgentError>
    where
        Value: From<<T as AgentExecutor>::Output>,
        <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output>,
    {
        let submission_id = task.submission_id;
        let tx = self.tx().map_err(|_| RunnableAgentError::EmptyTx)?;
        let tx_event = Some(tx.clone());

        let context = self.create_context();

        //Run Hook
        let hook_outcome = self.inner.on_run_start(&task, &context).await;
        match hook_outcome {
            HookOutcome::Abort => return Err(RunnableAgentError::Abort),
            HookOutcome::Continue => {}
        }

        // Execute the agent's logic using the executor
        match self.inner().execute(&task, context.clone()).await {
            Ok(output) => {
                let value: Value = output.clone().into();
                #[cfg(not(target_arch = "wasm32"))]
                EventHelper::send_task_completed_value(
                    &tx_event,
                    submission_id,
                    self.id,
                    self.name().to_string(),
                    &value,
                )
                .await
                .map_err(|e| RunnableAgentError::ExecutorError(e.to_string()))?;

                //Extract Agent output into the desired type
                let agent_out: <T as AgentDeriveT>::Output = output.into();

                //Run On complete Hook
                self.inner
                    .on_run_complete(&task, &agent_out, &context)
                    .await;

                Ok(agent_out)
            }
            Err(e) => {
                #[cfg(not(target_arch = "wasm32"))]
                EventHelper::send_task_error(&tx_event, submission_id, self.id, e.to_string())
                    .await;
                Err(RunnableAgentError::ExecutorError(e.to_string()))
            }
        }
    }

    pub async fn run_stream(
        self: Arc<Self>,
        task: Task,
    ) -> Result<
        std::pin::Pin<
            Box<dyn Stream<Item = Result<<T as AgentDeriveT>::Output, RunnableAgentError>> + Send>,
        >,
        RunnableAgentError,
    >
    where
        <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output>,
    {
        // let submission_id = task.submission_id;
        let context = self.create_context();

        // Execute the agent's streaming logic using the executor
        match self.inner().execute_stream(&task, context).await {
            Ok(stream) => {
                use futures::StreamExt;
                // Transform the stream to convert agent output to TaskResult
                let transformed_stream = stream.map(move |result| {
                    match result {
                        Ok(output) => Ok(output.into()),
                        Err(e) => {
                            // Handle error
                            let error_msg = e.to_string();
                            Err(RunnableAgentError::ExecutorError(error_msg))
                        }
                    }
                });

                Ok(Box::pin(transformed_stream))
            }
            Err(e) => {
                // Send error event for stream creation failure
                Err(RunnableAgentError::ExecutorError(e.to_string()))
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
impl<T: AgentDeriveT + AgentExecutor + AgentHooks> Actor for AgentActor<T>
where
    T: Send + Sync + 'static,
    serde_json::Value: From<<T as AgentExecutor>::Output>,
    <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output>,
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
        let task = message;

        //Run agent
        if agent.stream() {
            let _ = agent.run_stream(task).await?;
            Ok(())
        } else {
            let _ = agent.run(task).await?;
            Ok(())
        }
    }
}

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod tests {
    use super::*;
    use crate::actor::{LocalTransport, Topic, Transport};
    use crate::runtime::{Runtime, RuntimeError};
    use crate::tests::{MockAgentImpl, MockLLMProvider};
    use crate::utils::BoxEventStream;
    use async_trait::async_trait;
    use futures::stream;
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
}
