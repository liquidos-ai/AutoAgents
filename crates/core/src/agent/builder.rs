#[cfg(not(target_arch = "wasm32"))]
use crate::actor::Topic;
use crate::agent::base::AgentType;
use crate::agent::error::AgentBuildError;
use crate::agent::memory::MemoryProvider;
use crate::agent::state::AgentState;
use crate::agent::task::Task;
#[cfg(not(target_arch = "wasm32"))]
use crate::agent::AgentHandle;
use crate::agent::{ActorAgent, AgentDeriveT, AgentExecutor, BaseAgent, DirectAgent};
use crate::error::Error;
#[cfg(not(target_arch = "wasm32"))]
use crate::runtime::{Runtime, TypedRuntime};
use async_trait::async_trait;
use autoagents_llm::LLMProvider;
#[cfg(target_arch = "wasm32")]
use futures::SinkExt;
#[cfg(not(target_arch = "wasm32"))]
use ractor::Actor;
#[cfg(not(target_arch = "wasm32"))]
use ractor::{ActorProcessingErr, ActorRef};
use std::marker::PhantomData;
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub struct AgentActor<T: AgentDeriveT + AgentExecutor, A: AgentType>(pub Arc<BaseAgent<T, A>>);

#[cfg(not(target_arch = "wasm32"))]
impl<T: AgentDeriveT + AgentExecutor, A: AgentType> AgentActor<T, A> {}

/// Builder for creating BaseAgent instances from AgentDeriveT implementations
pub struct AgentBuilder<T: AgentDeriveT + AgentExecutor, A: AgentType> {
    inner: T,
    stream: bool,
    llm: Option<Arc<dyn LLMProvider>>,
    memory: Option<Box<dyn MemoryProvider>>,
    #[cfg(not(target_arch = "wasm32"))]
    runtime: Option<Arc<dyn Runtime>>,
    #[cfg(not(target_arch = "wasm32"))]
    subscribed_topics: Vec<Topic<Task>>,
    marker: PhantomData<A>,
}

impl<T: AgentDeriveT + AgentExecutor, A: AgentType> AgentBuilder<T, A> {
    /// Create a new builder with an AgentDeriveT implementation
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            llm: None,
            memory: None,
            #[cfg(not(target_arch = "wasm32"))]
            runtime: None,
            stream: false,
            #[cfg(not(target_arch = "wasm32"))]
            subscribed_topics: vec![],
            marker: PhantomData,
        }
    }

    /// Set the LLM provider
    pub fn llm(mut self, llm: Arc<dyn LLMProvider>) -> Self {
        self.llm = Some(llm);
        self
    }

    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Set the memory provider
    pub fn memory(mut self, memory: Box<dyn MemoryProvider>) -> Self {
        self.memory = Some(memory);
        self
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn runtime(mut self, runtime: Arc<dyn Runtime>) -> Self {
        self.runtime = Some(runtime);
        self
    }
}

impl<T: AgentDeriveT + AgentExecutor> AgentBuilder<T, DirectAgent> {
    /// Build the BaseAgent and return a wrapper that includes the actor reference
    #[allow(clippy::result_large_err)]
    pub fn build(self) -> Result<BaseAgent<T, DirectAgent>, Error> {
        let llm = self.llm.ok_or(AgentBuildError::BuildFailure(
            "LLM provider is required".to_string(),
        ))?;
        let agent: BaseAgent<T, DirectAgent> =
            BaseAgent::<T, DirectAgent>::new(self.inner, llm, self.memory, self.stream);
        Ok(agent)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<T: AgentDeriveT + AgentExecutor> AgentBuilder<T, ActorAgent>
where
    T: Send + Sync + 'static,
    serde_json::Value: From<<T as AgentExecutor>::Output>,
{
    /// Build the BaseAgent and return a wrapper that includes the actor reference
    pub async fn build(self) -> Result<AgentHandle<T, ActorAgent>, Error> {
        let llm = self.llm.ok_or(AgentBuildError::BuildFailure(
            "LLM provider is required".to_string(),
        ))?;
        let runtime = self.runtime.ok_or(AgentBuildError::BuildFailure(
            "Runtime should be defined".into(),
        ))?;
        let tx = runtime.tx();

        let agent: Arc<BaseAgent<T, ActorAgent>> = Arc::new(BaseAgent::<T, ActorAgent>::new(
            self.inner,
            llm,
            self.memory,
            tx,
            self.stream,
        ));

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

        Ok(AgentHandle { agent, actor_ref })
    }

    pub fn subscribe(mut self, topic: Topic<Task>) -> Self {
        self.subscribed_topics.push(topic);
        self
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
impl<T: AgentDeriveT + AgentExecutor> Actor for AgentActor<T, ActorAgent>
where
    T: Send + Sync + 'static,
    serde_json::Value: From<<T as AgentExecutor>::Output>,
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
mod tests {
    use super::*;
    use crate::actor::Topic;
    use crate::agent::task::Task;
    use autoagents_test_utils::agent::MockAgentImpl;

    #[test]
    fn test_agent_builder_with_subscribe_topic() {
        let mock_agent = MockAgentImpl::new("topic_agent", "test topic agent");
        let topic = Topic::<Task>::new("test_topic");

        let builder = AgentBuilder::new(mock_agent).subscribe(topic);

        assert_eq!(builder.subscribed_topics.len(), 1);
        assert_eq!(builder.subscribed_topics[0].name(), "test_topic");
    }

    #[test]
    fn test_agent_builder_multiple_topics() {
        let mock_agent = MockAgentImpl::new("multi_topic_agent", "test multiple topics");
        let topic1 = Topic::<Task>::new("topic1");
        let topic2 = Topic::<Task>::new("topic2");

        let builder = AgentBuilder::new(mock_agent)
            .subscribe(topic1)
            .subscribe(topic2);

        assert_eq!(builder.subscribed_topics.len(), 2);
        assert_eq!(builder.subscribed_topics[0].name(), "topic1");
        assert_eq!(builder.subscribed_topics[1].name(), "topic2");
    }
}
