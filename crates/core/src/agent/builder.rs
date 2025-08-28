#[cfg(not(target_arch = "wasm32"))]
use crate::actor::Topic;
use crate::agent::error::AgentBuildError;
use crate::agent::memory::MemoryProvider;
#[cfg(target_arch = "wasm32")]
use crate::agent::runnable::RunnableAgentImpl;
use crate::agent::task::Task;
#[cfg(not(target_arch = "wasm32"))]
use crate::agent::{AgentActor, AgentHandle};
use crate::agent::{AgentDeriveT, AgentExecutor, BaseAgent, IntoRunnable};
use crate::error::Error;
use crate::protocol::Event;
#[cfg(not(target_arch = "wasm32"))]
use crate::runtime::{Runtime, TypedRuntime};
use autoagents_llm::LLMProvider;
#[cfg(target_arch = "wasm32")]
use futures::SinkExt;
#[cfg(not(target_arch = "wasm32"))]
use ractor::Actor;
use std::sync::Arc;

/// Builder for creating BaseAgent instances from AgentDeriveT implementations
pub struct AgentBuilder<T: AgentDeriveT + AgentExecutor> {
    inner: T,
    stream: bool,
    llm: Option<Arc<dyn LLMProvider>>,
    memory: Option<Box<dyn MemoryProvider>>,
    #[cfg(not(target_arch = "wasm32"))]
    runtime: Option<Arc<dyn Runtime>>,
    #[cfg(not(target_arch = "wasm32"))]
    subscribed_topics: Vec<Topic<Task>>,
}

impl<T: AgentDeriveT + AgentExecutor> AgentBuilder<T> {
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
        }
    }

    /// Set the LLM provider
    pub fn with_llm(mut self, llm: Arc<dyn LLMProvider>) -> Self {
        self.llm = Some(llm);
        self
    }

    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Set the memory provider
    pub fn with_memory(mut self, memory: Box<dyn MemoryProvider>) -> Self {
        self.memory = Some(memory);
        self
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn subscribe_topic(mut self, topic: Topic<Task>) -> Self {
        self.subscribed_topics.push(topic);
        self
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Build the BaseAgent and return a wrapper that includes the actor reference
    pub async fn build(self) -> Result<AgentHandle<T>, Error> {
        let llm = self.llm.ok_or(AgentBuildError::BuildFailure(
            "LLM provider is required".to_string(),
        ))?;
        let runtime = self.runtime.ok_or(AgentBuildError::BuildFailure(
            "Runtime should be defined".into(),
        ))?;
        let tx = runtime.tx().await;

        let runnable: Arc<BaseAgent<T>> =
            BaseAgent::new(self.inner, llm, self.memory, tx, self.stream).into_runnable();

        // Create agent actor
        let agent_actor = AgentActor(runnable.clone());
        let actor_ref = Actor::spawn(Some(runnable.inner.name().into()), agent_actor, ())
            .await
            .map_err(AgentBuildError::SpawnError)?
            .0;

        // Subscribe to topics
        for topic in self.subscribed_topics {
            runtime.subscribe(&topic, actor_ref.clone()).await?;
        }

        Ok(AgentHandle {
            agent: runnable,
            actor_ref,
        })
    }

    #[cfg(target_arch = "wasm32")]
    pub fn build_runnable(
        self,
    ) -> Result<
        (
            Arc<RunnableAgentImpl<T>>,
            futures::channel::mpsc::Receiver<Event>,
        ),
        Error,
    >
    where
        Error: From<AgentBuildError>,
    {
        // Ensure LLM provider exists
        let llm = self
            .llm
            .ok_or_else(|| AgentBuildError::BuildFailure("LLM provider is required".to_string()))?;

        // Create channel for events
        let (tx, rx) = futures::channel::mpsc::channel::<Event>(100);

        // Build BaseAgent and convert to runnable
        let runnable =
            BaseAgent::new(self.inner, llm, self.memory, tx, self.stream).into_runnable();

        Ok((runnable, rx))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn runtime(mut self, runtime: Arc<dyn Runtime>) -> Self {
        self.runtime = Some(runtime);
        self
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

        let builder = AgentBuilder::new(mock_agent).subscribe_topic(topic);

        assert_eq!(builder.subscribed_topics.len(), 1);
        assert_eq!(builder.subscribed_topics[0].name(), "test_topic");
    }

    #[test]
    fn test_agent_builder_multiple_topics() {
        let mock_agent = MockAgentImpl::new("multi_topic_agent", "test multiple topics");
        let topic1 = Topic::<Task>::new("topic1");
        let topic2 = Topic::<Task>::new("topic2");

        let builder = AgentBuilder::new(mock_agent)
            .subscribe_topic(topic1)
            .subscribe_topic(topic2);

        assert_eq!(builder.subscribed_topics.len(), 2);
        assert_eq!(builder.subscribed_topics[0].name(), "topic1");
        assert_eq!(builder.subscribed_topics[1].name(), "topic2");
    }
}
