use crate::actor::Topic;
use crate::agent::memory::MemoryProvider;
use crate::agent::task::Task;
use crate::agent::{
    AgentActor, AgentBuildError, AgentDeriveT, AgentExecutor, AgentHandle, BaseAgent, IntoRunnable,
};
use crate::error::Error;
use crate::runtime::{Runtime, TypedRuntime};
use autoagents_llm::LLMProvider;
use ractor::Actor;
use std::sync::Arc;

/// Builder for creating BaseAgent instances from AgentDeriveT implementations
pub struct AgentBuilder<T: AgentDeriveT + AgentExecutor> {
    inner: T,
    stream: bool,
    llm: Option<Arc<dyn LLMProvider>>,
    memory: Option<Box<dyn MemoryProvider>>,
    runtime: Option<Arc<dyn Runtime>>,
    subscribed_topics: Vec<Topic<Task>>,
}

impl<T: AgentDeriveT + AgentExecutor> AgentBuilder<T> {
    /// Create a new builder with an AgentDeriveT implementation
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            llm: None,
            memory: None,
            runtime: None,
            stream: false,
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

    pub fn subscribe_topic(mut self, topic: Topic<Task>) -> Self {
        self.subscribed_topics.push(topic);
        self
    }

    /// Build the BaseAgent and return a wrapper that includes the actor reference
    pub async fn build(self) -> Result<AgentHandle<T>, Error> {
        let llm = self.llm.ok_or(AgentBuildError::BuildFailure(
            "LLM provider is required".to_string(),
        ))?;
        let runnable: Arc<BaseAgent<T>> =
            BaseAgent::new(self.inner, llm, self.memory, self.stream).into_runnable();

        let runtime = self.runtime.ok_or(AgentBuildError::BuildFailure(
            "Runtime should be defined".into(),
        ))?;

        // Create agent actor
        let agent_actor = AgentActor(runnable.clone());
        let tx = runtime.tx().await;
        let actor_ref = Actor::spawn(Some(runnable.inner.name().into()), agent_actor, tx.clone())
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
