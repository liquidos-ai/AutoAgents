#[cfg(not(target_arch = "wasm32"))]
use crate::actor::Topic;
use crate::agent::base::AgentType;
use crate::agent::hooks::AgentHooks;
use crate::agent::memory::MemoryProvider;
use crate::agent::task::Task;
use crate::agent::{AgentDeriveT, AgentExecutor};
#[cfg(not(target_arch = "wasm32"))]
use crate::runtime::Runtime;
use autoagents_llm::LLMProvider;
use std::marker::PhantomData;
use std::sync::Arc;

/// Builder for creating BaseAgent instances from AgentDeriveT implementations
pub struct AgentBuilder<T: AgentDeriveT + AgentExecutor + AgentHooks, A: AgentType> {
    pub(crate) inner: T,
    pub(crate) stream: bool,
    pub(crate) llm: Option<Arc<dyn LLMProvider>>,
    pub(crate) memory: Option<Box<dyn MemoryProvider>>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) runtime: Option<Arc<dyn Runtime>>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) subscribed_topics: Vec<Topic<Task>>,
    marker: PhantomData<A>,
}

impl<T: AgentDeriveT + AgentExecutor + AgentHooks, A: AgentType> AgentBuilder<T, A> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::Topic;
    use crate::agent::task::Task;
    use crate::tests::MockAgentImpl;

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
