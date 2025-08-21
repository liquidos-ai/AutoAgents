use super::{
    error::AgentBuildError, output::AgentOutputT, AgentActor, AgentExecutor, IntoRunnable,
};
use crate::actor::Topic;
use crate::agent::config::AgentConfig;
use crate::agent::memory::MemoryProvider;
use crate::agent::task::Task;
use crate::runtime::TypedRuntime;
use crate::{
    error::Error, protocol::ActorID,
    runtime::Runtime, tool::ToolT,
};
use async_trait::async_trait;
use autoagents_llm::LLMProvider;
use ractor::{Actor, ActorRef};
use serde_json::Value;
use std::{fmt::Debug, sync::Arc};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Core trait that defines agent metadata and behavior
/// This trait is implemented via the #[agent] macro
#[async_trait]
pub trait AgentDeriveT: Send + Sync + 'static + AgentExecutor + Debug {
    /// The output type this agent produces
    type Output: AgentOutputT;

    /// Get the agent's description
    fn description(&self) -> &'static str;

    fn output_schema(&self) -> Option<Value>;

    /// Get the agent's name
    fn name(&self) -> &'static str;

    /// Get the tools available to this agent
    fn tools(&self) -> Vec<Box<dyn ToolT>>;
}

/// Base agent type that wraps an AgentDeriveT implementation with additional runtime components
#[derive(Clone)]
pub struct BaseAgent<T: AgentDeriveT> {
    /// The inner agent implementation (from macro)
    pub inner: Arc<T>,
    /// LLM provider for this agent
    pub llm: Arc<dyn LLMProvider>,
    // Agent ID
    pub id: ActorID,
    /// Optional memory provider
    pub memory: Option<Arc<RwLock<Box<dyn MemoryProvider>>>>,
    //Stream
    pub stream: bool,
}

impl<T: AgentDeriveT> Debug for BaseAgent<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.inner().name())
    }
}

impl<T: AgentDeriveT> BaseAgent<T> {
    /// Create a new BaseAgent wrapping an AgentDeriveT implementation
    pub fn new(
        inner: T,
        llm: Arc<dyn LLMProvider>,
        memory: Option<Box<dyn MemoryProvider>>,
        stream: bool,
    ) -> Self {
        // Convert tools to Arc for efficient sharing
        Self {
            inner: Arc::new(inner),
            id: Uuid::new_v4(),
            llm,
            memory: memory.map(|m| Arc::new(RwLock::new(m))),
            stream,
        }
    }

    pub fn inner(&self) -> Arc<T> {
        self.inner.clone()
    }

    /// Get the agent's name
    pub fn name(&self) -> &'static str {
        self.inner.name()
    }

    /// Get the agent's description
    pub fn description(&self) -> &'static str {
        self.inner.description()
    }

    /// Get the tools as Arc-wrapped references
    pub fn tools(&self) -> Vec<Box<dyn ToolT>> {
        self.inner.tools()
    }

    pub fn agent_config(&self) -> AgentConfig {
        let output_schema = self.inner().output_schema();
        let structured_schema = output_schema.map(|schema| serde_json::from_value(schema).unwrap());
        AgentConfig {
            name: self.name().into(),
            description: self.description().into(),
            id: self.id,
            output_schema: structured_schema,
        }
    }

    /// Get the LLM provider
    pub fn llm(&self) -> Arc<dyn LLMProvider> {
        self.llm.clone()
    }

    /// Get the memory provider if available
    pub fn memory(&self) -> Option<Arc<RwLock<Box<dyn MemoryProvider>>>> {
        self.memory.clone()
    }
}

/// Handle for an agent that includes both the agent and its actor reference
pub struct AgentHandle<T: AgentDeriveT> {
    pub agent: Arc<BaseAgent<T>>,
    pub actor_ref: ActorRef<Task>,
}

impl<T: AgentDeriveT> AgentHandle<T> {
    /// Get the actor reference for direct messaging
    pub fn addr(&self) -> ActorRef<Task> {
        self.actor_ref.clone()
    }

    /// Get the agent reference
    pub fn agent(&self) -> Arc<BaseAgent<T>> {
        self.agent.clone()
    }
}

impl<T: AgentDeriveT> Debug for AgentHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentHandle")
            .field("agent", &self.agent)
            .finish()
    }
}

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

        let runtime = self.runtime.ok_or(AgentBuildError::BuildFailure("Runtime should be defined".into()))?;

        // Create agent actor
        let agent_actor = AgentActor(runnable.clone());
        let tx = runtime.tx().await;
        let actor_ref = Actor::spawn(Some(runnable.inner.name().into()), agent_actor, tx.clone()).await.map_err(AgentBuildError::SpawnError)?.0;

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
    use crate::agent::memory::MemoryProvider;
    use crate::agent::task::Task;
    use crate::agent::{AgentDeriveT, Context, ExecutorConfig};
    use async_trait::async_trait;
    use autoagents_llm::{chat::StructuredOutputFormat, LLMProvider};
    use autoagents_test_utils::agent::{MockAgentImpl, TestAgentOutput, TestError};
    use autoagents_test_utils::llm::MockLLMProvider;
    use std::sync::Arc;

    impl AgentOutputT for TestAgentOutput {
        fn output_schema() -> &'static str {
            r#"{"type":"object","properties":{"result":{"type":"string"}},"required":["result"]}"#
        }

        fn structured_output_format() -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "result": {"type": "string"}
                },
                "required": ["result"]
            })
        }
    }

    #[async_trait]
    impl AgentDeriveT for MockAgentImpl {
        type Output = TestAgentOutput;

        fn name(&self) -> &'static str {
            Box::leak(self.name.clone().into_boxed_str())
        }

        fn description(&self) -> &'static str {
            Box::leak(self.description.clone().into_boxed_str())
        }

        fn output_schema(&self) -> Option<Value> {
            Some(TestAgentOutput::structured_output_format())
        }

        fn tools(&self) -> Vec<Box<dyn ToolT>> {
            vec![]
        }
    }

    #[async_trait]
    impl AgentExecutor for MockAgentImpl {
        type Output = TestAgentOutput;
        type Error = TestError;

        fn config(&self) -> ExecutorConfig {
            ExecutorConfig::default()
        }

        async fn execute(
            &self,
            task: &Task,
            context: Context,
        ) -> Result<Self::Output, Self::Error> {
            if self.should_fail {
                return Err(TestError::TestError("Mock execution failed".to_string()));
            }

            Ok(TestAgentOutput {
                result: format!("Processed: {}", task.prompt),
            })
        }
    }

    #[test]
    fn test_agent_config_creation() {
        let config = AgentConfig {
            name: "test_agent".to_string(),
            id: Uuid::new_v4(),
            description: "A test agent".to_string(),
            output_schema: None,
        };

        assert_eq!(config.name, "test_agent");
        assert_eq!(config.description, "A test agent");
        assert!(config.output_schema.is_none());
    }

    #[test]
    fn test_agent_config_with_schema() {
        let schema = StructuredOutputFormat {
            name: "TestSchema".to_string(),
            description: Some("Test schema".to_string()),
            schema: Some(serde_json::json!({"type": "object"})),
            strict: Some(true),
        };

        let config = AgentConfig {
            name: "test_agent".to_string(),
            id: Uuid::new_v4(),
            description: "A test agent".to_string(),
            output_schema: Some(schema.clone()),
        };

        assert_eq!(config.name, "test_agent");
        assert_eq!(config.description, "A test agent");
        assert!(config.output_schema.is_some());
        assert_eq!(config.output_schema.unwrap().name, "TestSchema");
    }


    #[test]
    fn test_base_agent_creation() {
        let mock_agent = MockAgentImpl::new("test", "test description");
        let llm = Arc::new(MockLLMProvider);
        let base_agent = BaseAgent::new(mock_agent, llm, None, false);

        assert_eq!(base_agent.name(), "test");
        assert_eq!(base_agent.description(), "test description");
        assert!(base_agent.memory().is_none());
    }

    #[test]
    fn test_base_agent_with_memory() {
        let mock_agent = MockAgentImpl::new("test", "test description");
        let llm = Arc::new(MockLLMProvider);
        let memory = Box::new(crate::agent::memory::SlidingWindowMemory::new(5));
        let base_agent = BaseAgent::new(mock_agent, llm, Some(memory), false);

        assert_eq!(base_agent.name(), "test");
        assert_eq!(base_agent.description(), "test description");
        assert!(base_agent.memory().is_some());
    }

    #[test]
    fn test_base_agent_inner() {
        let mock_agent = MockAgentImpl::new("test", "test description");
        let llm = Arc::new(MockLLMProvider);
        let base_agent = BaseAgent::new(mock_agent, llm, None, false);

        let inner = base_agent.inner();
        assert_eq!(inner.name(), "test");
        assert_eq!(inner.description(), "test description");
    }

    #[test]
    fn test_base_agent_tools() {
        let mock_agent = MockAgentImpl::new("test", "test description");
        let llm = Arc::new(MockLLMProvider);
        let base_agent = BaseAgent::new(mock_agent, llm, None, false);

        let tools = base_agent.tools();
        assert!(tools.is_empty());
    }

    #[test]
    fn test_base_agent_llm() {
        let mock_agent = MockAgentImpl::new("test", "test description");
        let llm = Arc::new(MockLLMProvider);
        let base_agent = BaseAgent::new(mock_agent, llm.clone(), None, false);

        let agent_llm = base_agent.llm();
        // The llm() method returns Arc<dyn LLMProvider>, so we just verify it exists
        assert!(Arc::strong_count(&agent_llm) > 0);
    }

    #[test]
    fn test_base_agent_with_streaming() {
        let mock_agent = MockAgentImpl::new("streaming_agent", "test streaming agent");
        let llm = Arc::new(MockLLMProvider);
        let base_agent = BaseAgent::new(mock_agent, llm, None, true);

        assert_eq!(base_agent.name(), "streaming_agent");
        assert_eq!(base_agent.description(), "test streaming agent");
        assert!(base_agent.memory().is_none());
        assert_eq!(base_agent.stream, true);
    }

    #[test]
    fn test_agent_builder_with_subscribe_topic() {
        let mock_agent = MockAgentImpl::new("topic_agent", "test topic agent");
        let topic = Topic::<Task>::new("test_topic");

        let builder = AgentBuilder::new(mock_agent)
            .subscribe_topic(topic);

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

