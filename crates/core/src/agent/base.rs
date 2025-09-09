use crate::agent::config::AgentConfig;
use crate::agent::memory::MemoryProvider;
use crate::agent::{output::AgentOutputT, AgentExecutor, Context};
use crate::protocol::Event;
use crate::{protocol::ActorID, tool::ToolT};
use async_trait::async_trait;
use autoagents_llm::LLMProvider;

use serde_json::Value;
use std::marker::PhantomData;
use std::{fmt::Debug, sync::Arc};

#[cfg(target_arch = "wasm32")]
pub use futures::lock::Mutex;
#[cfg(not(target_arch = "wasm32"))]
pub use tokio::sync::Mutex;

#[cfg(target_arch = "wasm32")]
use futures::channel::mpsc::Sender;

#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::mpsc::Sender;

use crate::agent::error::RunnableAgentError;
use crate::agent::hooks::AgentHooks;
use uuid::Uuid;

/// Core trait that defines agent metadata and behavior
/// This trait is implemented via the #[agent] macro
#[async_trait]
pub trait AgentDeriveT: Send + Sync + 'static + Debug {
    /// The output type this agent produces
    type Output: AgentOutputT;

    /// Get the agent's description
    fn description(&self) -> &'static str;

    // If you provide None then its taken as String output
    fn output_schema(&self) -> Option<Value>;

    /// Get the agent's name
    fn name(&self) -> &'static str;

    /// Get the tools available to this agent
    fn tools(&self) -> Vec<Box<dyn ToolT>>;
}

pub trait AgentType: 'static + Send + Sync {
    fn type_name() -> &'static str;
}

/// Base agent type that wraps an AgentDeriveT implementation with additional runtime components
#[derive(Clone)]
pub struct BaseAgent<T: AgentDeriveT + AgentExecutor + AgentHooks + Send + Sync, A: AgentType> {
    /// The inner agent implementation (from macro)
    pub(crate) inner: Arc<T>,
    /// LLM provider for this agent
    pub(crate) llm: Arc<dyn LLMProvider>,
    /// Agent ID
    pub id: ActorID,
    /// Optional memory provider
    pub(crate) memory: Option<Arc<Mutex<Box<dyn MemoryProvider>>>>,
    /// Tx sender
    pub(crate) tx: Option<Sender<Event>>,
    //Stream
    pub(crate) stream: bool,
    pub(crate) marker: PhantomData<A>,
}

impl<T: AgentDeriveT + AgentExecutor + AgentHooks, A: AgentType> Debug for BaseAgent<T, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("A: {} - T: {}", self.inner().name(), A::type_name()).as_str())
    }
}

impl<T: AgentDeriveT + AgentExecutor + AgentHooks, A: AgentType> BaseAgent<T, A> {
    /// Create a new BaseAgent wrapping an AgentDeriveT implementation
    pub async fn new(
        inner: T,
        llm: Arc<dyn LLMProvider>,
        memory: Option<Box<dyn MemoryProvider>>,
        tx: Sender<Event>,
        stream: bool,
    ) -> Result<Self, RunnableAgentError> {
        let agent = Self {
            inner: Arc::new(inner),
            id: Uuid::new_v4(),
            llm,
            tx: Some(tx),
            memory: memory.map(|m| Arc::new(Mutex::new(m))),
            stream,
            marker: PhantomData,
        };

        //Run Hook
        agent.inner().on_agent_create().await;

        Ok(agent)
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

    pub fn stream(&self) -> bool {
        self.stream
    }

    pub(crate) fn create_context(&self) -> Arc<Context> {
        Arc::new(
            Context::new(self.llm(), self.tx.clone())
                .with_memory(self.memory())
                .with_tools(self.tools())
                .with_config(self.agent_config())
                .with_stream(self.stream()),
        )
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
    pub fn memory(&self) -> Option<Arc<Mutex<Box<dyn MemoryProvider>>>> {
        self.memory.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AgentConfig, DirectAgent};
    use crate::tests::agent::MockAgentImpl;
    use autoagents_llm::chat::StructuredOutputFormat;
    use autoagents_test_utils::llm::MockLLMProvider;
    use std::sync::Arc;
    use tokio::sync::mpsc::{channel, Receiver};
    use uuid::Uuid;

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

    #[tokio::test]
    async fn test_base_agent_creation() {
        let mock_agent = MockAgentImpl::new("test", "test description");
        let llm = Arc::new(MockLLMProvider);
        let (tx, _): (Sender<Event>, Receiver<Event>) = channel(32);
        let base_agent = BaseAgent::<_, DirectAgent>::new(mock_agent, llm, None, tx, false)
            .await
            .unwrap();

        assert_eq!(base_agent.name(), "test");
        assert_eq!(base_agent.description(), "test description");
        assert!(base_agent.memory().is_none());
    }

    #[tokio::test]
    async fn test_base_agent_with_memory() {
        let mock_agent = MockAgentImpl::new("test", "test description");
        let llm = Arc::new(MockLLMProvider);
        let memory = Box::new(crate::agent::memory::SlidingWindowMemory::new(5));
        let (tx, _): (Sender<Event>, Receiver<Event>) = channel(32);
        let base_agent = BaseAgent::<_, DirectAgent>::new(mock_agent, llm, Some(memory), tx, false)
            .await
            .unwrap();

        assert_eq!(base_agent.name(), "test");
        assert_eq!(base_agent.description(), "test description");
        assert!(base_agent.memory().is_some());
    }

    #[tokio::test]
    async fn test_base_agent_inner() {
        let mock_agent = MockAgentImpl::new("test", "test description");
        let llm = Arc::new(MockLLMProvider);
        let (tx, _): (Sender<Event>, Receiver<Event>) = channel(32);
        let base_agent = BaseAgent::<_, DirectAgent>::new(mock_agent, llm, None, tx, false)
            .await
            .unwrap();

        let inner = base_agent.inner();
        assert_eq!(inner.name(), "test");
        assert_eq!(inner.description(), "test description");
    }

    #[tokio::test]
    async fn test_base_agent_tools() {
        let mock_agent = MockAgentImpl::new("test", "test description");
        let llm = Arc::new(MockLLMProvider);
        let (tx, _): (Sender<Event>, Receiver<Event>) = channel(32);
        let base_agent = BaseAgent::<_, DirectAgent>::new(mock_agent, llm, None, tx, false)
            .await
            .unwrap();

        let tools = base_agent.tools();
        assert!(tools.is_empty());
    }

    #[tokio::test]
    async fn test_base_agent_llm() {
        let mock_agent = MockAgentImpl::new("test", "test description");
        let llm = Arc::new(MockLLMProvider);
        let (tx, _): (Sender<Event>, Receiver<Event>) = channel(32);
        let base_agent = BaseAgent::<_, DirectAgent>::new(mock_agent, llm.clone(), None, tx, false)
            .await
            .unwrap();

        let agent_llm = base_agent.llm();
        // The llm() method returns Arc<dyn LLMProvider>, so we just verify it exists
        assert!(Arc::strong_count(&agent_llm) > 0);
    }

    #[tokio::test]
    async fn test_base_agent_with_streaming() {
        let mock_agent = MockAgentImpl::new("streaming_agent", "test streaming agent");
        let llm = Arc::new(MockLLMProvider);
        let (tx, _): (Sender<Event>, Receiver<Event>) = channel(32);
        let base_agent = BaseAgent::<_, DirectAgent>::new(mock_agent, llm, None, tx, true)
            .await
            .unwrap();

        assert_eq!(base_agent.name(), "streaming_agent");
        assert_eq!(base_agent.description(), "test streaming agent");
        assert!(base_agent.memory().is_none());
        assert!(base_agent.stream);
    }
}
