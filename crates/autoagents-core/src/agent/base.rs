use crate::agent::config::AgentConfig;
use crate::agent::memory::MemoryProvider;
use crate::agent::{AgentExecutor, Context, output::AgentOutputT};
use crate::tool::ToolT;
use async_trait::async_trait;
use autoagents_llm::LLMProvider;
use autoagents_protocol::{ActorID, Event};

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
    fn description(&self) -> &str;

    // If you provide None then its taken as String output
    fn output_schema(&self) -> Option<Value>;

    /// Get the agent's name
    fn name(&self) -> &str;

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
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    /// Get the agent's description
    pub fn description(&self) -> &str {
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
        let structured_schema =
            output_schema.and_then(|schema| serde_json::from_value(schema).ok());
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
    use crate::agent::memory::SlidingWindowMemory;
    use crate::agent::{AgentConfig, DirectAgent};
    use crate::tests::{MockAgentImpl, MockLLMProvider};
    use autoagents_llm::chat::StructuredOutputFormat;
    use std::sync::Arc;
    use tokio::sync::mpsc::{Receiver, channel};
    use uuid::Uuid;

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
    async fn test_base_agent_creation_with_memory_and_stream() {
        let mock_agent = MockAgentImpl::new("test", "test description");
        let llm = Arc::new(MockLLMProvider);
        let memory = Box::new(SlidingWindowMemory::new(5));
        let (tx, _): (Sender<Event>, Receiver<Event>) = channel(32);
        let base_agent = BaseAgent::<_, DirectAgent>::new(mock_agent, llm, Some(memory), tx, true)
            .await
            .unwrap();

        assert_eq!(base_agent.name(), "test");
        assert_eq!(base_agent.description(), "test description");
        assert!(base_agent.memory().is_some());
        assert!(base_agent.stream);
    }

    #[tokio::test]
    async fn test_base_agent_create_context_populates_config() {
        let mock_agent = MockAgentImpl::new("ctx_agent", "context agent");
        let llm = Arc::new(MockLLMProvider);
        let (tx, _): (Sender<Event>, Receiver<Event>) = channel(32);
        let base_agent = BaseAgent::<_, DirectAgent>::new(mock_agent, llm, None, tx, false)
            .await
            .unwrap();

        let context = base_agent.create_context();
        let config = context.config();
        assert_eq!(config.name, "ctx_agent");
        assert_eq!(config.description, "context agent");
    }
}
