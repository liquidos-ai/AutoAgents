use super::{
    error::AgentBuildError, output::AgentOutputT, AgentExecutor, IntoRunnable, RunnableAgent,
};
use crate::{error::Error, memory::MemoryProvider, protocol::AgentID, runtime::Runtime};
use async_trait::async_trait;
use autoagents_llm::{chat::StructuredOutputFormat, LLMProvider, ToolT};
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

pub struct AgentConfig {
    /// The agent's name
    pub name: String,
    /// The agent's description
    pub description: String,
    /// The Agent ID
    pub id: AgentID,
    /// The output schema for the agent
    pub output_schema: Option<StructuredOutputFormat>,
}

/// Base agent type that wraps an AgentDeriveT implementation with additional runtime components
#[derive(Clone)]
pub struct BaseAgent<T: AgentDeriveT> {
    /// The inner agent implementation (from macro)
    pub inner: Arc<T>,
    /// LLM provider for this agent
    pub llm: Arc<dyn LLMProvider>,
    // Agent ID
    pub id: AgentID,
    /// Optional memory provider
    pub memory: Option<Arc<RwLock<Box<dyn MemoryProvider>>>>,
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
    ) -> Self {
        // Convert tools to Arc for efficient sharing
        Self {
            inner: Arc::new(inner),
            id: Uuid::new_v4(),
            llm,
            memory: memory.map(|m| Arc::new(RwLock::new(m))),
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

/// Builder for creating BaseAgent instances from AgentDeriveT implementations
pub struct AgentBuilder<T: AgentDeriveT + AgentExecutor> {
    inner: T,
    llm: Option<Arc<dyn LLMProvider>>,
    memory: Option<Box<dyn MemoryProvider>>,
    runtime: Option<Arc<dyn Runtime>>,
    subscribed_topics: Vec<String>,
}

impl<T: AgentDeriveT + AgentExecutor> AgentBuilder<T> {
    /// Create a new builder with an AgentDeriveT implementation
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            llm: None,
            memory: None,
            runtime: None,
            subscribed_topics: vec![],
        }
    }

    /// Set the LLM provider
    pub fn with_llm(mut self, llm: Arc<dyn LLMProvider>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Set the memory provider
    pub fn with_memory(mut self, memory: Box<dyn MemoryProvider>) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn subscribe_topic<S: Into<String>>(mut self, topic: S) -> Self {
        self.subscribed_topics.push(topic.into());
        self
    }

    /// Build the BaseAgent
    pub async fn build(self) -> Result<Arc<dyn RunnableAgent>, Error> {
        let llm = self.llm.ok_or(AgentBuildError::BuildFailure(
            "LLM provider is required".to_string(),
        ))?;
        let runnable = BaseAgent::new(self.inner, llm, self.memory).into_runnable();
        if let Some(runtime) = self.runtime {
            runtime.register_agent(runnable.clone()).await?;
            for topic in self.subscribed_topics {
                runtime.subscribe(runnable.id(), topic).await?;
            }
        } else {
            return Err(AgentBuildError::BuildFailure("Runtime should be defined".into()).into());
        }
        Ok(runnable)
    }

    pub fn runtime(mut self, runtime: Arc<dyn Runtime>) -> Self {
        self.runtime = Some(runtime);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AgentDeriveT, AgentState, ExecutorConfig};
    use crate::memory::MemoryProvider;
    use crate::protocol::Event;
    use crate::runtime::Task;
    use async_trait::async_trait;
    use autoagents_llm::{chat::StructuredOutputFormat, LLMProvider, ToolT};
    use autoagents_test_utils::agent::{MockAgentImpl, TestAgentOutput, TestError};
    use autoagents_test_utils::llm::MockLLMProvider;
    use std::sync::Arc;
    use tokio::sync::mpsc;

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
            _llm: Arc<dyn LLMProvider>,
            _memory: Option<Arc<RwLock<Box<dyn MemoryProvider>>>>,
            _tools: Vec<Box<dyn ToolT>>,
            _agent_config: &AgentConfig,
            task: Task,
            _state: Arc<RwLock<AgentState>>,
            _tx_event: mpsc::Sender<Event>,
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
        let base_agent = BaseAgent::new(mock_agent, llm, None);

        assert_eq!(base_agent.name(), "test");
        assert_eq!(base_agent.description(), "test description");
        assert!(base_agent.memory().is_none());
    }

    #[test]
    fn test_base_agent_with_memory() {
        let mock_agent = MockAgentImpl::new("test", "test description");
        let llm = Arc::new(MockLLMProvider);
        let memory = Box::new(crate::memory::SlidingWindowMemory::new(5));
        let base_agent = BaseAgent::new(mock_agent, llm, Some(memory));

        assert_eq!(base_agent.name(), "test");
        assert_eq!(base_agent.description(), "test description");
        assert!(base_agent.memory().is_some());
    }

    #[test]
    fn test_base_agent_inner() {
        let mock_agent = MockAgentImpl::new("test", "test description");
        let llm = Arc::new(MockLLMProvider);
        let base_agent = BaseAgent::new(mock_agent, llm, None);

        let inner = base_agent.inner();
        assert_eq!(inner.name(), "test");
        assert_eq!(inner.description(), "test description");
    }

    #[test]
    fn test_base_agent_tools() {
        let mock_agent = MockAgentImpl::new("test", "test description");
        let llm = Arc::new(MockLLMProvider);
        let base_agent = BaseAgent::new(mock_agent, llm, None);

        let tools = base_agent.tools();
        assert!(tools.is_empty());
    }

    #[test]
    fn test_base_agent_llm() {
        let mock_agent = MockAgentImpl::new("test", "test description");
        let llm = Arc::new(MockLLMProvider);
        let base_agent = BaseAgent::new(mock_agent, llm.clone(), None);

        let agent_llm = base_agent.llm();
        // The llm() method returns Arc<dyn LLMProvider>, so we just verify it exists
        assert!(Arc::strong_count(&agent_llm) > 0);
    }
}
