use super::{
    error::AgentBuildError, output::AgentOutputT, AgentExecutor, IntoRunnable, RunnableAgent,
};
use crate::{error::Error, memory::MemoryProvider};
use async_trait::async_trait;
use autoagents_llm::{chat::StructuredOutputFormat, LLMProvider, ToolT};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Core trait that defines agent metadata and behavior
/// This trait is implemented via the #[agent] macro
#[async_trait]
pub trait AgentDeriveT: Send + Sync + 'static + AgentExecutor {
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
    /// Optional memory provider
    pub memory: Option<Arc<RwLock<Box<dyn MemoryProvider>>>>,
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
}

impl<T: AgentDeriveT + AgentExecutor> AgentBuilder<T> {
    /// Create a new builder with an AgentDeriveT implementation
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            llm: None,
            memory: None,
        }
    }

    /// Create a builder from an existing agent (for compatibility)
    pub fn from_agent(inner: T) -> Self {
        Self::new(inner)
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

    /// Build the BaseAgent
    pub fn build(self) -> Result<Arc<dyn RunnableAgent>, Error> {
        let llm = self.llm.ok_or(AgentBuildError::BuildFailure(
            "LLM provider is required".to_string(),
        ))?;
        Ok(BaseAgent::new(self.inner, llm, self.memory).into_runnable())
    }

    /// Build the BaseAgent with memory (for compatibility)
    pub fn build_with_memory(
        self,
        memory: Box<dyn MemoryProvider>,
    ) -> Result<Arc<dyn RunnableAgent>, Error> {
        let llm = self.llm.ok_or(AgentBuildError::BuildFailure(
            "LLM provider is required".to_string(),
        ))?;
        Ok(BaseAgent::new(self.inner, llm, Some(memory)).into_runnable())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::executor::{AgentExecutor, ExecutorConfig};
    use crate::agent::output::AgentOutputT;
    use crate::agent::runnable::AgentState;
    use crate::memory::MemoryProvider;
    use crate::protocol::Event;
    use crate::session::Task;
    use async_trait::async_trait;
    use autoagents_llm::{
        chat::{ChatMessage, ChatProvider, ChatResponse, StructuredOutputFormat},
        completion::{CompletionProvider, CompletionRequest, CompletionResponse},
        embedding::EmbeddingProvider,
        error::LLMError,
        models::ModelsProvider,
        LLMProvider, ToolCall, ToolT,
    };
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;
    use tokio::sync::{mpsc, RwLock};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct TestAgentOutput {
        result: String,
    }

    impl From<TestAgentOutput> for Value {
        fn from(output: TestAgentOutput) -> Self {
            serde_json::to_value(output).unwrap_or(Value::Null)
        }
    }

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

    #[derive(Debug, thiserror::Error)]
    enum TestError {
        #[error("Test error: {0}")]
        TestError(String),
    }

    struct MockAgentImpl {
        name: String,
        description: String,
        should_fail: bool,
    }

    impl MockAgentImpl {
        fn new(name: &str, description: &str) -> Self {
            Self {
                name: name.to_string(),
                description: description.to_string(),
                should_fail: false,
            }
        }

        fn with_failure(name: &str, description: &str) -> Self {
            Self {
                name: name.to_string(),
                description: description.to_string(),
                should_fail: true,
            }
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

    // Mock LLM Provider
    struct MockLLMProvider;

    #[async_trait]
    impl ChatProvider for MockLLMProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[autoagents_llm::chat::Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            Ok(Box::new(MockChatResponse {
                text: Some("Mock response".to_string()),
            }))
        }
    }

    #[async_trait]
    impl CompletionProvider for MockLLMProvider {
        async fn complete(
            &self,
            _req: &CompletionRequest,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<CompletionResponse, LLMError> {
            Ok(CompletionResponse {
                text: "Mock completion".to_string(),
            })
        }
    }

    #[async_trait]
    impl EmbeddingProvider for MockLLMProvider {
        async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
            Ok(vec![vec![0.1, 0.2, 0.3]])
        }
    }

    #[async_trait]
    impl ModelsProvider for MockLLMProvider {}

    impl LLMProvider for MockLLMProvider {}

    struct MockChatResponse {
        text: Option<String>,
    }

    impl ChatResponse for MockChatResponse {
        fn text(&self) -> Option<String> {
            self.text.clone()
        }

        fn tool_calls(&self) -> Option<Vec<ToolCall>> {
            None
        }
    }

    impl std::fmt::Debug for MockChatResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "MockChatResponse")
        }
    }

    impl std::fmt::Display for MockChatResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.text.as_deref().unwrap_or(""))
        }
    }

    #[test]
    fn test_agent_config_creation() {
        let config = AgentConfig {
            name: "test_agent".to_string(),
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
