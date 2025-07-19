use crate::agent::base::AgentConfig;
use crate::agent::runnable::AgentState;
use crate::memory::MemoryProvider;
use crate::protocol::Event;
use crate::session::Task;
use async_trait::async_trait;
use autoagents_llm::{LLMProvider, ToolT};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;
use std::error::Error;
use std::fmt::Debug;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

/// Result of processing a single turn in the agent's execution
#[derive(Debug)]
pub enum TurnResult<T> {
    /// Continue processing with optional intermediate data
    Continue(Option<T>),
    /// Final result obtained
    Complete(T),
    /// Error occurred but can continue
    Error(String),
    /// Fatal error, must stop
    Fatal(Box<dyn Error + Send + Sync>),
}

/// Configuration for executors
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    pub max_turns: usize,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self { max_turns: 10 }
    }
}

/// Base trait for agent execution strategies
///
/// Executors are responsible for implementing the specific execution logic
/// for agents, such as ReAct loops, chain-of-thought, or custom patterns.
#[async_trait]
pub trait AgentExecutor: Send + Sync + 'static {
    type Output: Serialize + DeserializeOwned + Clone + Send + Sync + Into<Value>;
    type Error: Error + Send + Sync + 'static;

    /// Get the configuration for this executor
    fn config(&self) -> ExecutorConfig;

    /// Execute the agent with the given task
    #[allow(clippy::too_many_arguments)]
    async fn execute(
        &self,
        llm: Arc<dyn LLMProvider>,
        memory: Option<Arc<RwLock<Box<dyn MemoryProvider>>>>,
        tools: Vec<Box<dyn ToolT>>,
        agent_config: &AgentConfig,
        task: Task,
        state: Arc<RwLock<AgentState>>,
        tx_event: mpsc::Sender<Event>,
    ) -> Result<Self::Output, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::base::AgentConfig;
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
    struct TestOutput {
        message: String,
    }

    impl From<TestOutput> for Value {
        fn from(output: TestOutput) -> Self {
            serde_json::to_value(output).unwrap_or(Value::Null)
        }
    }

    #[derive(Debug, thiserror::Error)]
    enum TestError {
        #[error("Test error: {0}")]
        TestError(String),
    }

    struct MockExecutor {
        should_fail: bool,
        max_turns: usize,
    }

    impl MockExecutor {
        fn new(should_fail: bool) -> Self {
            Self {
                should_fail,
                max_turns: 5,
            }
        }

        fn with_max_turns(max_turns: usize) -> Self {
            Self {
                should_fail: false,
                max_turns,
            }
        }
    }

    #[async_trait]
    impl AgentExecutor for MockExecutor {
        type Output = TestOutput;
        type Error = TestError;

        fn config(&self) -> ExecutorConfig {
            ExecutorConfig {
                max_turns: self.max_turns,
            }
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

            Ok(TestOutput {
                message: format!("Processed: {}", task.prompt),
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
    fn test_executor_config_default() {
        let config = ExecutorConfig::default();
        assert_eq!(config.max_turns, 10);
    }

    #[test]
    fn test_executor_config_custom() {
        let config = ExecutorConfig { max_turns: 5 };
        assert_eq!(config.max_turns, 5);
    }

    #[test]
    fn test_executor_config_clone() {
        let config = ExecutorConfig { max_turns: 15 };
        let cloned = config.clone();
        assert_eq!(config.max_turns, cloned.max_turns);
    }

    #[test]
    fn test_executor_config_debug() {
        let config = ExecutorConfig { max_turns: 20 };
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("ExecutorConfig"));
        assert!(debug_str.contains("20"));
    }

    #[test]
    fn test_turn_result_continue() {
        let result = TurnResult::<String>::Continue(Some("partial".to_string()));
        match result {
            TurnResult::Continue(Some(data)) => assert_eq!(data, "partial"),
            _ => panic!("Expected Continue variant"),
        }
    }

    #[test]
    fn test_turn_result_continue_none() {
        let result = TurnResult::<String>::Continue(None);
        match result {
            TurnResult::Continue(None) => {}
            _ => panic!("Expected Continue(None) variant"),
        }
    }

    #[test]
    fn test_turn_result_complete() {
        let result = TurnResult::Complete("final".to_string());
        match result {
            TurnResult::Complete(data) => assert_eq!(data, "final"),
            _ => panic!("Expected Complete variant"),
        }
    }

    #[test]
    fn test_turn_result_error() {
        let result = TurnResult::<String>::Error("error message".to_string());
        match result {
            TurnResult::Error(msg) => assert_eq!(msg, "error message"),
            _ => panic!("Expected Error variant"),
        }
    }

    #[test]
    fn test_turn_result_fatal() {
        let error = Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Fatal error",
        ));
        let result = TurnResult::<String>::Fatal(error);
        match result {
            TurnResult::Fatal(err) => assert_eq!(err.to_string(), "Fatal error"),
            _ => panic!("Expected Fatal variant"),
        }
    }

    #[test]
    fn test_turn_result_debug() {
        let result = TurnResult::Complete("test".to_string());
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("Complete"));
        assert!(debug_str.contains("test"));
    }

    #[tokio::test]
    async fn test_mock_executor_success() {
        let executor = MockExecutor::new(false);
        let llm = Arc::new(MockLLMProvider);
        let tools = vec![];
        let agent_config = AgentConfig {
            name: "test".to_string(),
            description: "test agent".to_string(),
            output_schema: None,
        };
        let task = Task::new("test task", None);
        let state = Arc::new(RwLock::new(AgentState::new()));
        let (tx_event, _rx_event) = mpsc::channel(100);

        let result = executor
            .execute(llm, None, tools, &agent_config, task, state, tx_event)
            .await;

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.message, "Processed: test task");
    }

    #[tokio::test]
    async fn test_mock_executor_failure() {
        let executor = MockExecutor::new(true);
        let llm = Arc::new(MockLLMProvider);
        let tools = vec![];
        let agent_config = AgentConfig {
            name: "test".to_string(),
            description: "test agent".to_string(),
            output_schema: None,
        };
        let task = Task::new("test task", None);
        let state = Arc::new(RwLock::new(AgentState::new()));
        let (tx_event, _rx_event) = mpsc::channel(100);

        let result = executor
            .execute(llm, None, tools, &agent_config, task, state, tx_event)
            .await;

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert_eq!(error.to_string(), "Test error: Mock execution failed");
    }

    #[test]
    fn test_mock_executor_config() {
        let executor = MockExecutor::with_max_turns(3);
        let config = executor.config();
        assert_eq!(config.max_turns, 3);
    }

    #[test]
    fn test_mock_executor_config_default() {
        let executor = MockExecutor::new(false);
        let config = executor.config();
        assert_eq!(config.max_turns, 5);
    }

    #[test]
    fn test_test_output_serialization() {
        let output = TestOutput {
            message: "test message".to_string(),
        };
        let serialized = serde_json::to_string(&output).unwrap();
        assert!(serialized.contains("test message"));
    }

    #[test]
    fn test_test_output_deserialization() {
        let json = r#"{"message":"test message"}"#;
        let output: TestOutput = serde_json::from_str(json).unwrap();
        assert_eq!(output.message, "test message");
    }

    #[test]
    fn test_test_output_clone() {
        let output = TestOutput {
            message: "original".to_string(),
        };
        let cloned = output.clone();
        assert_eq!(output.message, cloned.message);
    }

    #[test]
    fn test_test_output_debug() {
        let output = TestOutput {
            message: "debug test".to_string(),
        };
        let debug_str = format!("{:?}", output);
        assert!(debug_str.contains("TestOutput"));
        assert!(debug_str.contains("debug test"));
    }

    #[test]
    fn test_test_output_into_value() {
        let output = TestOutput {
            message: "value test".to_string(),
        };
        let value: Value = output.into();
        assert_eq!(value["message"], "value test");
    }

    #[test]
    fn test_test_error_display() {
        let error = TestError::TestError("display test".to_string());
        assert_eq!(error.to_string(), "Test error: display test");
    }

    #[test]
    fn test_test_error_debug() {
        let error = TestError::TestError("debug test".to_string());
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("TestError"));
        assert!(debug_str.contains("debug test"));
    }

    #[test]
    fn test_test_error_source() {
        let error = TestError::TestError("source test".to_string());
        assert!(error.source().is_none());
    }
}
