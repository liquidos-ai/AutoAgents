use crate::agent::config::AgentConfig;
use crate::agent::memory::MemoryProvider;
use crate::agent::task::Task;
use crate::agent::{output::AgentOutputT, AgentExecutor, Context};
use crate::protocol::Event;
use crate::{protocol::ActorID, tool::ToolT};
use async_trait::async_trait;
use autoagents_llm::LLMProvider;
#[cfg(not(target_arch = "wasm32"))]
use ractor::ActorRef;

use serde_json::Value;
use std::marker::PhantomData;
use std::{fmt::Debug, sync::Arc};
use uuid::Uuid;

#[cfg(not(target_arch = "wasm32"))]
pub use tokio::sync::{mpsc::Sender, Mutex};

use crate::agent::error::RunnableAgentError;
use crate::error::Error;
#[cfg(target_arch = "wasm32")]
pub use futures::channel::mpsc::Sender;
#[cfg(target_arch = "wasm32")]
pub use futures::lock::Mutex;
use futures::Stream;

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

pub trait AgentType: 'static + Send + Sync {
    fn type_name() -> &'static str;
}

pub struct DirectAgent {}

impl AgentType for DirectAgent {
    fn type_name() -> &'static str {
        "direct_agent"
    }
}

pub struct ActorAgent {}

impl AgentType for ActorAgent {
    fn type_name() -> &'static str {
        "protocol_agent"
    }
}

/// Base agent type that wraps an AgentDeriveT implementation with additional runtime components
#[derive(Clone)]
pub struct BaseAgent<T: AgentDeriveT, A: AgentType> {
    /// The inner agent implementation (from macro)
    inner: Arc<T>,
    /// LLM provider for this agent
    llm: Arc<dyn LLMProvider>,
    /// Agent ID
    pub id: ActorID,
    /// Optional memory provider
    memory: Option<Arc<Mutex<Box<dyn MemoryProvider>>>>,
    /// Tx sender
    tx: Option<Sender<Event>>,
    //Stream
    stream: bool,
    marker: PhantomData<A>,
}

impl<T: AgentDeriveT, A: AgentType> Debug for BaseAgent<T, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("A: {} - T: {}", self.inner().name(), A::type_name()).as_str())
    }
}

impl<T: AgentDeriveT> BaseAgent<T, DirectAgent> {
    /// Create a new BaseAgent wrapping an AgentDeriveT implementation without sender
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
            tx: None,
            memory: memory.map(|m| Arc::new(Mutex::new(m))),
            stream,
            marker: PhantomData,
        }
    }

    pub async fn run(&self, task: Task) -> Result<<T as AgentDeriveT>::Output, RunnableAgentError>
    where
        <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output>,
    {
        let context = Context::new(self.llm(), None)
            .with_memory(self.memory())
            .with_tools(self.tools())
            .with_config(self.agent_config())
            .with_stream(self.stream());

        // Execute the agent's logic using the executor
        match self.inner().execute(&task, Arc::new(context)).await {
            Ok(output) => {
                let output: <T as AgentExecutor>::Output = output;
                Ok(output.into())
            }
            Err(e) => {
                // Send error event
                Err(RunnableAgentError::ExecutorError(e.to_string()))
            }
        }
    }

    pub async fn run_stream(
        &self,
        task: Task,
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<<T as AgentDeriveT>::Output, Error>> + Send>>,
        RunnableAgentError,
    >
    where
        <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output>,
    {
        let context = Context::new(self.llm(), None)
            .with_memory(self.memory())
            .with_tools(self.tools())
            .with_config(self.agent_config())
            .with_stream(self.stream());

        // Execute the agent's streaming logic using the executor
        match self.inner().execute_stream(&task, Arc::new(context)).await {
            Ok(stream) => {
                use futures::StreamExt;
                // Transform the stream - the From implementation handles streaming safely
                let transformed_stream = stream.map(move |result| match result {
                    Ok(output) => Ok(output.into()),
                    Err(e) => {
                        let error_msg = e.to_string();
                        Err(RunnableAgentError::ExecutorError(error_msg).into())
                    }
                });

                Ok(Box::pin(transformed_stream))
            }
            Err(e) => {
                // Send error event for stream creation failure
                Err(RunnableAgentError::ExecutorError(e.to_string()))
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<T: AgentDeriveT> BaseAgent<T, ActorAgent> {
    /// Create a new BaseAgent wrapping an AgentDeriveT implementation
    pub fn new(
        inner: T,
        llm: Arc<dyn LLMProvider>,
        memory: Option<Box<dyn MemoryProvider>>,
        tx: Sender<Event>,
        stream: bool,
    ) -> Self {
        // Convert tools to Arc for efficient sharing
        Self {
            inner: Arc::new(inner),
            id: Uuid::new_v4(),
            llm,
            tx: Some(tx),
            memory: memory.map(|m| Arc::new(Mutex::new(m))),
            stream,
            marker: PhantomData,
        }
    }

    pub fn tx(&self) -> Result<Sender<Event>, RunnableAgentError> {
        self.tx.clone().ok_or(RunnableAgentError::EmptyTx)
    }

    pub async fn run(self: Arc<Self>, task: Task) -> Result<(), RunnableAgentError>
    where
        Value: From<<T as AgentExecutor>::Output>,
    {
        let submission_id = task.submission_id;
        let tx = self.tx().map_err(|_| RunnableAgentError::EmptyTx)?;
        let context = Context::new(self.llm(), Some(tx.clone()))
            .with_memory(self.memory())
            .with_tools(self.tools())
            .with_config(self.agent_config())
            .with_stream(self.stream());

        // Execute the agent's logic using the executor
        match self.inner().execute(&task, Arc::new(context)).await {
            Ok(output) => {
                let value: Value = output.into();

                #[cfg(not(target_arch = "wasm32"))]
                tx.send(Event::TaskComplete {
                    sub_id: submission_id,
                    result: serde_json::to_string_pretty(&value)
                        .map_err(|e| RunnableAgentError::ExecutorError(e.to_string()))?,
                })
                .await
                .map_err(|e| RunnableAgentError::ExecutorError(e.to_string()))?;

                Ok(())
            }
            Err(e) => {
                #[cfg(not(target_arch = "wasm32"))]
                tx.send(Event::TaskError {
                    sub_id: submission_id,
                    error: e.to_string(),
                })
                .await
                .map_err(|e| RunnableAgentError::ExecutorError(e.to_string()))?;
                Err(RunnableAgentError::ExecutorError(e.to_string()))
            }
        }
    }

    pub async fn run_stream(
        self: Arc<Self>,
        task: Task,
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<Value, RunnableAgentError>> + Send>>,
        RunnableAgentError,
    >
    where
        Value: From<<T as AgentExecutor>::Output>,
    {
        // let submission_id = task.submission_id;
        let tx = self.tx().map_err(|_| RunnableAgentError::EmptyTx)?;
        let context = Context::new(self.llm(), Some(tx))
            .with_memory(self.memory())
            .with_tools(self.tools())
            .with_config(self.agent_config())
            .with_stream(self.stream());

        // Execute the agent's streaming logic using the executor
        match self.inner().execute_stream(&task, Arc::new(context)).await {
            Ok(stream) => {
                use futures::StreamExt;
                // Transform the stream to convert agent output to TaskResult
                let transformed_stream = stream.map(move |result| {
                    match result {
                        Ok(output) => Ok(output.into()),
                        Err(e) => {
                            // Handle error
                            let error_msg = e.to_string();
                            Err(RunnableAgentError::ExecutorError(error_msg))
                        }
                    }
                });

                Ok(Box::pin(transformed_stream))
            }
            Err(e) => {
                // Send error event for stream creation failure
                Err(RunnableAgentError::ExecutorError(e.to_string()))
            }
        }
    }
}

impl<T: AgentDeriveT, A: AgentType> BaseAgent<T, A> {
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

/// Handle for an agent that includes both the agent and its actor reference
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
pub struct AgentHandle<T: AgentDeriveT, A: AgentType> {
    pub agent: Arc<BaseAgent<T, A>>,
    pub actor_ref: ActorRef<Task>,
}

#[cfg(not(target_arch = "wasm32"))]
impl<T: AgentDeriveT, A: AgentType> AgentHandle<T, A> {
    /// Get the actor reference for direct messaging
    pub fn addr(&self) -> ActorRef<Task> {
        self.actor_ref.clone()
    }

    /// Get the agent reference
    pub fn agent(&self) -> Arc<BaseAgent<T, A>> {
        self.agent.clone()
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<T: AgentDeriveT, A: AgentType> Debug for AgentHandle<T, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentHandle")
            .field("agent", &self.agent)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::task::Task;
    use crate::agent::{AgentDeriveT, Context, ExecutorConfig};
    use async_trait::async_trait;
    use autoagents_llm::chat::StructuredOutputFormat;
    use autoagents_test_utils::agent::{MockAgentImpl, TestAgentOutput, TestError};
    use autoagents_test_utils::llm::MockLLMProvider;
    use futures::Stream;
    use std::sync::Arc;

    impl AgentOutputT for TestAgentOutput {
        fn output_schema() -> &'static str {
            r#"{"type":"object","properties":{"result":{"type":"string"}},"required":["result"]}"#
        }

        fn structured_output_format() -> serde_json::Value {
            serde_json::json!({
                "name": "TestAgentOutput",
                "description": "Test agent output schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "result": {"type": "string"}
                    },
                    "required": ["result"]
                },
                "strict": true
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
            _context: Arc<Context>,
        ) -> Result<Self::Output, Self::Error> {
            if self.should_fail {
                return Err(TestError::TestError("Mock execution failed".to_string()));
            }

            Ok(TestAgentOutput {
                result: format!("Processed: {}", task.prompt),
            })
        }
        async fn execute_stream(
            &self,
            _task: &Task,
            _context: Arc<Context>,
        ) -> Result<
            std::pin::Pin<Box<dyn Stream<Item = Result<Self::Output, Self::Error>> + Send>>,
            Self::Error,
        > {
            unimplemented!()
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
        let base_agent = BaseAgent::<_, DirectAgent>::new(mock_agent, llm, None, false);

        assert_eq!(base_agent.name(), "test");
        assert_eq!(base_agent.description(), "test description");
        assert!(base_agent.memory().is_none());
    }

    #[test]
    fn test_base_agent_with_memory() {
        let mock_agent = MockAgentImpl::new("test", "test description");
        let llm = Arc::new(MockLLMProvider);
        let memory = Box::new(crate::agent::memory::SlidingWindowMemory::new(5));
        let base_agent = BaseAgent::<_, DirectAgent>::new(mock_agent, llm, Some(memory), false);

        assert_eq!(base_agent.name(), "test");
        assert_eq!(base_agent.description(), "test description");
        assert!(base_agent.memory().is_some());
    }

    #[test]
    fn test_base_agent_inner() {
        let mock_agent = MockAgentImpl::new("test", "test description");
        let llm = Arc::new(MockLLMProvider);
        let base_agent = BaseAgent::<_, DirectAgent>::new(mock_agent, llm, None, false);

        let inner = base_agent.inner();
        assert_eq!(inner.name(), "test");
        assert_eq!(inner.description(), "test description");
    }

    #[test]
    fn test_base_agent_tools() {
        let mock_agent = MockAgentImpl::new("test", "test description");
        let llm = Arc::new(MockLLMProvider);
        let base_agent = BaseAgent::<_, DirectAgent>::new(mock_agent, llm, None, false);

        let tools = base_agent.tools();
        assert!(tools.is_empty());
    }

    #[test]
    fn test_base_agent_llm() {
        let mock_agent = MockAgentImpl::new("test", "test description");
        let llm = Arc::new(MockLLMProvider);
        let base_agent = BaseAgent::<_, DirectAgent>::new(mock_agent, llm.clone(), None, false);

        let agent_llm = base_agent.llm();
        // The llm() method returns Arc<dyn LLMProvider>, so we just verify it exists
        assert!(Arc::strong_count(&agent_llm) > 0);
    }

    #[test]
    fn test_base_agent_with_streaming() {
        let mock_agent = MockAgentImpl::new("streaming_agent", "test streaming agent");
        let llm = Arc::new(MockLLMProvider);
        let base_agent = BaseAgent::<_, DirectAgent>::new(mock_agent, llm, None, true);

        assert_eq!(base_agent.name(), "streaming_agent");
        assert_eq!(base_agent.description(), "test streaming agent");
        assert!(base_agent.memory().is_none());
        assert!(base_agent.stream);
    }
}
