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
    pub system_prompt: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_turns: 10,
            system_prompt: None,
            temperature: None,
            max_tokens: None,
        }
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
    fn config(&self) -> &ExecutorConfig;

    /// Process a single turn of execution
    #[allow(clippy::too_many_arguments)]
    async fn process_turn(
        &self,
        llm: Arc<dyn LLMProvider>,
        memory: Option<Arc<RwLock<Box<dyn MemoryProvider>>>>,
        tools: &[Arc<Box<dyn ToolT>>],
        agent_config: &AgentConfig,
        task: &Task,
        state: Arc<RwLock<AgentState>>,
        turn_number: usize,
        tx_event: mpsc::Sender<Event>,
    ) -> Result<TurnResult<Self::Output>, Self::Error>;

    /// Execute the agent with the given task
    #[allow(clippy::too_many_arguments)]
    async fn execute(
        &self,
        llm: Arc<dyn LLMProvider>,
        memory: Option<Arc<RwLock<Box<dyn MemoryProvider>>>>,
        tools: &Vec<Arc<Box<dyn ToolT>>>,
        agent_config: &AgentConfig,
        task: Task,
        state: Arc<RwLock<AgentState>>,
        tx_event: mpsc::Sender<Event>,
    ) -> Result<Self::Output, Self::Error> {
        let max_turns = self.config().max_turns;

        for turn in 0..max_turns {
            match self
                .process_turn(
                    llm.clone(),
                    memory.clone(),
                    tools,
                    agent_config,
                    &task,
                    state.clone(),
                    turn,
                    tx_event.clone(),
                )
                .await?
            {
                TurnResult::Complete(result) => {
                    return Ok(result);
                }
                TurnResult::Continue(_) => {
                    continue;
                }
                TurnResult::Error(msg) => {
                    eprintln!("Turn {} error: {}", turn, msg);
                    continue;
                }
                TurnResult::Fatal(_error) => {
                    // Fatal errors should be of type Self::Error
                    return Err(self.max_turns_error());
                }
            }
        }

        // Reached max turns without completion
        eprintln!("Reached maximum turns ({}) without completion", max_turns);
        Err(self.max_turns_error())
    }

    /// Create an error for when max turns is exceeded
    fn max_turns_error(&self) -> Self::Error;
}
