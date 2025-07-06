use super::{error::AgentBuildError, AgentExecutor, IntoRunnable, RunnableAgent};
use crate::memory::MemoryProvider;
use async_trait::async_trait;
use autoagents_llm::{LLMProvider, ToolT};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Core trait that defines agent metadata and behavior
/// This trait is implemented via the #[agent] macro
#[async_trait]
pub trait AgentDeriveT: Send + Sync + 'static + AgentExecutor {
    /// The output type this agent produces
    type Output: Serialize + DeserializeOwned + Send + Sync + Into<Value>;

    /// Get the agent's description
    fn description(&self) -> &'static str;

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
        AgentConfig {
            name: self.name().into(),
            description: self.description().into(),
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
    pub fn build(self) -> Result<Arc<dyn RunnableAgent>, AgentBuildError> {
        let llm = self.llm.ok_or(AgentBuildError::BuildFailure(
            "LLM provider is required".to_string(),
        ))?;
        Ok(BaseAgent::new(self.inner, llm, self.memory).into_runnable())
    }

    /// Build the BaseAgent with memory (for compatibility)
    pub fn build_with_memory(
        self,
        memory: Box<dyn MemoryProvider>,
    ) -> Result<Arc<dyn RunnableAgent>, AgentBuildError> {
        let llm = self.llm.ok_or(AgentBuildError::BuildFailure(
            "LLM provider is required".to_string(),
        ))?;
        Ok(BaseAgent::new(self.inner, llm, Some(memory)).into_runnable())
    }
}
