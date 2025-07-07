use super::executor::AgentExecutor;
use async_trait::async_trait;
use autoagents_llm::{LLMProvider, ToolT};
use std::sync::Arc;

/// Base agent type that uses an executor strategy
#[derive(Clone)]
pub struct BaseAgent<E: AgentExecutor> {
    pub name: String,
    pub prompt: String,
    pub executor: E,
    pub tools: Vec<Arc<Box<dyn ToolT>>>,
    pub(crate) llm: Arc<dyn LLMProvider>,
}

impl<E: AgentExecutor> BaseAgent<E> {
    pub fn new(
        name: String,
        prompt: String,
        executor: E,
        tools: Vec<Arc<Box<dyn ToolT>>>,
        llm: Arc<dyn LLMProvider>,
    ) -> Self {
        Self {
            name,
            prompt,
            executor,
            tools,
            llm,
        }
    }

    /// Add a tool to this agent
    pub fn with_tool(mut self, tool: Arc<Box<dyn ToolT>>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Add multiple tools to this agent
    pub fn with_tools(mut self, tools: Vec<Arc<Box<dyn ToolT>>>) -> Self {
        self.tools.extend(tools);
        self
    }
}

#[async_trait]
pub trait AgentDeriveT: Send + Sync {
    fn prompt(&self) -> &'static str;
    fn name(&self) -> &'static str;
    fn tools(&self) -> Vec<Box<dyn ToolT>>;
}
