use crate::protocol::Event;
use crate::session::{Session, Task};
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use uuid::Uuid;

use super::runner::AgentRunner;

/// A trait for agents that can be run by the Environment without exposing executor details
#[async_trait]
pub trait RunnableAgent: Send + Sync + 'static {
    /// Get the agent's name
    fn name(&self) -> &str;

    /// Get the agent's description
    fn description(&self) -> &str;

    /// Get the agent's unique ID
    fn id(&self) -> Uuid;

    /// Run the agent with the provided session and event channel
    /// The implementation is responsible for handling the specific LLM type
    async fn run(
        self: Arc<Self>,
        session: Session,
        task: Task,
        tx_event: mpsc::Sender<Event>,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>>;

    /// Create a task handle for running this agent
    fn spawn_task(
        self: Arc<Self>,
        session: Session,
        task: Task,
        tx_event: mpsc::Sender<Event>,
    ) -> JoinHandle<Result<Value, Box<dyn std::error::Error + Send + Sync>>> {
        tokio::spawn(async move { self.run(session, task, tx_event).await })
    }
}

/// Wrapper that makes a BaseAgent<E> implement RunnableAgent for a specific LLM type
pub struct RunnableAgentImpl<E, L>
where
    E: crate::agent::executor::AgentExecutor + Clone + Send + Sync + 'static,
    L: autoagents_llm::llm::LLM + Clone + 'static,
{
    agent: crate::agent::base::BaseAgent<E>,
    llm: L,
    id: Uuid,
}

impl<E, L> RunnableAgentImpl<E, L>
where
    E: crate::agent::executor::AgentExecutor + Clone + Send + Sync + 'static,
    L: autoagents_llm::llm::LLM + Clone + 'static,
{
    pub fn new(agent: crate::agent::base::BaseAgent<E>, llm: L) -> Self {
        Self {
            agent,
            llm,
            id: Uuid::new_v4(),
        }
    }
}

#[async_trait]
impl<E, L> RunnableAgent for RunnableAgentImpl<E, L>
where
    E: crate::agent::executor::AgentExecutor + Clone + Send + Sync + 'static,
    E::Output: Into<Value> + Send,
    E::Error: std::error::Error + Send + Sync + 'static,
    L: autoagents_llm::llm::LLM + Clone + 'static,
{
    fn name(&self) -> &str {
        &self.agent.name
    }

    fn description(&self) -> &str {
        &self.agent.description
    }

    fn id(&self) -> Uuid {
        self.id
    }

    async fn run(
        self: Arc<Self>,
        session: Session,
        task: Task,
        tx_event: mpsc::Sender<Event>,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        // Execute the agent
        let runner = AgentRunner::new(self.agent.clone(), tx_event);
        runner
            .run(self.llm.clone(), session, task)
            .await
            .map(Into::into)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    }
}

/// Builder for creating runnable agents
pub struct RunnableAgentBuilder<L: autoagents_llm::llm::LLM + Clone + 'static> {
    llm: L,
}

impl<L: autoagents_llm::llm::LLM + Clone + 'static> RunnableAgentBuilder<L> {
    pub fn new(llm: L) -> Self {
        Self { llm }
    }

    /// Build a runnable agent from a BaseAgent
    pub fn build<E>(self, agent: crate::agent::base::BaseAgent<E>) -> Arc<dyn RunnableAgent>
    where
        E: crate::agent::executor::AgentExecutor + Clone + Send + Sync + 'static,
        E::Output: Into<Value> + Send,
        E::Error: std::error::Error + Send + Sync + 'static,
    {
        Arc::new(RunnableAgentImpl::new(agent, self.llm))
    }
}
