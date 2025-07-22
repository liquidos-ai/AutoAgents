use super::base::{AgentDeriveT, BaseAgent};
use super::error::RunnableAgentError;
use crate::error::Error;
use crate::memory::MemoryProvider;
use crate::protocol::{Event, TaskResult};
use crate::runtime::Task;
use crate::tool::ToolCallResult;
use async_trait::async_trait;
use serde_json::Value;
use std::fmt::Debug;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio::task::JoinHandle;
use uuid::Uuid;

/// State tracking for agent execution
#[derive(Debug, Default, Clone)]
pub struct AgentState {
    /// Tool calls made during execution
    pub tool_calls: Vec<ToolCallResult>,
    /// Tasks that have been executed
    pub task_history: Vec<Task>,
}

impl AgentState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_tool_call(&mut self, tool_call: ToolCallResult) {
        self.tool_calls.push(tool_call);
    }

    pub fn record_task(&mut self, task: Task) {
        self.task_history.push(task);
    }
}

/// Trait for agents that can be executed within the system
#[async_trait]
pub trait RunnableAgent: Send + Sync + 'static + Debug {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn id(&self) -> Uuid;

    async fn run(self: Arc<Self>, task: Task, tx_event: mpsc::Sender<Event>) -> Result<(), Error>;

    fn memory(&self) -> Option<Arc<RwLock<Box<dyn MemoryProvider>>>>;

    fn spawn_task(
        self: Arc<Self>,
        task: Task,
        tx_event: mpsc::Sender<Event>,
    ) -> JoinHandle<Result<(), Error>> {
        tokio::spawn(async move { self.run(task, tx_event).await })
    }
}

/// Wrapper that makes BaseAgent<T> implement RunnableAgent
#[derive(Debug)]
pub struct RunnableAgentImpl<T: AgentDeriveT> {
    agent: BaseAgent<T>,
    state: Arc<RwLock<AgentState>>,
}

impl<T: AgentDeriveT> RunnableAgentImpl<T> {
    pub fn new(agent: BaseAgent<T>) -> Self {
        Self {
            agent,
            state: Arc::new(RwLock::new(AgentState::new())),
        }
    }

    #[allow(dead_code)]
    pub fn state(&self) -> Arc<RwLock<AgentState>> {
        self.state.clone()
    }
}

#[async_trait]
impl<T> RunnableAgent for RunnableAgentImpl<T>
where
    T: AgentDeriveT,
{
    fn name(&self) -> &'static str {
        self.agent.name()
    }

    fn description(&self) -> &'static str {
        self.agent.description()
    }

    fn id(&self) -> Uuid {
        self.agent.id
    }

    fn memory(&self) -> Option<Arc<RwLock<Box<dyn MemoryProvider>>>> {
        self.agent.memory()
    }

    async fn run(self: Arc<Self>, task: Task, tx_event: mpsc::Sender<Event>) -> Result<(), Error> {
        // Execute the agent's logic using the executor
        match self
            .agent
            .inner()
            .execute(
                self.agent.llm(),
                self.agent.memory(),
                self.agent.tools(),
                &self.agent.agent_config(),
                task.clone(),
                self.state.clone(),
                tx_event.clone(),
            )
            .await
        {
            Ok(output) => {
                // Convert output to Value
                let value: Value = output.into();

                // Send completion event
                let _ = tx_event
                    .send(Event::TaskComplete {
                        sub_id: task.submission_id,
                        result: TaskResult::Value(value),
                    })
                    .await
                    .map_err(RunnableAgentError::event_send_error)?;

                Ok(())
            }
            Err(e) => {
                // Send error event
                let error_msg = e.to_string();
                let _ = tx_event
                    .send(Event::TaskComplete {
                        sub_id: task.submission_id,
                        result: TaskResult::Failure(error_msg.clone()),
                    })
                    .await;

                Err(RunnableAgentError::ExecutorError(error_msg).into())
            }
        }
    }
}

/// Extension trait for converting BaseAgent to RunnableAgent
pub trait IntoRunnable<T: AgentDeriveT> {
    fn into_runnable(self) -> Arc<dyn RunnableAgent>;
}

impl<T: AgentDeriveT> IntoRunnable<T> for BaseAgent<T> {
    fn into_runnable(self) -> Arc<dyn RunnableAgent> {
        Arc::new(RunnableAgentImpl::new(self))
    }
}
