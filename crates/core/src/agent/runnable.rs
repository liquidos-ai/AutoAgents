use super::base::{AgentDeriveT, BaseAgent};
use super::executor::AgentExecutor;
use super::error::RunnableAgentError;
use super::result::AgentRunResult;
use crate::error::Error;
use crate::memory::MemoryProvider;
use crate::protocol::{Event, TaskResult};
use crate::session::Task;
use crate::tool::ToolCallResult;
use futures::{Stream, StreamExt};
use std::pin::Pin;
use async_trait::async_trait;
use autoagents_llm::chat::{ChatMessage, ChatRole, MessageType};
use serde_json::Value;
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
pub trait RunnableAgent: Send + Sync + 'static {
    fn name(&self) -> String;
    fn description(&self) -> String;
    fn id(&self) -> Uuid;

    async fn run(
        self: Arc<Self>,
        task: Task,
        tx_event: mpsc::Sender<Event>,
    ) -> Result<AgentRunResult, Error>;

    fn stream(
        self: Arc<Self>,
        task: Task,
    ) -> Pin<Box<dyn Stream<Item = Result<Event, Error>> + Send>>;

    fn memory(&self) -> Option<Arc<RwLock<Box<dyn MemoryProvider>>>>;

    fn spawn_task(
        self: Arc<Self>,
        task: Task,
        tx_event: mpsc::Sender<Event>,
    ) -> JoinHandle<Result<AgentRunResult, Error>> {
        tokio::spawn(async move { self.run(task, tx_event).await })
    }
}

/// Wrapper that makes BaseAgent<T> implement RunnableAgent
pub struct RunnableAgentImpl<T: AgentDeriveT> {
    agent: BaseAgent<T>,
    state: Arc<RwLock<AgentState>>,
    id: Uuid,
}

impl<T: AgentDeriveT> RunnableAgentImpl<T> {
    pub fn new(agent: BaseAgent<T>) -> Self {
        Self {
            agent,
            state: Arc::new(RwLock::new(AgentState::new())),
            id: Uuid::new_v4(),
        }
    }

    pub fn state(&self) -> Arc<RwLock<AgentState>> {
        self.state.clone()
    }
}

#[async_trait]
impl<T> RunnableAgent for RunnableAgentImpl<T>
where
    T: AgentExecutor + AgentDeriveT,
    T::Error: Into<Error>,
{
    fn name(&self) -> String {
        self.agent.name()
    }

    fn description(&self) -> String {
        self.agent.description()
    }

    fn id(&self) -> Uuid {
        self.id
    }

    fn memory(&self) -> Option<Arc<RwLock<Box<dyn MemoryProvider>>>> {
        self.agent.memory()
    }

    async fn run(
        self: Arc<Self>,
        task: Task,
        tx_event: mpsc::Sender<Event>,
    ) -> Result<AgentRunResult, Error> {
        // Record the task in state
        {
            let mut state = self.state.write().await;
            state.record_task(task.clone());
        }

        // Store the task in memory if available
        if let Some(memory) = self.agent.memory() {
            let mut mem = memory.write().await;
            let chat_msg = ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: task.prompt.clone(),
            };
            let _ = mem.remember(&chat_msg).await;
        }

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

                // Create task result
                let task_result = TaskResult::Value(value.clone());

                // Send completion event
                let _ = tx_event
                    .send(Event::TaskComplete {
                        sub_id: task.submission_id,
                        result: task_result,
                    })
                    .await
                    .map_err(RunnableAgentError::event_send_error)?;

                Ok(AgentRunResult::success(value))
            }
            Err(e) => {
                // Send error event
                let error_msg = e.to_string();
                let _ = tx_event
                    .send(Event::Error {
                        sub_id: task.submission_id,
                        error: error_msg.clone(),
                    })
                    .await;

                Err(RunnableAgentError::ExecutorError(error_msg).into())
            }
        }
    }

    fn stream(
        self: Arc<Self>,
        task: Task,
    ) -> Pin<Box<dyn Stream<Item = Result<Event, Error>> + Send>> {
        let stream = self
            .agent
            .inner()
            .stream(
                self.agent.llm(),
                self.agent.memory(),
                self.agent.tools(),
                &self.agent.agent_config(),
                task,
                self.state.clone(),
            )
            .map(|result| result.map_err(Into::into));

        Box::pin(stream)
    }
}

/// Extension trait for converting BaseAgent to RunnableAgent
pub trait IntoRunnable<T: AgentDeriveT> {
    fn into_runnable(self) -> Arc<dyn RunnableAgent>;
}

impl<T: AgentDeriveT> IntoRunnable<T> for BaseAgent<T>
where
    T::Error: Into<Error>,
{
    fn into_runnable(self) -> Arc<dyn RunnableAgent> {
        Arc::new(RunnableAgentImpl::new(self))
    }
}
