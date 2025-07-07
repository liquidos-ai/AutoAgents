use super::base::BaseAgent;
use super::executor::AgentExecutor;
use super::result::AgentRunResult;
use crate::error::Error;
use crate::protocol::Event;
use crate::session::Task;
use crate::tool::ToolCallResult;
use async_trait::async_trait;
use autoagents_llm::chat::ChatMessage;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use uuid::Uuid;

#[derive(Debug, Default, Clone)]
pub struct History {
    pub messages: Vec<ChatMessage>,
    pub tool_calls: Vec<ToolCallResult>,
    pub tasks: Vec<Task>,
}

#[derive(Default)]
pub struct AgentState {
    history: History,
}

impl AgentState {
    pub(crate) fn get_history(&self) -> History {
        self.history.clone()
    }

    pub(crate) fn record_conversation(&mut self, message: ChatMessage) {
        self.history.messages.push(message);
    }

    pub(crate) fn record_tool_call(&mut self, tool_call: ToolCallResult) {
        self.history.tool_calls.push(tool_call);
    }
}

/// RunnableAgent trait with concrete Output and Error types for easier trait object usage
#[async_trait]
pub trait RunnableAgent: Send + Sync + 'static {
    fn name(&self) -> &str;
    fn prompt(&self) -> &str;
    fn id(&self) -> Uuid;

    async fn run(
        self: Arc<Self>,
        task: Task,
        tx_event: mpsc::Sender<Event>,
    ) -> Result<AgentRunResult, Error>;

    fn spawn_task(
        self: Arc<Self>,
        task: Task,
        tx_event: mpsc::Sender<Event>,
    ) -> JoinHandle<Result<AgentRunResult, Error>> {
        tokio::spawn(async move { self.run(task, tx_event).await })
    }
}

/// Concrete RunnableAgent implementation wrapping BaseAgent<E>
pub struct RunnableAgentImpl<E>
where
    E: AgentExecutor,
{
    agent: BaseAgent<E>,
    state: Arc<Mutex<AgentState>>,
    id: Uuid,
}

impl<E> RunnableAgentImpl<E>
where
    E: AgentExecutor,
{
    pub fn new(agent: BaseAgent<E>) -> Self {
        Self {
            agent,
            state: Arc::new(Mutex::new(AgentState::default())),
            id: Uuid::new_v4(),
        }
    }
}

#[async_trait]
impl<E> RunnableAgent for RunnableAgentImpl<E>
where
    E: AgentExecutor,
    E::Output: Into<Value> + Send + Sync,
    E::Error: std::error::Error + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.agent.name
    }

    fn prompt(&self) -> &str {
        &self.agent.prompt
    }

    fn id(&self) -> Uuid {
        self.id
    }

    async fn run(
        self: Arc<Self>,
        task: Task,
        tx_event: mpsc::Sender<Event>,
    ) -> Result<AgentRunResult, Error> {
        let llm = self.agent.llm.clone();
        let result = self
            .agent
            .executor
            .execute(llm, task.clone(), self.state.clone())
            .await;

        // Convert result to unified types (Value and boxed error)
        let task_result = match &result {
            Ok(val) => crate::protocol::TaskResult::Value(
                serde_json::to_value(val).unwrap_or(serde_json::Value::Null),
            ),
            Err(err) => crate::protocol::TaskResult::Failure(err.to_string()),
        };

        let _ = tx_event
            .send(Event::TaskComplete {
                sub_id: task.submission_id,
                result: task_result,
            })
            .await;

        match result {
            Ok(val) => Ok(AgentRunResult::success(val.into())),
            Err(e) => Ok(AgentRunResult::failure(e.to_string())),
        }
    }
}

/// Builder for runnable agents
pub struct RunnableAgentBuilder {}

impl Default for RunnableAgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RunnableAgentBuilder {
    pub fn new() -> Self {
        Self {}
    }

    pub fn build<E>(self, agent: BaseAgent<E>) -> Arc<dyn RunnableAgent>
    where
        E: AgentExecutor,
        E::Output: Into<Value> + Send + Sync,
        E::Error: std::error::Error + Send + Sync + 'static,
    {
        Arc::new(RunnableAgentImpl::new(agent))
    }
}
