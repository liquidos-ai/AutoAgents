use super::base::{AgentDeriveT, BaseAgent};
use super::error::RunnableAgentError;
use crate::agent::context::Context;
use crate::agent::memory::MemoryProvider;
use crate::agent::task::Task;
use crate::error::Error;
use crate::protocol::{Event, TaskResult};
use crate::tool::ToolCallResult;
use ractor::{async_trait, Actor, ActorProcessingErr, ActorRef};
use serde_json::Value;
use std::fmt::Debug;
use std::sync::Arc;
use tokio::sync::mpsc::Sender;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use uuid::Uuid;

/// State tracking for agent execution
#[derive(Debug, Default, Clone)]
pub struct AgentState {
    /// Tool calls made during execution
    pub tool_calls: Vec<ToolCallResult>,
    /// Tasks that have been executed
    pub task_history: Vec<Task>,
    pub tx: Option<Sender<Event>>,
}

impl AgentState {
    pub fn new(tx: Sender<Event>) -> Self {
        Self {
            tool_calls: vec![],
            task_history: vec![],
            tx: Some(tx),
        }
    }

    pub fn record_tool_call(&mut self, tool_call: ToolCallResult) {
        self.tool_calls.push(tool_call);
    }

    pub fn record_task(&mut self, task: Task) {
        self.task_history.push(task);
    }

    pub fn set_tx(&mut self, tx: Sender<Event>) -> &mut Self {
        self.tx = Some(tx);
        self
    }
}

/// Trait for agents that can be executed within the system
#[async_trait]
pub trait RunnableAgent: Send + Sync + 'static + Debug {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn id(&self) -> Uuid;

    async fn run(self: Arc<Self>, task: Task, tx_event: Sender<Event>) -> Result<(), Error>;

    fn memory(&self) -> Option<Arc<RwLock<Box<dyn MemoryProvider>>>>;

    fn spawn_task(
        self: Arc<Self>,
        task: Task,
        tx_event: Sender<Event>,
    ) -> JoinHandle<Result<(), Error>> {
        tokio::spawn(async move { self.run(task.clone(), tx_event).await })
    }
}

/// Enhanced BaseAgent that includes runtime state for execution
pub type RunnableAgentImpl<T> = BaseAgent<T>;

#[async_trait]
impl<T> RunnableAgent for BaseAgent<T>
where
    T: AgentDeriveT,
{
    fn name(&self) -> &'static str {
        BaseAgent::name(self)
    }

    fn description(&self) -> &'static str {
        BaseAgent::description(self)
    }

    fn id(&self) -> Uuid {
        self.id
    }

    async fn run(self: Arc<Self>, task: Task, tx_event: Sender<Event>) -> Result<(), Error> {
        let submission_id = task.submission_id;

        let context = Context::new(self.llm(), tx_event.clone())
            .with_memory(self.memory())
            .with_tools(self.tools())
            .with_config(self.agent_config())
            .with_stream(self.stream);
        // Execute the agent's logic using the executor
        match self.inner().execute(&task, context).await {
            Ok(output) => {
                // Convert output to Value
                let value: Value = output.into();

                // Send completion event
                let _ = tx_event
                    .send(Event::TaskComplete {
                        sub_id: submission_id,
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
                        sub_id: submission_id,
                        result: TaskResult::Failure(error_msg.clone()),
                    })
                    .await;

                Err(RunnableAgentError::ExecutorError(error_msg).into())
            }
        }
    }

    fn memory(&self) -> Option<Arc<RwLock<Box<dyn MemoryProvider>>>> {
        BaseAgent::memory(self)
    }
}

#[derive(Debug)]
pub struct AgentActor<T: AgentDeriveT>(pub Arc<RunnableAgentImpl<T>>);

impl<T: AgentDeriveT> AgentActor<T> {
    pub fn id(&self) -> Uuid {
        self.0.id
    }
}

#[async_trait]
impl<T: AgentDeriveT> Actor for AgentActor<T>
where
    T: Send + Sync + 'static,
{
    type Msg = Task;
    type State = AgentState;
    type Arguments = Sender<Event>;

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        Ok(AgentState::new(args))
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        let agent = self.0.clone();
        let task = message;
        let tx = state.tx.as_ref().unwrap().clone();

        //Run agent
        let _ = agent.run(task, tx).await;

        Ok(())
    }
}

/// Extension trait for converting BaseAgent to RunnableAgent
pub trait IntoRunnable<T: AgentDeriveT> {
    fn into_runnable(self) -> Arc<RunnableAgentImpl<T>>;
}

impl<T: AgentDeriveT> IntoRunnable<T> for BaseAgent<T> {
    fn into_runnable(self) -> Arc<RunnableAgentImpl<T>> {
        Arc::new(self)
    }
}
