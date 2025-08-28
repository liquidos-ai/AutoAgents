use super::base::{AgentDeriveT, BaseAgent};
use super::error::RunnableAgentError;
use crate::agent::context::Context;
use crate::agent::memory::MemoryProvider;
use crate::agent::state::AgentState;
use crate::agent::task::Task;
use crate::error::Error;
use crate::protocol::{Event, TaskResult};
use async_trait::async_trait;
use futures::Stream;
#[cfg(not(target_arch = "wasm32"))]
use ractor::{Actor, ActorProcessingErr, ActorRef};
use serde_json::Value;
use std::fmt::Debug;
use std::sync::Arc;
use uuid::Uuid;

#[cfg(target_arch = "wasm32")]
pub use futures::channel::mpsc::Sender;
#[cfg(target_arch = "wasm32")]
pub use futures::lock::Mutex;
#[cfg(target_arch = "wasm32")]
use futures::SinkExt;
#[cfg(not(target_arch = "wasm32"))]
pub use tokio::sync::{mpsc::Sender, Mutex};

/// Trait for agents that can be executed within the system
#[async_trait]
pub trait RunnableAgent: Send + Sync + 'static + Debug {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn id(&self) -> Uuid;

    fn tx(&self) -> Sender<Event>;

    async fn run(self: Arc<Self>, task: Task) -> Result<TaskResult, Error>;

    async fn run_stream(
        self: Arc<Self>,
        task: Task,
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<TaskResult, Error>> + Send>>, Error>;

    fn memory(&self) -> Option<Arc<Mutex<Box<dyn MemoryProvider>>>>;
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

    fn tx(&self) -> Sender<Event> {
        self.tx.clone()
    }

    async fn run(self: Arc<Self>, task: Task) -> Result<TaskResult, Error> {
        let submission_id = task.submission_id;
        let mut tx_event = self.tx();

        let context = Context::new(self.llm(), tx_event.clone())
            .with_memory(self.memory())
            .with_tools(self.tools())
            .with_config(self.agent_config())
            .with_stream(self.stream());

        // Execute the agent's logic using the executor
        match self.inner().execute(&task, Arc::new(context)).await {
            Ok(output) => {
                // Convert output to Value
                let value: Value = output.into();

                // Send completion event
                let _ = tx_event
                    .send(Event::TaskComplete {
                        sub_id: submission_id,
                        result: TaskResult::Value(value.clone()),
                    })
                    .await
                    .map_err(RunnableAgentError::event_send_error)?;

                Ok(TaskResult::Value(value))
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

    async fn run_stream(
        self: Arc<Self>,
        task: Task,
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<TaskResult, Error>> + Send>>, Error>
    {
        let submission_id = task.submission_id;
        let mut tx_event = self.tx();

        let context = Context::new(self.llm(), tx_event.clone())
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
                        Ok(output) => {
                            // Convert output to Value
                            let value: Value = output.into();
                            Ok(TaskResult::Value(value))
                        }
                        Err(e) => {
                            // Handle error
                            let error_msg = e.to_string();
                            Err(RunnableAgentError::ExecutorError(error_msg).into())
                        }
                    }
                });

                Ok(Box::pin(transformed_stream))
            }
            Err(e) => {
                // Send error event for stream creation failure
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

    fn memory(&self) -> Option<Arc<Mutex<Box<dyn MemoryProvider>>>> {
        BaseAgent::memory(self)
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub struct AgentActor<T: AgentDeriveT>(pub Arc<RunnableAgentImpl<T>>);

#[cfg(not(target_arch = "wasm32"))]
impl<T: AgentDeriveT> AgentActor<T> {
    pub fn id(&self) -> Uuid {
        self.0.id
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
impl<T: AgentDeriveT> Actor for AgentActor<T>
where
    T: Send + Sync + 'static,
{
    type Msg = Task;
    type State = AgentState;
    type Arguments = ();

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        _args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        Ok(AgentState::new())
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        _state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        let agent = self.0.clone();
        let task = message;

        //Run agent
        if agent.stream() {
            let _ = agent.run_stream(task).await?;
            Ok(())
        } else {
            let _ = agent.run(task).await?;
            Ok(())
        }
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
