#[cfg(not(target_arch = "wasm32"))]
use crate::actor::Topic;
use crate::agent::base::AgentType;
use crate::agent::error::{AgentBuildError, RunnableAgentError};
use crate::agent::hooks::AgentHooks;
use crate::agent::state::AgentState;
use crate::agent::task::Task;
use crate::agent::{AgentBuilder, AgentDeriveT, AgentExecutor, BaseAgent, HookOutcome};
use crate::channel::Sender;
use crate::error::Error;
use crate::protocol::Event;
#[cfg(not(target_arch = "wasm32"))]
use crate::runtime::TypedRuntime;
use async_trait::async_trait;
#[cfg(target_arch = "wasm32")]
use futures::SinkExt;
use futures::Stream;
#[cfg(not(target_arch = "wasm32"))]
use ractor::Actor;
#[cfg(not(target_arch = "wasm32"))]
use ractor::{ActorProcessingErr, ActorRef};
use serde_json::Value;
use std::fmt::Debug;
use std::sync::Arc;

pub struct ActorAgent {}

impl AgentType for ActorAgent {
    fn type_name() -> &'static str {
        "protocol_agent"
    }
}

/// Handle for an agent that includes both the agent and its actor reference
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
pub struct ActorAgentHandle<T: AgentDeriveT + AgentExecutor + AgentHooks + Send + Sync> {
    pub agent: Arc<BaseAgent<T, ActorAgent>>,
    pub actor_ref: ActorRef<Task>,
}

#[cfg(not(target_arch = "wasm32"))]
impl<T: AgentDeriveT + AgentExecutor + AgentHooks> ActorAgentHandle<T> {
    /// Get the actor reference for direct messaging
    pub fn addr(&self) -> ActorRef<Task> {
        self.actor_ref.clone()
    }

    /// Get the agent reference
    pub fn agent(&self) -> Arc<BaseAgent<T, ActorAgent>> {
        self.agent.clone()
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<T: AgentDeriveT + AgentExecutor + AgentHooks> Debug for ActorAgentHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentHandle")
            .field("agent", &self.agent)
            .finish()
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub struct AgentActor<T: AgentDeriveT + AgentExecutor + AgentHooks>(
    pub Arc<BaseAgent<T, ActorAgent>>,
);

#[cfg(not(target_arch = "wasm32"))]
impl<T: AgentDeriveT + AgentExecutor + AgentHooks> AgentActor<T> {}

#[cfg(not(target_arch = "wasm32"))]
impl<T: AgentDeriveT + AgentExecutor + AgentHooks> AgentBuilder<T, ActorAgent>
where
    T: Send + Sync + 'static,
    serde_json::Value: From<<T as AgentExecutor>::Output>,
    <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output>,
{
    /// Build the BaseAgent and return a wrapper that includes the actor reference
    pub async fn build(self) -> Result<ActorAgentHandle<T>, Error> {
        let llm = self.llm.ok_or(AgentBuildError::BuildFailure(
            "LLM provider is required".to_string(),
        ))?;
        let runtime = self.runtime.ok_or(AgentBuildError::BuildFailure(
            "Runtime should be defined".into(),
        ))?;
        let tx = runtime.tx();

        let agent: Arc<BaseAgent<T, ActorAgent>> = Arc::new(
            BaseAgent::<T, ActorAgent>::new(self.inner, llm, self.memory, tx, self.stream).await?,
        );

        // Create agent actor
        let agent_actor = AgentActor(agent.clone());
        let actor_ref = Actor::spawn(Some(agent_actor.0.name().into()), agent_actor, ())
            .await
            .map_err(AgentBuildError::SpawnError)?
            .0;

        // Subscribe to topics
        for topic in self.subscribed_topics {
            runtime.subscribe(&topic, actor_ref.clone()).await?;
        }

        Ok(ActorAgentHandle { agent, actor_ref })
    }

    pub fn subscribe(mut self, topic: Topic<Task>) -> Self {
        self.subscribed_topics.push(topic);
        self
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<T: AgentDeriveT + AgentExecutor + AgentHooks> BaseAgent<T, ActorAgent> {
    pub fn tx(&self) -> Result<Sender<Event>, RunnableAgentError> {
        self.tx.clone().ok_or(RunnableAgentError::EmptyTx)
    }

    pub async fn run(
        self: Arc<Self>,
        task: Task,
    ) -> Result<<T as AgentDeriveT>::Output, RunnableAgentError>
    where
        Value: From<<T as AgentExecutor>::Output>,
        <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output>,
    {
        let submission_id = task.submission_id;
        let tx = self.tx().map_err(|_| RunnableAgentError::EmptyTx)?;

        let context = self.create_context();

        //Run Hook
        let hook_outcome = self.inner.on_run_start(&task, &context).await;
        match hook_outcome {
            HookOutcome::Abort => return Err(RunnableAgentError::Abort),
            HookOutcome::Continue => {}
        }

        // Execute the agent's logic using the executor
        match self.inner().execute(&task, context.clone()).await {
            Ok(output) => {
                let value: Value = output.clone().into();

                #[cfg(not(target_arch = "wasm32"))]
                tx.send(Event::TaskComplete {
                    sub_id: submission_id,
                    actor_id: self.id,
                    actor_name: self.name().to_string(),
                    result: serde_json::to_string_pretty(&value)
                        .map_err(|e| RunnableAgentError::ExecutorError(e.to_string()))?,
                })
                .await
                .map_err(|e| RunnableAgentError::ExecutorError(e.to_string()))?;

                //Extract Agent output into the desired type
                let agent_out: <T as AgentDeriveT>::Output = output.into();

                //Run On complete Hook
                self.inner
                    .on_run_complete(&task, &agent_out, &context)
                    .await;

                Ok(agent_out)
            }
            Err(e) => {
                #[cfg(not(target_arch = "wasm32"))]
                tx.send(Event::TaskError {
                    sub_id: submission_id,
                    actor_id: self.id,
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
        std::pin::Pin<
            Box<dyn Stream<Item = Result<<T as AgentDeriveT>::Output, RunnableAgentError>> + Send>,
        >,
        RunnableAgentError,
    >
    where
        <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output>,
    {
        // let submission_id = task.submission_id;
        let context = self.create_context();

        // Execute the agent's streaming logic using the executor
        match self.inner().execute_stream(&task, context).await {
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

#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
impl<T: AgentDeriveT + AgentExecutor + AgentHooks> Actor for AgentActor<T>
where
    T: Send + Sync + 'static,
    serde_json::Value: From<<T as AgentExecutor>::Output>,
    <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output>,
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

    async fn post_stop(
        &self,
        _myself: ActorRef<Self::Msg>,
        _state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        //Run Hook
        self.0.inner().on_agent_shutdown().await;
        Ok(())
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
