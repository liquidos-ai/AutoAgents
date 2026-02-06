use crate::agent::base::AgentType;
use crate::agent::error::{AgentBuildError, RunnableAgentError};
use crate::agent::task::Task;
use crate::agent::{AgentBuilder, AgentDeriveT, AgentExecutor, AgentHooks, BaseAgent, HookOutcome};
use crate::error::Error;
use autoagents_protocol::Event;
use futures::Stream;

use crate::agent::constants::DEFAULT_CHANNEL_BUFFER;

use crate::channel::{Receiver, Sender, channel};

#[cfg(not(target_arch = "wasm32"))]
use crate::event_fanout::EventFanout;
use crate::utils::{BoxEventStream, receiver_into_stream};
#[cfg(not(target_arch = "wasm32"))]
use futures_util::stream;

/// Marker type for direct (non-actor) agents.
///
/// Direct agents execute immediately within the caller's task without
/// requiring a runtime or event wiring. Use this for simple one-shot
/// invocations and unit tests.
pub struct DirectAgent {}

impl AgentType for DirectAgent {
    fn type_name() -> &'static str {
        "direct_agent"
    }
}

/// Handle for a direct agent containing the agent instance and an event stream
/// receiver. Use `agent.run(...)` for one-shot calls or `agent.run_stream(...)`
/// to receive streaming outputs.
pub struct DirectAgentHandle<T: AgentDeriveT + AgentExecutor + AgentHooks + Send + Sync> {
    pub agent: BaseAgent<T, DirectAgent>,
    pub rx: BoxEventStream<Event>,
    #[cfg(not(target_arch = "wasm32"))]
    fanout: Option<EventFanout>,
}

impl<T: AgentDeriveT + AgentExecutor + AgentHooks> DirectAgentHandle<T> {
    pub fn new(agent: BaseAgent<T, DirectAgent>, rx: BoxEventStream<Event>) -> Self {
        Self {
            agent,
            rx,
            #[cfg(not(target_arch = "wasm32"))]
            fanout: None,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn subscribe_events(&mut self) -> BoxEventStream<Event> {
        if let Some(fanout) = &self.fanout {
            return fanout.subscribe();
        }

        let stream = std::mem::replace(&mut self.rx, Box::pin(stream::empty::<Event>()));
        let fanout = EventFanout::new(stream, DEFAULT_CHANNEL_BUFFER);
        self.rx = fanout.subscribe();
        let stream = fanout.subscribe();
        self.fanout = Some(fanout);
        stream
    }
}

impl<T: AgentDeriveT + AgentExecutor + AgentHooks> AgentBuilder<T, DirectAgent> {
    /// Build the BaseAgent and return a wrapper
    #[allow(clippy::result_large_err)]
    pub async fn build(self) -> Result<DirectAgentHandle<T>, Error> {
        let llm = self.llm.ok_or(AgentBuildError::BuildFailure(
            "LLM provider is required".to_string(),
        ))?;
        let (tx, rx): (Sender<Event>, Receiver<Event>) = channel(DEFAULT_CHANNEL_BUFFER);
        let agent: BaseAgent<T, DirectAgent> =
            BaseAgent::<T, DirectAgent>::new(self.inner, llm, self.memory, tx, self.stream).await?;
        let stream = receiver_into_stream(rx);
        Ok(DirectAgentHandle::new(agent, stream))
    }
}

impl<T: AgentDeriveT + AgentExecutor + AgentHooks> BaseAgent<T, DirectAgent> {
    /// Execute the agent for a single task and return the final agent output.
    pub async fn run(&self, task: Task) -> Result<<T as AgentDeriveT>::Output, RunnableAgentError>
    where
        <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output>,
    {
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
                let output: <T as AgentExecutor>::Output = output;

                //Extract Agent output into the desired type
                let agent_out: <T as AgentDeriveT>::Output = output.into();

                //Run On complete Hook
                self.inner
                    .on_run_complete(&task, &agent_out, &context)
                    .await;
                Ok(agent_out)
            }
            Err(e) => {
                // Send error event
                Err(RunnableAgentError::ExecutorError(e.to_string()))
            }
        }
    }

    /// Execute the agent with streaming enabled and receive a stream of
    /// partial outputs which culminate in a final chunk with `done=true`.
    pub async fn run_stream(
        &self,
        task: Task,
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<<T as AgentDeriveT>::Output, Error>> + Send>>,
        RunnableAgentError,
    >
    where
        <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output>,
    {
        let context = self.create_context();

        //Run Hook
        let hook_outcome = self.inner.on_run_start(&task, &context).await;
        match hook_outcome {
            HookOutcome::Abort => return Err(RunnableAgentError::Abort),
            HookOutcome::Continue => {}
        }

        // Execute the agent's streaming logic using the executor
        match self.inner().execute_stream(&task, context.clone()).await {
            Ok(stream) => {
                use futures::StreamExt;
                // Convert the stream output
                let transformed_stream = stream.map(move |result| match result {
                    Ok(output) => Ok(output.into()),
                    Err(e) => {
                        let error_msg = e.to_string();
                        Err(RunnableAgentError::ExecutorError(error_msg).into())
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
