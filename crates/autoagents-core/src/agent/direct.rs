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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::hooks::HookOutcome;
    use crate::agent::output::AgentOutputT;
    use crate::agent::task::Task;
    use crate::agent::{Context, ExecutorConfig};
    use crate::tests::{ConfigurableLLMProvider, MockAgentImpl, TestAgentOutput, TestError};
    use crate::tool::ToolT;
    use async_trait::async_trait;
    use futures::StreamExt;
    use serde_json::Value;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    #[tokio::test]
    async fn test_direct_agent_build_requires_llm() {
        let mock_agent = MockAgentImpl::new("direct", "direct agent");
        let err = match AgentBuilder::<_, DirectAgent>::new(mock_agent)
            .build()
            .await
        {
            Ok(_) => panic!("expected missing llm error"),
            Err(err) => err,
        };

        assert!(matches!(err, crate::error::Error::AgentBuildError(_)));
    }

    #[tokio::test]
    async fn test_direct_agent_run_success() {
        let mock_agent = MockAgentImpl::new("direct", "direct agent");
        let llm = Arc::new(ConfigurableLLMProvider::default());
        let handle = AgentBuilder::<_, DirectAgent>::new(mock_agent)
            .llm(llm)
            .build()
            .await
            .expect("build should succeed");

        let task = Task::new("hello");
        let result = handle.agent.run(task).await.expect("run should succeed");
        assert_eq!(result.result, "Processed: hello");
    }

    #[tokio::test]
    async fn test_direct_agent_run_executor_error() {
        let mock_agent = MockAgentImpl::new("direct", "direct agent").with_failure(true);
        let llm = Arc::new(ConfigurableLLMProvider::default());
        let handle = AgentBuilder::<_, DirectAgent>::new(mock_agent)
            .llm(llm)
            .build()
            .await
            .expect("build should succeed");

        let task = Task::new("fail");
        let err = handle.agent.run(task).await.expect_err("expected error");
        assert!(matches!(err, RunnableAgentError::ExecutorError(_)));
    }

    #[derive(Clone, Debug)]
    struct StreamAgent;

    #[async_trait]
    impl AgentDeriveT for StreamAgent {
        type Output = TestAgentOutput;

        fn description(&self) -> &'static str {
            "stream agent"
        }

        fn output_schema(&self) -> Option<Value> {
            Some(TestAgentOutput::structured_output_format())
        }

        fn name(&self) -> &'static str {
            "stream_agent"
        }

        fn tools(&self) -> Vec<Box<dyn ToolT>> {
            vec![]
        }
    }

    #[async_trait]
    impl AgentExecutor for StreamAgent {
        type Output = TestAgentOutput;
        type Error = TestError;

        fn config(&self) -> ExecutorConfig {
            ExecutorConfig::default()
        }

        async fn execute(
            &self,
            task: &Task,
            _context: Arc<Context>,
        ) -> Result<Self::Output, Self::Error> {
            Ok(TestAgentOutput {
                result: format!("Streamed: {}", task.prompt),
            })
        }
    }

    impl AgentHooks for StreamAgent {}

    #[tokio::test]
    async fn test_direct_agent_run_stream_default_executes_once() {
        let llm = Arc::new(ConfigurableLLMProvider::default());
        let handle = AgentBuilder::<_, DirectAgent>::new(StreamAgent)
            .llm(llm)
            .build()
            .await
            .expect("build should succeed");

        let task = Task::new("stream");
        let stream = handle
            .agent
            .run_stream(task)
            .await
            .expect("stream should succeed");
        let outputs: Vec<_> = stream.collect().await;
        assert_eq!(outputs.len(), 1);
        let output = outputs[0].as_ref().expect("expected Ok output");
        assert_eq!(output.result, "Streamed: stream");
    }

    #[derive(Debug)]
    struct AbortAgent {
        executed: Arc<AtomicBool>,
    }

    #[async_trait]
    impl AgentDeriveT for AbortAgent {
        type Output = TestAgentOutput;

        fn description(&self) -> &'static str {
            "abort agent"
        }

        fn output_schema(&self) -> Option<Value> {
            Some(TestAgentOutput::structured_output_format())
        }

        fn name(&self) -> &'static str {
            "abort_agent"
        }

        fn tools(&self) -> Vec<Box<dyn ToolT>> {
            vec![]
        }
    }

    #[async_trait]
    impl AgentExecutor for AbortAgent {
        type Output = TestAgentOutput;
        type Error = TestError;

        fn config(&self) -> ExecutorConfig {
            ExecutorConfig::default()
        }

        async fn execute(
            &self,
            _task: &Task,
            _context: Arc<Context>,
        ) -> Result<Self::Output, Self::Error> {
            self.executed.store(true, Ordering::SeqCst);
            Ok(TestAgentOutput {
                result: "should-not-run".to_string(),
            })
        }
    }

    #[async_trait]
    impl AgentHooks for AbortAgent {
        async fn on_run_start(&self, _task: &Task, _ctx: &Context) -> HookOutcome {
            HookOutcome::Abort
        }
    }

    #[tokio::test]
    async fn test_direct_agent_run_aborts_before_execute() {
        let executed = Arc::new(AtomicBool::new(false));
        let agent = AbortAgent {
            executed: Arc::clone(&executed),
        };
        let llm = Arc::new(ConfigurableLLMProvider::default());
        let handle = AgentBuilder::<_, DirectAgent>::new(agent)
            .llm(llm)
            .build()
            .await
            .expect("build should succeed");

        let task = Task::new("abort");
        let err = handle.agent.run(task).await.expect_err("expected abort");
        assert!(matches!(err, RunnableAgentError::Abort));
        assert!(!executed.load(Ordering::SeqCst));
    }
}
