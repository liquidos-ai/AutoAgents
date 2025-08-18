use autoagents::async_trait;
use autoagents::core::agent::{
    ActorTask, AgentBuilder, AgentConfig, AgentDeriveT, AgentExecutor, AgentOutputT, AgentState,
    ExecutorConfig,
};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::memory::{MemoryProvider, SlidingWindowMemory};
use autoagents::core::protocol::{Event, TaskResult};
use autoagents::core::runtime::{Runtime, SingleThreadedRuntime, Task};
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents::llm::LLMProvider;
use autoagents_derive::{agent, tool, AgentOutput, ToolInput};
use colored::*;
use futures::stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{sync::Arc, time::Instant};
use tokio::sync::{mpsc, RwLock};
use tokio::time::{sleep, Duration};
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct AdditionArgs {
    #[input(description = "Left Operand for addition")]
    left: i64,
    #[input(description = "Right Operand for addition")]
    right: i64,
}

#[tool(
    name = "Addition",
    description = "Use this tool to Add two numbers",
    input = AdditionArgs,
)]
struct Addition {}

impl ToolRuntime for Addition {
    fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let typed_args: AdditionArgs = serde_json::from_value(args)?;
        let result = typed_args.left + typed_args.right;
        Ok(result.into())
    }
}

/// Math agent output with Value and Explanation
#[derive(Debug, Serialize, Deserialize, AgentOutput)]
pub struct MathAgentOutput {
    #[output(description = "The addition result")]
    value: i64,
    #[output(description = "Explanation of the logic")]
    explanation: String,
}

#[agent(
    name = "math_agent",
    description = "You are a Math agent",
    tools = [Addition]
    output = MathAgentOutput
)]
pub struct MathAgent {}

#[async_trait]
impl AgentExecutor for MathAgent {
    type Error = Error;
    type Output = String;

    fn config(&self) -> autoagents::core::agent::ExecutorConfig {
        ExecutorConfig::default()
    }

    async fn execute(
        &self,
        _llm: Arc<dyn LLMProvider>,
        _memory: Option<Arc<RwLock<Box<dyn MemoryProvider>>>>,
        _tools: Vec<Box<dyn ToolT>>,
        _agent_config: &AgentConfig,
        task: Box<dyn ActorTask>,
        _state: Arc<RwLock<AgentState>>,
        _tx_event: mpsc::Sender<Event>,
        stream: bool,
    ) -> Result<Self::Output, Self::Error> {
        let start = Instant::now();
        let task = task
            .as_any()
            .downcast_ref::<Task>()
            .expect("Expected Task type")
            .clone();

        println!(
            "[{:?}] MathAgent received: {}",
            start.elapsed(),
            task.prompt
        );

        let sleep_time = if task.prompt.contains("Hello") { 10 } else { 2 };

        sleep(Duration::from_secs(sleep_time)).await;

        let end = Instant::now();
        println!(
            "[{:?}] MathAgent finished: {} (slept {}s)",
            end.elapsed(),
            task.prompt,
            sleep_time
        );

        Ok(format!("Processed task: {}", task.prompt))
    }
}

pub async fn single_threaded_agent(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent = MathAgent {};

    let runtime = SingleThreadedRuntime::new(None);

    let _ = AgentBuilder::new(agent)
        .with_llm(llm)
        .runtime(runtime.clone())
        .subscribe_topic("test")
        .with_memory(sliding_window_memory)
        .build()
        .await?;

    // Create environment and set up event handling
    let mut environment = Environment::new(None);
    let _ = environment.register_runtime(runtime.clone()).await;

    let receiver = environment.take_event_receiver(None).await?;
    handle_events(receiver);

    runtime
        .publish_message("Hello".into(), "test".into())
        .await
        .unwrap();
    runtime
        .publish_message("How are you?".into(), "test".into())
        .await
        .unwrap();

    let _ = environment.run().await;
    Ok(())
}

fn handle_events(mut event_stream: ReceiverStream<Event>) {
    tokio::spawn(async move {
        while let Some(event) = event_stream.next().await {
            match event {
                Event::TaskComplete { result, .. } => match result {
                    TaskResult::Value(val) => {
                        let agent_out: String = serde_json::from_value(val).unwrap();
                        println!("{}", format!("Result: {agent_out}",).green());
                    }
                    _ => {
                        println!("{}", format!("Error!!!").red());
                    }
                },
                _ => {
                    //Do nothing
                }
            }
        }
    });
}
