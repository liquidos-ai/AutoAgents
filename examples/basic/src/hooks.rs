use autoagents::async_trait;
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::BasicAgent;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentHooks, Context, DirectAgent, HookOutcome};
use autoagents::core::error::Error;
use autoagents::llm::LLMProvider;
use autoagents_derive::agent;
use std::sync::Arc;
use std::time::Duration;

#[agent(name = "hooks_agent", description = "You are a helpful assistant")]
#[derive(Default, Clone)]
pub struct Agent {}

#[async_trait]
impl AgentHooks for Agent {
    async fn on_agent_create(&self) {
        println!("Agent Create Hook");
    }

    async fn on_run_start(&self, _task: &Task, _ctx: &Context) -> HookOutcome {
        println!("Agent Start Hook");
        HookOutcome::Continue
    }
}

pub async fn hooks_agent(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent_handle = AgentBuilder::<_, DirectAgent>::new(BasicAgent::new(Agent {}))
        .llm(llm)
        .memory(sliding_window_memory)
        .build()
        .await?;

    println!("Running hooks example");

    let result = agent_handle.agent.run(Task::new("Hey, There!")).await?;
    println!("Result: {:?}", result);
    tokio::time::sleep(Duration::from_secs(5)).await;
    Ok(())
}
