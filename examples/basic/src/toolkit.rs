use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::ReActAgent;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, DirectAgent};
use autoagents::core::error::Error;
use autoagents::llm::LLMProvider;
use autoagents_derive::{agent, AgentHooks};
use autoagents_toolkit::search::BraveSearch;
use std::sync::Arc;

#[agent(
    name = "agent",
    description = "You are a helpful agent who can search the internet for information",
    tools = [BraveSearch::new()],
)]
#[derive(Clone, AgentHooks)]
pub struct SearchAgent {}

pub async fn run_agent(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent_handle = AgentBuilder::<_, DirectAgent>::new(ReActAgent::new(SearchAgent {}))
        .llm(llm)
        .memory(sliding_window_memory)
        .build()
        .await?;

    let result = agent_handle
        .agent
        .run(Task::new("What is the current news related to AI?"))
        .await?;
    println!("Result: {:?}", result);
    Ok(())
}
