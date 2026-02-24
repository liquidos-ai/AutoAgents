use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::{BasicAgent, BasicAgentOutput};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentOutputT, DirectAgent};
use autoagents::core::error::Error;
use autoagents::llm::backends::minimax::MiniMax;
use autoagents::llm::builder::LLMBuilder;
use autoagents_derive::{AgentHooks, AgentOutput, agent};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Serialize, Deserialize, AgentOutput)]
struct ChatAgentOutput {
    #[output(description = "The response from the agent")]
    response: String,
}

impl From<BasicAgentOutput> for ChatAgentOutput {
    fn from(output: BasicAgentOutput) -> Self {
        ChatAgentOutput {
            response: output.response,
        }
    }
}

#[agent(
    name = "chat_agent",
    description = "You are a helpful assistant.",
    output = ChatAgentOutput,
)]
#[derive(Default, Clone, AgentHooks)]
struct ChatAgent {}

pub async fn run() -> Result<(), Error> {
    let api_key = std::env::var("MINIMAX_API_KEY").unwrap_or_else(|_| "".into());

    let llm: Arc<MiniMax> = LLMBuilder::<MiniMax>::new()
        .api_key(api_key)
        .base_url("https://api.minimax.chat/v1/")
        .model("MiniMax-M2.5")
        .max_tokens(512)
        .temperature(0.7)
        .build()
        .expect("Failed to build MiniMax LLM");

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent = BasicAgent::new(ChatAgent {});
    let agent_handle = AgentBuilder::<_, DirectAgent>::new(agent)
        .llm(llm)
        .memory(sliding_window_memory)
        .build()
        .await?;

    let result = agent_handle
        .agent
        .run(Task::new("What is 20 + 10?"))
        .await?;
    println!("Result: {:?}", result);
    Ok(())
}
