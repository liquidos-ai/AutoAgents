use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::{BasicAgent, BasicAgentOutput};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentOutputT, DirectAgent};
use autoagents::core::error::Error;
use autoagents::llm::backends::ollama::Ollama;
use autoagents::llm::builder::LLMBuilder;
use autoagents_derive::{AgentHooks, AgentOutput, agent};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Serialize, Deserialize, AgentOutput)]
struct MathAgentOutput {
    #[output(description = "The addition result")]
    value: i64,
    #[output(description = "Explanation of the logic")]
    explanation: String,
    #[output(description = "If user asks other than math questions, use this to answer them.")]
    generic: Option<String>,
}

impl From<BasicAgentOutput> for MathAgentOutput {
    fn from(output: BasicAgentOutput) -> Self {
        let resp = output.response;
        if output.done && !resp.trim().is_empty() {
            // Try to parse as structured JSON first
            if let Ok(value) = serde_json::from_str::<MathAgentOutput>(&resp) {
                return value;
            }
        }
        // For streaming chunks or unparseable content, create a default response
        MathAgentOutput {
            value: 0,
            explanation: resp,
            generic: None,
        }
    }
}

#[agent(
    name = "math_agent",
    description = "You are a Math agent",
    output = MathAgentOutput
)]
#[derive(Default, Clone, AgentHooks)]
struct MathAgent {}

pub async fn run() -> Result<(), Error> {
    // Ollama runs locally, no API key needed

    // Initialize and configure the LLM client
    let llm: Arc<Ollama> = LLMBuilder::<Ollama>::new()
        .base_url("http://localhost:11434") // Local Ollama server
        .model("llama3.2:3b")
        .keep_alive("0")
        .max_tokens(512) // Limit response length
        .temperature(0.2) // Control response randomness (0.0-1.0)
        .build()
        .expect("Failed to build LLM");

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent = BasicAgent::new(MathAgent {});
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
