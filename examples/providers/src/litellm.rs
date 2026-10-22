use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::{BasicAgent, BasicAgentOutput};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentOutputT, DirectAgent};
use autoagents::core::error::Error;
use autoagents::llm::backends::litellm::LiteLLM;
use autoagents::llm::builder::LLMBuilder;
use autoagents::llm::models::ModelsProvider;
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
    // LiteLLM's own variable names; both are optional for a local proxy
    // started without a master key.
    let base_url = std::env::var("LITELLM_PROXY_API_BASE")
        .map(|base| format!("{}/v1", base.trim_end_matches('/')))
        .unwrap_or_else(|_| "http://localhost:4000/v1".into());

    let mut builder = LLMBuilder::<LiteLLM>::new()
        .base_url(base_url)
        // A `model_name` from your proxy config.
        .model("gpt-4o-mini")
        .max_tokens(512)
        .temperature(0.7);
    if let Ok(api_key) = std::env::var("LITELLM_PROXY_API_KEY") {
        builder = builder.api_key(api_key);
    }
    let llm: Arc<LiteLLM> = builder.build().expect("Failed to build LiteLLM LLM");

    match llm.list_models(None).await {
        Ok(models) => println!("Models on the proxy: {:?}", models.get_models()),
        Err(err) => println!("Could not list proxy models: {err}"),
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn litellm_output_preserves_response_text() {
        let output = ChatAgentOutput::from(BasicAgentOutput {
            response: "hello".to_string(),
            done: true,
        });
        assert_eq!(output.response, "hello");
    }
}
