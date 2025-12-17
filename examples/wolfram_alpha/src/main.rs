use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::ReActAgent;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentHooks, Context, DirectAgent};
use autoagents::core::error::Error;
use autoagents::llm::backends::openai::OpenAI;
use autoagents::llm::builder::LLMBuilder;
use autoagents::llm::ToolCall;
use autoagents::prelude::ToolCallResult;
use autoagents::{async_trait, init_logging};
use autoagents_derive::agent;
use autoagents_toolkit::tools::wolfram_alpha::WolframAlphaLLMApi;
use std::sync::Arc;

#[agent(
    name = "wolfram_agent",
    description = "You are an expert scientific calculation agent using ReAct prompting; fetch the answer using the appropriate WolframAlpha tool, ensuring responses are precise, scientific, and mathematically accurate."
    tools = [
        WolframAlphaLLMApi::new(),
    ],
)]
#[derive(Clone)]
struct WolframAgent;

#[async_trait]
impl AgentHooks for WolframAgent {
    async fn on_tool_result(&self, _tool_call: &ToolCall, result: &ToolCallResult, _ctx: &Context) {
        println!("Tool Call Result: {:?}", result);
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    init_logging();

    let openai_api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set to run this example");
    // Fail fast if the Wolfram AppID is missing so users get a clear message.
    let _wolfram_app_id = std::env::var("WOLFRAM_ALPHA_APP_ID")
        .or_else(|_| std::env::var("WOLFRAM_APP_ID"))
        .expect("WOLFRAM_ALPHA_APP_ID or WOLFRAM_APP_ID must be set to run this example");

    let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(openai_api_key)
        .model("gpt-4o")
        .max_tokens(512)
        .temperature(0.2)
        .build()
        .expect("Failed to build OpenAI client");

    let memory = Box::new(SlidingWindowMemory::new(10));

    let agent_handle = AgentBuilder::<_, DirectAgent>::new(ReActAgent::new(WolframAgent {}))
        .llm(llm)
        .memory(memory)
        .build()
        .await?;

    let task = Task::new("get zodiac constellations visible from Chicago at 10PM");
    let result = agent_handle.agent.run(task).await?;
    println!("Agent result: {result}");

    Ok(())
}
