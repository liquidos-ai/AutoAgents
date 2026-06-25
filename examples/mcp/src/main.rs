use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::ReActAgent;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentDeriveT, DirectAgent};
use autoagents::core::error::Error;
use autoagents::core::tool::ToolT;
use autoagents::llm::{backends::openai::OpenAI, builder::LLMBuilder};
use autoagents_derive::AgentHooks;
use autoagents_toolkit::mcp::{McpToolWrapper, McpTools};
use std::env;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone, AgentHooks)]
pub struct McpAgent {
    tools: Vec<Arc<dyn ToolT>>,
}

impl McpAgent {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let config_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config.toml");
        let mcp_tools = McpTools::from_config(&config_path).await?;
        let tools = mcp_tools.get_tools().await;

        Ok(Self { tools })
    }
}

impl AgentDeriveT for McpAgent {
    type Output = String;

    fn name(&self) -> &'static str {
        "mcp_agent"
    }

    fn description(&self) -> &'static str {
        "An agent that can use tools from MCP servers"
    }

    fn tools(&self) -> Vec<Box<dyn ToolT>> {
        self.tools
            .iter()
            .map(|tool| Box::new(McpToolWrapper::new(Arc::clone(tool))) as Box<dyn ToolT>)
            .collect()
    }

    fn output_schema(&self) -> Option<serde_json::Value> {
        None
    }
}

#[tokio::main]
#[allow(clippy::result_large_err)]
pub async fn main() -> Result<(), Error> {
    env_logger::init();

    // For web search without MCP, use the built-in BraveSearch tool (feature `search`)
    // with BRAVE_SEARCH_API_KEY instead of an npx-based remote MCP server.
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");

    let llm = LLMBuilder::<OpenAI>::new()
        .api_key(&api_key)
        .model("gpt-4o-mini")
        .max_tokens(1024)
        .temperature(0.2)
        .build()
        .expect("Failed to build OpenAI client");

    let mcp_agent = McpAgent::new()
        .await
        .map_err(|e| Error::CustomError(format!("Failed to create MCP agent: {e}")))?;

    let task_description = "Use the echo tool to repeat the message: hello from AutoAgents MCP";

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent_handle = AgentBuilder::<_, DirectAgent>::new(ReActAgent::new(mcp_agent))
        .llm(llm)
        .memory(sliding_window_memory)
        .build()
        .await?;

    let result = agent_handle.agent.run(Task::new(task_description)).await?;

    println!("Result: {:?}", result);
    Ok(())
}
