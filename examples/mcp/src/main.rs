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
use std::sync::Arc;

#[derive(Debug, Clone, AgentHooks)]
pub struct McpAgent {
    tools: Vec<Arc<dyn ToolT>>,
}

impl McpAgent {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Load MCP tools from config
        let mcp_tools = McpTools::from_config("./examples/mcp/config.toml").await?;
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
        None // Simple string output, no schema needed
    }
}

#[tokio::main]
pub async fn main() -> Result<(), Error> {
    env_logger::init();

    // Get API keys from environment
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");

    // Build LLM
    let llm = LLMBuilder::<OpenAI>::new()
        .api_key(&api_key)
        .model("gpt-4o-mini")
        .max_tokens(1024)
        .temperature(0.2)
        .build()
        .expect("Failed to build OpenAI client");

    // Create agent with MCP tools
    let mcp_agent = McpAgent::new()
        .await
        .map_err(|e| Error::CustomError(format!("Failed to create MCP agent: {}", e)))?;

    let task_description = "What is the latest news about artificial intelligence?";

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
