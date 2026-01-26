#![allow(unused_imports)]
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentOutputT, DirectAgent};
use autoagents::core::error::Error;
use autoagents::core::tool::ToolCallError;
use autoagents::prelude::{ReActAgent, ReActAgentOutput};
use autoagents::prelude::{ToolInputT, ToolRuntime, ToolT};
use autoagents::{async_trait, init_logging};
use autoagents_derive::{AgentHooks, AgentOutput, ToolInput, agent, tool};
use autoagents_llamacpp::{LlamaCppProvider, ModelSource};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio_stream::StreamExt;

#[derive(Parser, Debug)]
#[command(version, about = "Run an AutoAgents math agent with llama.cpp", long_about = None)]
struct Args {
    #[arg(long, default_value = "What is 20 + 10?")]
    prompt: String,

    #[arg(long, help = "Optional chat template name or inline template")]
    chat_template: Option<String>,

    #[arg(long, help = "Context size override")]
    n_ctx: Option<u32>,

    #[arg(long, help = "Thread count override")]
    n_threads: Option<i32>,

    #[arg(long, default_value_t = 256, help = "Max tokens to generate")]
    max_tokens: u32,

    #[arg(long, default_value_t = 0.2, help = "Sampling temperature")]
    temperature: f32,
}

#[derive(Debug, Serialize, Deserialize, AgentOutput)]
#[allow(dead_code)]
struct MathAgentOutput {
    #[output(description = "The addition result")]
    value: i64,
    #[output(description = "Explanation of the logic")]
    explanation: String,
    #[output(description = "If user asks other than math questions, use this to answer them.")]
    generic: Option<String>,
}

impl From<ReActAgentOutput> for MathAgentOutput {
    fn from(output: ReActAgentOutput) -> Self {
        let resp = output.response;
        if output.done
            && !resp.trim().is_empty()
            && let Ok(value) = serde_json::from_str::<MathAgentOutput>(&resp)
        {
            return value;
        }
        MathAgentOutput {
            value: 0,
            explanation: resp,
            generic: None,
        }
    }
}

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

#[async_trait]
impl ToolRuntime for Addition {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        println!("execute tool: {:?}", args);
        let typed_args: AdditionArgs = serde_json::from_value(args)?;
        let result = typed_args.left + typed_args.right;
        Ok(result.into())
    }
}

#[agent(
    name = "math_agent",
    description = "You are a Math agent, answer user question and also explain them why you got the answer",
    tools = [Addition],
    // output = MathAgentOutput //Does not work properly comment for now TODO
)]
#[derive(Default, Clone, AgentHooks)]
struct MathAgent {}

#[tokio::main]
#[allow(clippy::result_large_err)]
async fn main() -> Result<(), Error> {
    init_logging();
    let args = Args::parse();

    let mut builder = LlamaCppProvider::builder()
        .model_source(ModelSource::HuggingFace {
            repo_id: "unsloth/Llama-3.2-3B-Instruct-GGUF".to_string(),
            filename: Some("Llama-3.2-3B-Instruct-Q8_0.gguf".to_string()),
        })
        .max_tokens(args.max_tokens)
        .temperature(args.temperature);

    if let Some(template) = args.chat_template {
        builder = builder.chat_template(template);
    }
    if let Some(n_ctx) = args.n_ctx {
        builder = builder.n_ctx(n_ctx);
    }
    if let Some(n_threads) = args.n_threads {
        builder = builder.n_threads(n_threads);
    }

    let llm = builder
        .build()
        .await
        .expect("Failed to build llama.cpp provider");
    let llm: Arc<dyn autoagents::llm::LLMProvider> = Arc::new(llm);

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent = ReActAgent::new(MathAgent {});
    let agent_handle = AgentBuilder::<_, DirectAgent>::new(agent)
        .llm(llm)
        .memory(sliding_window_memory)
        .build()
        .await?;

    let result = agent_handle
        .agent
        .run(Task::new(args.prompt.clone()))
        .await?;
    println!("Result: {:?}", result);

    // Process the stream directly
    let mut stream = agent_handle
        .agent
        .run_stream(Task::new(args.prompt))
        .await?;

    println!("🌊 Agent Streaming Example");
    println!("🔄 Processing stream tokens...\n");

    while let Some(result) = stream.next().await {
        match result {
            Ok(output) => {
                print!("{}", output);
            }
            _ => {
                //
            }
        }
    }

    Ok(())
}
