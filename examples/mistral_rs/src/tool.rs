//! Tool calling example with math operations

use autoagents::async_trait;
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::ReActAgent;
use autoagents::core::agent::prebuilt::executor::ReActAgentOutput;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentOutputT, DirectAgent};
use autoagents::core::error::Error;
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents_derive::{agent, tool, AgentHooks, AgentOutput, ToolInput};
use autoagents_mistral_rs::models::ModelType;
use autoagents_mistral_rs::{IsqType, MistralRsProvider, ModelSource};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

// Tool definitions for the math agent
#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct AdditionArgs {
    #[input(description = "Left operand for addition")]
    left: i32,
    #[input(description = "Right operand for addition")]
    right: i32,
}

#[tool(name = "addition", description = "Add two numbers together", input = AdditionArgs)]
struct Addition {}

#[async_trait]
impl ToolRuntime for Addition {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        println!("Executing Addition");
        let typed_args: AdditionArgs = serde_json::from_value(args)?;
        let result = typed_args.left + typed_args.right;
        Ok(result.into())
    }
}

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct MultiplicationArgs {
    #[input(description = "Left operand for multiplication")]
    left: i32,
    #[input(description = "Right operand for multiplication")]
    right: i32,
}

#[tool(name = "multiplication", description = "Multiply two numbers together", input = MultiplicationArgs
)]
struct Multiplication {}

#[async_trait]
impl ToolRuntime for Multiplication {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        println!("Executing Multiplication");
        let typed_args: MultiplicationArgs = serde_json::from_value(args)?;
        let result = typed_args.left * typed_args.right;
        Ok(result.into())
    }
}

// Agent output structure
#[derive(Serialize, Deserialize, AgentOutput, Debug, Clone)]
pub struct MathAgentOutput {
    #[output(description = "The numerical result of the calculation")]
    value: i32,
    #[output(description = "Explanation of the calculation")]
    explanation: String,
}

impl From<ReActAgentOutput> for MathAgentOutput {
    fn from(output: ReActAgentOutput) -> Self {
        let resp = output.response;
        if output.done && !resp.trim().is_empty() {
            // Try to parse as JSON first
            if let Ok(value) = serde_json::from_str::<MathAgentOutput>(&resp) {
                return value;
            }
        }
        MathAgentOutput {
            value: 0,
            explanation: resp,
        }
    }
}

#[agent(
    name = "math_agent",
    description = "You solve math problems by calling tools step-by-step.

RULES:
1. ALWAYS call tools for calculations - NEVER calculate yourself
2. For multi-step problems, call tools multiple times in sequence
3. After ALL calculations are done, return ONLY valid JSON in this EXACT format:
{\"value\": <final_number>, \"explanation\": \"<how you solved it>\"}

Example for \"What is (20 + 30) * 10?\":
- Step 1: Call addition(20, 30) → get 50
- Step 2: Return {\"value\": 50, \"explanation\": \"Add 20 + 30 to  get 50\"}

CRITICAL: Your final response MUST be valid JSON with 'value' and 'explanation' fields.",
    tools = [Addition, Multiplication],
    output = MathAgentOutput,
)]
#[derive(Default, Clone, AgentHooks)]
pub struct MathAgent {}

pub struct ToolArgs {
    pub repo_id: Option<String>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub paged_attention: bool,
    pub verbose: bool,
}

/// Load a tool-calling model from HuggingFace
pub async fn load_model(args: &ToolArgs) -> Result<Arc<MistralRsProvider>, Error> {
    let repo_id = args
        .repo_id
        .clone()
        .unwrap_or_else(|| "NousResearch/Hermes-3-Llama-3.1-8B".to_string());

    println!("   Repository: {}", repo_id);
    println!("   Quantization: ISQ Q8_0 (8-bit)");
    println!("   Max tokens: {}", args.max_tokens);
    println!("   Temperature: {}", args.temperature);
    println!("   Paged attention: {}", args.paged_attention);
    println!("   Verbose logging: {}\n", args.verbose);

    let mut builder = MistralRsProvider::builder()
        .model_source(ModelSource::HuggingFace {
            repo_id,
            revision: None,
            model_type: ModelType::Text,
        })
        .with_isq(IsqType::Q8_0)
        .max_tokens(args.max_tokens)
        .temperature(args.temperature);

    if args.paged_attention {
        builder = builder.with_paged_attention();
    }

    if args.verbose {
        builder = builder.with_logging();
    }

    let provider = builder
        .build()
        .await
        .map_err(|e| Error::CustomError(e.to_string()))?;

    Ok(Arc::new(provider))
}

/// Run queries suitable for tool-calling models
pub async fn run_example(llm: Arc<MistralRsProvider>) -> Result<(), Error> {
    println!("Running Tool Calling Queries ...");

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));
    let agent = ReActAgent::new(MathAgent {});
    let agent_handle = AgentBuilder::<_, DirectAgent>::new(agent)
        .llm(llm)
        .memory(sliding_window_memory)
        .build()
        .await?;

    println!("Query 1: Simple addition");
    let result1 = agent_handle
        .agent
        .run(Task::new("What is 42 + 58?"))
        .await?;
    println!("Result: {:?}\n", result1);

    println!("Query 2: Multiplication");
    let result2 = agent_handle
        .agent
        .run(Task::new("Calculate 15 multiplied by 8"))
        .await?;
    println!("Result: {:?}\n", result2);

    Ok(())
}
