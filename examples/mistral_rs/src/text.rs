//! Text model example

use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::BasicAgent;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, DirectAgent};
use autoagents::core::error::Error;
use autoagents_derive::{AgentHooks, agent};
use autoagents_mistral_rs::models::ModelType;
use autoagents_mistral_rs::{IsqType, MistralRsProvider, ModelSource};
use std::sync::Arc;

#[agent(
    name = "demo_agent",
    description = "You are a helpful AI assistant, Your name is Emma, Your job is to answer questions clearly and concisely. When user asks your name reply with Emma",
    tools = [],
)]
#[derive(Default, Clone, AgentHooks)]
pub struct DemoAgent {}

pub struct TextArgs {
    pub repo_id: Option<String>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub paged_attention: bool,
    pub verbose: bool,
}

/// Load a text model from HuggingFace with ISQ quantization
pub async fn load_model(args: &TextArgs) -> Result<Arc<MistralRsProvider>, Error> {
    let repo_id = args
        .repo_id
        .clone()
        .unwrap_or_else(|| "microsoft/Phi-3.5-mini-instruct".to_string());

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

/// Run queries suitable for text models
pub async fn run_example(llm: Arc<MistralRsProvider>) -> Result<(), Error> {
    println!("Running Text Queries ...");

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));
    let agent = BasicAgent::new(DemoAgent {});
    let agent_handle = AgentBuilder::<_, DirectAgent>::new(agent)
        .llm(llm)
        .memory(sliding_window_memory)
        .build()
        .await?;

    let result = agent_handle
        .agent
        .run(Task::new("What is your name?"))
        .await?;

    println!("Response: {:?}\n", result);

    Ok(())
}
