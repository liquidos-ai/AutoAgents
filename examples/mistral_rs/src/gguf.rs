//! GGUF quantized model example

use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::BasicAgent;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, DirectAgent};
use autoagents::core::error::Error;
use autoagents_mistral_rs::{GgufQuant, HFModels, MistralRsProvider};
use std::sync::Arc;

pub use crate::text::DemoAgent;

pub struct GgufArgs {
    pub model_dir: String,
    pub quant: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub paged_attention: bool,
    pub verbose: bool,
}

/// Load a GGUF model from local files
pub async fn load_model(args: &GgufArgs) -> Result<Arc<MistralRsProvider>, Error> {
    // Parse quantization type
    let quant = match args.quant.to_lowercase().as_str() {
        "q4-k-m" | "q4km" => GgufQuant::Q4_K_M,
        "q4-k-s" | "q4ks" => GgufQuant::Q4_K_S,
        "q5-k-m" | "q5km" => GgufQuant::Q5_K_M,
        "q5-k-s" | "q5ks" => GgufQuant::Q5_K_S,
        "q8-0" | "q8" => GgufQuant::Q8_0,
        "f16" => GgufQuant::F16,
        "f32" => GgufQuant::F32,
        _ => {
            println!(
                "WARNING: Unknown quantization '{}', using Q4_K_M",
                args.quant
            );
            GgufQuant::Q4_K_M
        }
    };

    println!("   Model directory: {}", args.model_dir);
    println!("   Quantization: {}", args.quant);
    println!("   Max tokens: {}", args.max_tokens);
    println!("   Temperature: {}", args.temperature);
    println!("   Paged attention: {}", args.paged_attention);
    println!("   Verbose logging: {}\n", args.verbose);

    // Use SupportedModel for GGUF
    let model = HFModels::Phi35MiniInstructGguf {
        quant,
        model_dir: args.model_dir.clone(),
    };

    let mut builder = MistralRsProvider::builder()
        .model_source(model.to_source())
        .max_tokens(args.max_tokens)
        .temperature(args.temperature);

    // Note: Paged attention with GGUF on CUDA can cause issues
    // Only enable if explicitly requested
    if args.paged_attention {
        println!("   WARNING: Paged attention with GGUF + CUDA may cause errors");
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

/// Run queries suitable for GGUF models (same as text models)
pub async fn run_example(llm: Arc<MistralRsProvider>) -> Result<(), Error> {
    println!("Running GGUF model queries...\n");

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
