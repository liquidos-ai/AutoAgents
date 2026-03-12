//! Thinking/reasoning stream example for mistral.rs

use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::ReActAgent;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, DirectAgent};
use autoagents::core::error::Error;
use autoagents::protocol::{Event, StreamChunk};
use autoagents_derive::{AgentHooks, agent};
use autoagents_mistral_rs::models::ModelType;
use autoagents_mistral_rs::{IsqType, MistralRsProvider, ModelSource};
use std::sync::Arc;
use tokio::select;
use tokio_stream::StreamExt;

#[agent(
    name = "thinking_agent",
    description = "You are a careful reasoning assistant. Think step-by-step and answer clearly.",
    tools = [],
)]
#[derive(Default, Clone, AgentHooks)]
pub struct ThinkingAgent {}

pub struct ThinkingArgs {
    pub repo_id: Option<String>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub paged_attention: bool,
    pub verbose: bool,
}

pub async fn load_model(args: &ThinkingArgs) -> Result<Arc<MistralRsProvider>, Error> {
    let repo_id = args
        .repo_id
        .clone()
        .unwrap_or_else(|| "Qwen/Qwen3-8B".to_string());

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

pub async fn run_example(llm: Arc<MistralRsProvider>) -> Result<(), Error> {
    println!("Running Thinking/Reasoning Query ...");

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));
    let mut agent_handle = AgentBuilder::<_, DirectAgent>::new(ReActAgent::new(ThinkingAgent {}))
        .llm(llm)
        .memory(sliding_window_memory)
        .stream(true)
        .build()
        .await?;

    let mut event_stream = agent_handle.subscribe_events();
    let mut stream = agent_handle
        .agent
        .run_stream(Task::new("What is (20 + 30) * 10?"))
        .await?;
    let mut stream_done = false;
    let mut events_done = false;
    let mut saw_reasoning_event = false;

    while !(stream_done && events_done) {
        select! {
            item = stream.next(), if !stream_done => {
                match item {
                    Some(Ok(output)) => {
                        print!("{output}");
                    }
                    Some(Err(err)) => {
                        println!("stream error: {err}");
                        stream_done = true;
                    }
                    None => {
                        stream_done = true;
                    }
                }
            }
            event = event_stream.next(), if !events_done => {
                match event {
                    Some(Event::StreamChunk { chunk, .. }) => match chunk {
                        StreamChunk::ReasoningContent(content) => {
                            if !content.is_empty() {
                                saw_reasoning_event = true;
                                println!("\nreasoning event: {}", content);
                            }
                        }
                        StreamChunk::Text(content) => {
                            if !content.is_empty() {
                                println!("\ntext event: {}", content);
                            }
                        }
                        _ => {}
                    },
                    Some(Event::StreamComplete { .. }) => {
                        events_done = true;
                    }
                    Some(_) => {}
                    None => {
                        events_done = true;
                    }
                }
            }
        }
    }

    if !saw_reasoning_event {
        println!("note: no reasoning_content events were emitted by this model/provider.");
    }

    Ok(())
}
