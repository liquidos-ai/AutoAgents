use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::ReActAgent;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, DirectAgent};
use autoagents::core::error::Error;
use autoagents::llm::backends::openai::OpenAI;
use autoagents::llm::builder::LLMBuilder;
use autoagents::protocol::{Event, StreamChunk};
use std::sync::Arc;
use tokio::select;
use tokio_stream::StreamExt;

pub async fn agent_with_thinking() -> Result<(), Error> {
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "EMPTY".to_string());
    let base_url = std::env::var("VLLM_BASE_URL")
        .or_else(|_| std::env::var("OPENAI_BASE_URL"))
        .unwrap_or_else(|_| "http://127.0.0.1:8000/v1".to_string());
    let model = std::env::var("VLLM_MODEL").unwrap_or_else(|_| "Qwen/Qwen3-8B".to_string());

    let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(api_key)
        .base_url(base_url)
        .model(model)
        .max_tokens(512)
        .reasoning(true)
        .reasoning_effort(autoagents::llm::chat::ReasoningEffort::Medium)
        .extra_body(serde_json::json!({
            "chat_template_kwargs": {
                "enable_thinking": true
            }
        }))
        .temperature(0.2)
        .build()
        .map_err(|e| Error::CustomError(e.to_string()))?;

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));
    let mut agent_handle =
        AgentBuilder::<_, DirectAgent>::new(ReActAgent::new(crate::basic::MathAgent {}))
            .llm(llm)
            .memory(sliding_window_memory)
            .stream(true)
            .build()
            .await?;

    let mut event_stream = agent_handle.subscribe_events();

    println!("Running run_stream() and reading reasoning from events");
    let mut stream = agent_handle
        .agent
        .run_stream(Task::new("What is (20 + 30) * 10?"))
        .await?;
    let mut final_output = None;
    let mut stream_done = false;
    let mut events_done = false;
    let mut saw_reasoning_event = false;

    while !(stream_done && events_done) {
        select! {
            item = stream.next(), if !stream_done => {
                match item {
                    Some(Ok(output)) => {
                        final_output = Some(output);
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
                                println!("reasoning event: {}", content);
                            }
                        }
                        StreamChunk::Text(content) => {
                            if !content.is_empty() {
                                println!("text event: {}", content);
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

    if let Some(output) = final_output {
        println!("final output: {:?}", output);
    }
    if !saw_reasoning_event {
        println!(
            "note: no reasoning_content events were emitted by this model/provider for this run."
        );
    }

    Ok(())
}
