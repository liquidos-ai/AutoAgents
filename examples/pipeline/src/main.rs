//! Demonstrates the `PipelineBuilder` optimization pipeline.
//!
//! A bare [`LLMProvider`] is wrapped with [`RetryLayer`] and [`CacheLayer`].
//! The result is a plain [`Arc<dyn LLMProvider>`] — a transparent drop-in
//! that all agent and raw-LLM code accepts without modification.
//!
//! Four scenarios are covered in order of increasing complexity:
//!
//! 1. **Cache hit / miss** — timing comparison on a single repeated query.
//! 2. **Independent entries** — distinct queries each occupy their own slot.
//! 3. **TTL expiry** — a short-lived pipeline evicts stale entries on read.
//! 4. **Agent integration** — a ReAct agent uses the pipeline transparently;
//!    a second run with fresh memory shows every LLM call served from cache.
//!
//! ## Usage
//!
//! ```bash
//! export OPENAI_API_KEY=sk-...
//! cargo run --package pipeline-example
//! ```

use std::slice;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use autoagents::async_trait;
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::{ReActAgent, ReActAgentOutput};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentOutputT, DirectAgent};
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents::llm::LLMProvider;
use autoagents::llm::backends::openai::OpenAI;
use autoagents::llm::builder::LLMBuilder;
use autoagents::llm::chat::ChatMessage;
use autoagents::llm::optim::{CacheConfig, CacheLayer, RetryConfig, RetryLayer};
use autoagents::llm::pipeline::PipelineBuilder;
use autoagents_derive::{AgentHooks, AgentOutput, ToolInput, agent, tool};
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Scenario 1 — Cache hit / miss
// ---------------------------------------------------------------------------

/// Sends the same query twice and prints the elapsed time for each call.
///
/// The first call is a cache miss and incurs a network round-trip.
/// The second call is served instantly from the in-process cache.
async fn scenario_cache_hit_miss(llm: Arc<dyn LLMProvider>) -> Result<()> {
    println!("\nScenario 1 — cache hit / miss");
    println!("  query: \"What is the capital of France?\"");

    let msg = ChatMessage::user()
        .content("What is the capital of France?")
        .build();

    let t = Instant::now();
    let miss = llm.chat(slice::from_ref(&msg), None).await?;
    let miss_ms = t.elapsed().as_millis();

    let t = Instant::now();
    let hit = llm.chat(slice::from_ref(&msg), None).await?;
    let hit_ms = t.elapsed().as_millis();

    println!(
        "  MISS  {:>6}ms  {}",
        miss_ms,
        miss.text().as_deref().unwrap_or("<no text>")
    );
    println!(
        "  HIT   {:>6}ms  {}  (saved {}ms)",
        hit_ms,
        hit.text().as_deref().unwrap_or("<no text>"),
        miss_ms.saturating_sub(hit_ms),
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Scenario 2 — Independent cache entries
// ---------------------------------------------------------------------------

/// Sends three distinct queries, then repeats them to confirm each has its
/// own cache entry and all three are served as hits on the second pass.
async fn scenario_independent_entries(llm: Arc<dyn LLMProvider>) -> Result<()> {
    println!("\nScenario 2 — independent cache entries");

    let queries = [
        "What is 2 + 2?",
        "What is the boiling point of water in Celsius?",
        "Name one planet in our solar system.",
    ];

    println!("  Pass 1 — populate cache:");
    for q in &queries {
        let msg = ChatMessage::user().content(*q).build();
        let resp = llm.chat(slice::from_ref(&msg), None).await?;
        println!(
            "    MISS  {:?}  →  {}",
            q,
            resp.text().as_deref().unwrap_or("<no text>")
        );
    }

    println!("  Pass 2 — all queries served from cache:");
    for q in &queries {
        let msg = ChatMessage::user().content(*q).build();
        let t = Instant::now();
        let resp = llm.chat(slice::from_ref(&msg), None).await?;
        println!(
            "    HIT   {:>4}µs  →  {}",
            t.elapsed().as_micros(),
            resp.text().as_deref().unwrap_or("<no text>")
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Scenario 3 — TTL expiry
// ---------------------------------------------------------------------------

/// Builds a separate pipeline with a 200 ms TTL and verifies that a cached
/// entry is re-fetched from the provider after the TTL elapses.
async fn scenario_ttl_expiry(base: Arc<dyn LLMProvider>) -> Result<()> {
    println!("\nScenario 3 — TTL-based cache expiry  (TTL = 200ms)");

    let short_ttl = PipelineBuilder::new(base)
        .add_layer(CacheLayer::new(CacheConfig {
            ttl: Some(Duration::from_millis(200)),
            ..CacheConfig::default()
        }))
        .build();

    let msg = ChatMessage::user().content("Say 'hello'").build();

    short_ttl.chat(slice::from_ref(&msg), None).await?;
    println!("  MISS  initial request stored in cache");

    let t = Instant::now();
    short_ttl.chat(slice::from_ref(&msg), None).await?;
    println!(
        "  HIT   {:>4}µs  served from cache",
        t.elapsed().as_micros()
    );

    tokio::time::sleep(Duration::from_millis(300)).await;
    println!("  (waited 300ms — TTL has elapsed)");

    let t = Instant::now();
    short_ttl.chat(slice::from_ref(&msg), None).await?;
    println!(
        "  MISS  {:>4}ms  re-fetched from provider after expiry",
        t.elapsed().as_millis()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Scenario 4 — Agent integration
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, ToolInput)]
pub struct AddArgs {
    #[input(description = "Left operand")]
    left: i64,
    #[input(description = "Right operand")]
    right: i64,
}

#[tool(name = "Add", description = "Add two integers", input = AddArgs)]
struct Add;

#[async_trait]
impl ToolRuntime for Add {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let a: AddArgs = serde_json::from_value(args)?;
        println!("Tool called with args: {:?}", a);
        Ok((a.left + a.right).into())
    }
}

#[derive(Debug, Serialize, Deserialize, AgentOutput)]
pub struct CalcOutput {
    #[output(description = "The numeric result")]
    result: i64,
    #[output(description = "Brief explanation")]
    explanation: String,
}

#[agent(
    name = "calc_agent",
    description = "Solve arithmetic using the Add tool. \
                   Always call the tool; never compute mentally. \
                   Return JSON with 'result' and 'explanation'.",
    tools = [Add],
    output = CalcOutput,
)]
#[derive(Default, Clone, AgentHooks)]
pub struct CalcAgent;

impl From<ReActAgentOutput> for CalcOutput {
    fn from(out: ReActAgentOutput) -> Self {
        out.parse_or_map(|s| CalcOutput {
            result: 0,
            explanation: s.to_string(),
        })
    }
}

/// Runs the same arithmetic task twice through the cached pipeline.
///
/// Each run uses a **fresh** agent (empty memory) backed by the **same** LLM.
/// Because memory starts empty both runs produce an identical message sequence,
/// so every LLM call in the second run is served as a cache hit.
async fn scenario_agent_integration(llm: Arc<dyn LLMProvider>) -> Result<()> {
    println!("\nScenario 4 — agent integration");
    println!("  task: \"What is 17 + 25?\"");
    println!("  Both agents receive Arc<dyn LLMProvider>; neither knows a cache exists.");

    let task = "What is 17 + 25?";

    let t = Instant::now();
    let handle1 = AgentBuilder::<_, DirectAgent>::new(ReActAgent::new(CalcAgent))
        .llm(llm.clone())
        .memory(Box::new(SlidingWindowMemory::new(10)))
        .build()
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let result1 = handle1
        .agent
        .run(Task::new(task))
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let run1_ms = t.elapsed().as_millis();

    let t = Instant::now();
    let handle2 = AgentBuilder::<_, DirectAgent>::new(ReActAgent::new(CalcAgent))
        .llm(llm.clone())
        .memory(Box::new(SlidingWindowMemory::new(10)))
        .build()
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let result2 = handle2
        .agent
        .run(Task::new(task))
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let run2_ms = t.elapsed().as_millis();

    println!(
        "  run 1  {:>6}ms  (network)  result={}",
        run1_ms, result1.result
    );
    println!(
        "  run 2  {:>6}ms  (cache)    result={}",
        run2_ms, result2.result
    );
    println!(
        "  speedup {:.1}×  (saved {}ms)",
        run1_ms as f64 / run2_ms.max(1) as f64,
        run1_ms.saturating_sub(run2_ms),
    );

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    autoagents::init_logging();

    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();

    let base: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(api_key)
        .model("gpt-4o-mini")
        .max_tokens(256)
        .temperature(0.0)
        .build()
        .expect("failed to build OpenAI provider");

    // Pipeline:  CacheLayer  →  RetryLayer  →  OpenAI
    //
    // Layers are applied outermost-first: cache intercepts before retry, so
    // hits never reach the retry logic or the network.
    let llm: Arc<dyn LLMProvider> = PipelineBuilder::new(base.clone() as Arc<dyn LLMProvider>)
        .add_layer(CacheLayer::new(CacheConfig {
            ttl: Some(Duration::from_secs(3600)),
            max_size: Some(1000),
            cache_completions: true,
            cache_embeddings: true,
            cache_streaming: true,
        }))
        .add_layer(RetryLayer::new(RetryConfig {
            max_attempts: 3,
            initial_backoff: Duration::from_millis(200),
            ..RetryConfig::default()
        }))
        .build();

    scenario_cache_hit_miss(llm.clone()).await?;
    scenario_independent_entries(llm.clone()).await?;
    scenario_ttl_expiry(base.clone() as Arc<dyn LLMProvider>).await?;
    scenario_agent_integration(llm.clone()).await?;

    Ok(())
}
