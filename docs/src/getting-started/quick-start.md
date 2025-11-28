# Quick Start

## Installation

Before using AutoAgents, ensure you have:

- **Rust 1.91.1 or later** - Install using [rustup](https://rustup.rs/)
- **Cargo** package manager (comes with Rust)

Verify your installation:
```bash
rustc --version
cargo --version
```

## Create a Rust Project
```shell
cargo new MyAgent
cd MyAgent
```

## Add Dependencies

Add the core crate and macros. Enable a provider feature (e.g., `openai`) to use that backend.

```toml
# Cargo.toml
[dependencies]
autoagents = { version = "0.3.0", features = ["openai"] }
autoagents-derive = "0.3.0"
```

Optional tools (filesystem, search) live in `autoagents-toolkit`:

```toml
autoagents-toolkit = { version = "0.3.0", features = ["filesystem", "search"] }
```

Provider features available on `autoagents`: `openai`, `anthropic`, `openrouter`, `groq`, `google`, `azure_openai`, `xai`, `deepseek`, `ollama`. Use only what you need.

## Minimal Agent

A single‑turn agent using the Basic executor:

```rust
use autoagents::core::agent::prebuilt::executor::BasicAgent;
use autoagents::core::agent::{AgentBuilder, DirectAgent};
use autoagents::core::agent::task::Task;
use autoagents::llm::builder::LLMBuilder;
use autoagents::llm::backends::openai::OpenAI;
use autoagents_derive::{agent, AgentHooks};
use std::sync::Arc;

#[agent(name = "hello", description = "Helpful assistant")]
#[derive(Clone, AgentHooks, Default)]
struct HelloAgent;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Set OPENAI_API_KEY in your env
    let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o")
        .build()?;

    let agent = BasicAgent::new(HelloAgent);
    let handle = AgentBuilder::<_, DirectAgent>::new(agent)
        .llm(llm)
        .build()
        .await?;

    let out = handle.agent.run(Task::new("Say hi in one short sentence"))
        .await?;
    println!("{}", String::from(out));
    Ok(())
}
```

## Run

```bash
cargo run
```

Tip: for a ReAct multi‑turn agent with tools and streaming, see the examples in `examples/basic` and `examples/coding_agent`.
