# Your First Agent

This guide creates a minimal ReAct agent that can call a tool and return a structured answer.

## 1) Dependencies

```toml
[dependencies]
autoagents = { version = "0.3.4", features = ["openai"] }
autoagents-derive = "0.3.4"
# Optional if you want ready-made tools
autoagents-toolkit = { version = "0.3.4", features = ["filesystem", "search"] }
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

## 2) Define a Tool

```rust
use autoagents::async_trait;
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents_derive::{tool, ToolInput};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize, ToolInput, Debug)]
struct AddArgs { left: i64, right: i64 }

#[tool(name = "addition", description = "Add two numbers", input = AddArgs)]
struct Addition;

#[async_trait]
impl ToolRuntime for Addition {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let a: AddArgs = serde_json::from_value(args)?;
        Ok((a.left + a.right).into())
    }
}
```

## 3) Define Output (optional)

If you want type‑safe structured output, derive `AgentOutput` and convert from executor output.

```rust
use autoagents_derive::AgentOutput;
#[derive(Debug, Serialize, Deserialize, AgentOutput)]
struct MathOut {
    #[output(description = "The result value")] value: i64,
    #[output(description = "Short explanation")] explanation: String,
}
```

## 4) Define the Agent

```rust
use autoagents_derive::{agent, AgentHooks};
use autoagents::core::agent::prebuilt::executor::{ReActAgent, ReActAgentOutput};

#[agent(
    name = "math_agent",
    description = "Solve basic math using tools and return JSON",
    tools = [Addition],
    output = MathOut
)]
#[derive(Clone, AgentHooks, Default)]
struct MathAgent;

impl From<ReActAgentOutput> for MathOut {
    fn from(out: ReActAgentOutput) -> Self {
        serde_json::from_str(&out.response).unwrap_or(MathOut { value: 0, explanation: out.response })
    }
}
```

## 5) Build LLM and Run

```rust
use autoagents::core::agent::{AgentBuilder, DirectAgent};
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::task::Task;
use autoagents::llm::builder::LLMBuilder;
use autoagents::llm::backends::openai::OpenAI;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o")
        .build()?;

    let agent = ReActAgent::new(MathAgent);
    let handle = AgentBuilder::<_, DirectAgent>::new(agent)
        .llm(llm)
        .memory(Box::new(SlidingWindowMemory::new(10)))
        .build()
        .await?;

    let out = handle.agent.run(Task::new("Add 20 and 5 and explain"))
        .await?;
    println!("{:?}", out);
    Ok(())
}
```

## Environment Variables

Set provider keys as needed:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export OPENROUTER_API_KEY=...
export GROQ_API_KEY=...
export AZURE_OPENAI_API_KEY=...
```

For Brave Search (toolkit): `BRAVE_SEARCH_API_KEY` or `BRAVE_API_KEY`.

That’s it — you’ve built a ReAct agent with a tool and structured output.
