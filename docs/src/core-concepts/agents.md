# Agents

Agents are the core building blocks of the AutoAgents framework. They represent autonomous entities that can understand tasks, reason about solutions, execute actions through tools, and provide intelligent responses. This document explores the agent system in detail.

## What is an Agent?

An agent in AutoAgents is a software entity that exhibits the following characteristics:

- **Autonomy**: Can operate independently without constant human intervention
- **Reactivity**: Responds to changes in their environment
- **Proactivity**: Can take initiative to achieve goals
- **Social Ability**: Can interact with other agents and systems
- **Intelligence**: Uses reasoning patterns to solve problems

## Agent Lifecycle

```
Creation → Registration → Task Assignment → Execution → Response → Cleanup
```

### 1. Creation Phase
```rust
use autoagents::core::agent::{AgentBuilder, AgentDeriveT};
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::task::Task;
use autoagents::core::actor::Topic;
use autoagents::core::runtime::SingleThreadedRuntime;
use autoagents::core::tool::{ToolT, ToolCallError};
use autoagents::core::error::Error;
use autoagents_derive::{agent, tool, ToolInput, AgentOutput};
use serde::{Serialize, Deserialize};
use serde_json::Value;

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

impl ToolT for Addition {
    fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let typed_args: AdditionArgs = serde_json::from_value(args)?;
        let result = typed_args.left + typed_args.right;
        Ok(result.into())
    }
}

/// Math agent output with Value and Explanation
#[derive(Debug, Serialize, Deserialize, AgentOutput)]
pub struct MathAgentOutput {
    #[output(description = "The addition result")]
    value: i64,
    #[output(description = "Explanation of the logic")]
    explanation: String,
}

#[agent(
    name = "math_agent",
    description = "You are a Math agent",
    tools = [Addition],
)]
pub struct MathAgent {}

// Agent creation example
let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));
let agent = MathAgent {};
let runtime = SingleThreadedRuntime::new(None);
let topic = Topic::<Task>::new("math_topic");

let agent_instance = AgentBuilder::new(agent)
    .with_llm(llm)
    .runtime(runtime.clone())
    .subscribe_topic(topic.clone())
    .with_memory(sliding_window_memory)
    .build()
    .await?;
```

### 2. Registration Phase
```rust
// Create environment and set up event handling
let mut environment = Environment::new(None);
let _ = environment.register_runtime(runtime.clone()).await;
```

### 3. Task Assignment Phase
```rust
use autoagents::core::agent::task::Task;

// Publish a task to the agent
let task = Task::new("What is 2 + 2?");
runtime.publish(&topic, task).await?;
```

### 4. Execution Phase
```rust
// Run the environment to process tasks
tokio::select! {
    _ = environment.run() => {
        println!("Environment finished running.");
    }
    _ = tokio::signal::ctrl_c() => {
        println!("Ctrl+C detected. Shutting down...");
        environment.shutdown().await;
    }
}
```
