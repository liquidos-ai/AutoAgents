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

impl ToolRuntime for Addition {
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
    executor = ReActExecutor,
    output = MathAgentOutput
)]
pub struct MathAgent {}


let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));
let agent = MathAgent {};
let runtime = SingleThreadedRuntime::new(None);
let agent = AgentBuilder::new(agent)
        .with_llm(llm)
        .runtime(runtime.clone())
        .subscribe_topic("test")
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
runtime
        .publish_message("What is 2 + 2?".into(), "test".into())
        .await
        .unwrap();
```

### 4. Execution Phase
```rust
environment.run();
```
