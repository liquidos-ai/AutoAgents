# Your First Agent

In this guide, we'll walk through creating your first AutoAgents agent step by step, explaining each concept in detail. By the end, you'll have a solid understanding of how agents work and how to build them effectively.

## Understanding Agents

An agent in AutoAgents is an autonomous entity that can:
- **Think**: Process information and make decisions
- **Act**: Execute tools and functions
- **Remember**: Maintain context across interactions
- **Communicate**: Provide structured responses

## Agent Anatomy

Every AutoAgents agent consists of:

1. **Agent Struct**: The core definition of your agent
2. **Tools**: Functions the agent can call
3. **Executor**: The reasoning pattern (ReAct, Chain-of-Thought, etc.)
4. **Memory**: Context management system
5. **LLM Provider**: The language model backend

## Step-by-Step Guide

### Step 1: Project Setup

First, create a new Rust project:

```bash
cargo new my-first-agent
cd my-first-agent
```

Add dependencies to `Cargo.toml`:

```toml
[dependencies]
autoagents = { version = "0.2.0-alpha.0", features = ["openai"] }
autoagents-derive = "0.2.0-alpha.0"
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
uuid = { version = "1.0", features = ["v4"] }
```

### Step 2: Define Tool Input Structure

Tools need structured inputs. Let's create a calculator tool:

```rust
use autoagents_derive::ToolInput;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct CalculatorArgs {
    #[input(description = "First number")]
    a: f64,
    #[input(description = "Second number")]
    b: f64,
    #[input(description = "Operation: add, subtract, multiply, divide")]
    operation: String,
}
```

The `#[input]` attribute provides descriptions that help the LLM understand how to use the tool.

### Step 3: Implement the Tool

Tools are functions that agents can call:

```rust
use autoagents::llm::{ToolCallError, ToolT};
use autoagents_derive::tool;

#[tool(
    name = "CalculatorTool",
    description = "Performs basic arithmetic operations",
    input = CalculatorArgs,
)]
fn calculate(args: CalculatorArgs) -> Result<String, ToolCallError> {
    let result = match args.operation.as_str() {
        "add" => args.a + args.b,
        "subtract" => args.a - args.b,
        "multiply" => args.a * args.b,
        "divide" => {
            if args.b == 0.0 {
                return Err(ToolCallError::RuntimeError(
                    "Division by zero".into()
                ));
            }
            args.a / args.b
        }
        _ => return Err(ToolCallError::RuntimeError(
            format!("Unknown operation: {}", args.operation).into()
        )),
    };
    
    Ok(format!("{} {} {} = {}", args.a, args.operation, args.b, result))
}
```

### Step 4: Define Your Agent

Now let's create the agent itself:

```rust
use autoagents::core::agent::{AgentDeriveT, ReActExecutor};
use autoagents_derive::agent;

#[agent(
    name = "math_assistant",
    description = "A helpful assistant that can perform mathematical calculations",
    tools = [CalculatorTool],
    executor = ReActExecutor,
)]
pub struct MathAssistant {}
```

The agent macro generates all the boilerplate code needed to integrate with the framework.

### Step 5: Add Structured Output (Optional)

For type-safe responses, define output structures:

```rust
use autoagents_derive::ToolInput;

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct MathResult {
    #[output(description = "The calculation result")]
    result: f64,
    #[output(description = "The calculation expression")]
    expression: String,
    #[output(description = "Explanation of the calculation")]
    explanation: String,
}

#[agent(
    name = "math_assistant",
    description = "A helpful assistant that can perform mathematical calculations",
    tools = [CalculatorTool],
    executor = ReActExecutor,
    output = MathResult,
)]
pub struct MathAssistant {}
```

### Step 6: Complete Implementation

Here's the complete `main.rs`:

```rust
use autoagents::core::agent::base::AgentBuilder;
use autoagents::core::agent::{AgentDeriveT, ReActExecutor};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::memory::SlidingWindowMemory;
use autoagents::llm::{LLMProvider, OpenAIProvider, ToolCallError, ToolInputT, ToolT};
use autoagents_derive::{agent, tool, ToolInput};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// Tool input structure
#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct CalculatorArgs {
    #[input(description = "First number")]
    a: f64,
    #[input(description = "Second number")]
    b: f64,
    #[input(description = "Operation: add, subtract, multiply, divide")]
    operation: String,
}

// Tool implementation
#[tool(
    name = "CalculatorTool",
    description = "Performs basic arithmetic operations",
    input = CalculatorArgs,
)]
fn calculate(args: CalculatorArgs) -> Result<String, ToolCallError> {
    let result = match args.operation.as_str() {
        "add" => args.a + args.b,
        "subtract" => args.a - args.b,
        "multiply" => args.a * args.b,
        "divide" => {
            if args.b == 0.0 {
                return Err(ToolCallError::RuntimeError(
                    "Division by zero".into()
                ));
            }
            args.a / args.b
        }
        _ => return Err(ToolCallError::RuntimeError(
            format!("Unknown operation: {}", args.operation).into()
        )),
    };
    
    Ok(format!("{} {} {} = {}", args.a, args.operation, args.b, result))
}

// Output structure
#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct MathResult {
    #[output(description = "The calculation result")]
    result: f64,
    #[output(description = "The calculation expression")]
    expression: String,
    #[output(description = "Explanation of the calculation")]
    explanation: String,
}

// Agent definition
#[agent(
    name = "math_assistant",
    description = "A helpful assistant that can perform mathematical calculations and explain the process",
    tools = [CalculatorTool],
    executor = ReActExecutor,
    output = MathResult,
)]
pub struct MathAssistant {}

#[tokio::main]
async fn main() -> Result<(), Error> {
    println!("🧮 Math Assistant Demo\n");

    // Initialize LLM provider
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable must be set");
    
    let llm = Arc::new(OpenAIProvider::new(&api_key)?);

    // Create agent with memory
    let memory = Box::new(SlidingWindowMemory::new(5));
    let math_agent = MathAssistant {};

    let agent = AgentBuilder::new(math_agent)
        .with_llm(llm)
        .with_memory(memory)
        .build()?;

    // Set up environment
    let mut environment = Environment::new(None).await;
    let agent_id = environment.register_agent(agent, None).await?;

    // Add mathematical tasks
    let tasks = vec![
        "Calculate 15 + 27",
        "What's 144 divided by 12?",
        "Multiply 8 by 9, then add 5",
        "Calculate the area of a rectangle with length 12 and width 8",
    ];

    for task in tasks {
        environment.add_task(agent_id, task).await?;
    }

    // Execute all tasks
    println!("Executing mathematical tasks...\n");
    let results = environment.run_all(agent_id, None).await?;

    // Display results
    for (i, result) in results.iter().enumerate() {
        println!("Task {}: ", i + 1);
        if let Some(output) = &result.output {
            println!("{}\n", output);
        }
    }

    // Interactive mode
    println!("🎯 Interactive Mode - Enter calculations (type 'quit' to exit):");
    
    let mut input = String::new();
    loop {
        print!(">> ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        
        input.clear();
        if std::io::stdin().read_line(&mut input).is_ok() {
            let task = input.trim();
            if task == "quit" {
                break;
            }
            
            if !task.is_empty() {
                environment.add_task(agent_id, task).await?;
                let results = environment.run_all(agent_id, None).await?;
                
                if let Some(result) = results.last() {
                    if let Some(output) = &result.output {
                        println!("{}\n", output);
                    }
                }
            }
        }
    }

    // Cleanup
    environment.shutdown().await;
    println!("👋 Goodbye!");
    
    Ok(())
}
```

### Step 7: Run Your Agent

Set your API key and run:

```bash
export OPENAI_API_KEY="your-api-key"
cargo run
```

## Understanding the Flow

### 1. Agent Registration
```rust
let agent_id = environment.register_agent(agent, None).await?;
```
The agent is registered with the environment, receiving a unique ID.

### 2. Task Addition
```rust
environment.add_task(agent_id, "Calculate 15 + 27").await?;
```
Tasks are queued for the agent to process.

### 3. Execution
```rust
let results = environment.run_all(agent_id, None).await?;
```
The agent processes all tasks using the ReAct pattern:
- **Thought**: Analyzes the task
- **Action**: Calls the calculator tool
- **Observation**: Reviews the result
- **Response**: Provides the final answer

## Advanced Features

### Multiple Tools
```rust
#[agent(
    name = "advanced_math",
    tools = [CalculatorTool, StatisticsTool, GeometryTool],
    executor = ReActExecutor,
)]
pub struct AdvancedMathAgent {}
```

### Custom Memory
```rust
use autoagents::core::memory::PersistentMemory;

let memory = Box::new(PersistentMemory::new("./agent_memory.json")?);
```

### Error Handling
```rust
match environment.run_all(agent_id, None).await {
    Ok(results) => {
        // Process results
    }
    Err(e) => {
        eprintln!("Error: {}", e);
        // Handle error
    }
}
```

## Best Practices

### 1. Clear Tool Descriptions
```rust
#[tool(
    name = "CalculatorTool",
    description = "Performs basic arithmetic operations. Use this when you need to calculate mathematical expressions.",
    input = CalculatorArgs,
)]
```

### 2. Robust Error Handling
```rust
fn calculate(args: CalculatorArgs) -> Result<String, ToolCallError> {
    // Validate inputs
    if args.operation.is_empty() {
        return Err(ToolCallError::InvalidInput("Operation cannot be empty".into()));
    }
    
    // Handle edge cases
    if args.operation == "divide" && args.b == 0.0 {
        return Err(ToolCallError::RuntimeError("Division by zero".into()));
    }
    
    // Perform calculation
    // ...
}
```

### 3. Meaningful Agent Descriptions
```rust
#[agent(
    name = "math_assistant",
    description = "A precise mathematical assistant that can perform calculations, explain mathematical concepts, and help solve complex problems step by step.",
    tools = [CalculatorTool],
    executor = ReActExecutor,
)]
```

## Common Patterns

### State Management
```rust
use std::sync::{Arc, Mutex};

struct CalculatorState {
    history: Vec<String>,
}

lazy_static! {
    static ref STATE: Arc<Mutex<CalculatorState>> = Arc::new(Mutex::new(CalculatorState {
        history: Vec::new(),
    }));
}
```

### Async Tools
```rust
#[tool(
    name = "AsyncCalculatorTool",
    description = "Async calculator tool",
    input = CalculatorArgs,
)]
async fn async_calculate(args: CalculatorArgs) -> Result<String, ToolCallError> {
    // Async operations
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    // Calculation logic
    Ok(format!("Result: {}", args.a + args.b))
}
```

## Debugging Tips

### 1. Enable Logging
```bash
RUST_LOG=debug cargo run
```

### 2. Tool Call Inspection
```rust
#[tool(
    name = "DebugTool",
    description = "Debug tool",
    input = DebugArgs,
)]
fn debug_tool(args: DebugArgs) -> Result<String, ToolCallError> {
    println!("Tool called with: {:?}", args);
    // Tool logic
    Ok("Debug result".to_string())
}
```

### 3. Memory State
```rust
// Check memory state
if let Some(memory) = agent.memory() {
    let messages = memory.get_messages();
    println!("Memory contains {} messages", messages.len());
}
```

## Next Steps

Now that you've created your first agent, explore:

1. **[Core Concepts](../core-concepts/architecture.md)** - Deep dive into the architecture
2. **[Custom Tools](../agent-development/custom-tools.md)** - Build more sophisticated tools
3. **[Memory Systems](../core-concepts/memory.md)** - Advanced memory management
4. **[Multi-Agent Systems](../advanced/multi-agent.md)** - Coordinate multiple agents

## Troubleshooting

### Agent Not Calling Tools
- Check tool descriptions are clear
- Verify tool inputs match expected format
- Ensure tools are properly registered

### Memory Issues
- Monitor memory usage in long-running agents
- Use appropriate memory backends
- Clear memory when needed

### API Rate Limits
- Implement backoff strategies
- Use local models for development
- Cache responses when appropriate

Congratulations! You've successfully created your first AutoAgents agent. The framework provides powerful abstractions while maintaining flexibility for complex use cases. Happy building! 🚀