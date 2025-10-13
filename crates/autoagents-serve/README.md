# AutoAgents Serve

A library for running AutoAgents workflows from YAML configurations with support for multiple workflow patterns and HTTP REST API serving.

## Features

- **Multiple Workflow Types**:
  - **Direct**: Single agent execution
  - **Sequential**: Chain of agents (Actor-based with pub/sub)
  - **Parallel**: Concurrent agent execution
  - **Routing**: Actor-based routing with conditional handlers

- **Builder Pattern**: Fluent API for constructing workflows
- **LLM Provider Support**: OpenAI, Anthropic, Ollama, Groq, and more
- **Tool Integration**: Extensible tool system with built-in support
- **HTTP Server**: Optional REST API for serving workflows (feature: `http-serve`)
- **Type Safety**: Strong typing with Rust enums and validation

## Quick Start

### Basic Usage

```rust
use autoagents_serve::WorkflowBuilder;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let workflow = WorkflowBuilder::from_yaml_file("workflow.yaml")?
        .build()?;

    let result = workflow.run("Your input here".to_string()).await?;
    println!("Result: {:?}", result);
    Ok(())
}
```

### HTTP Server

```rust
use autoagents_serve::{serve, ServerConfig};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: 8080,
    };

    let mut workflows = HashMap::new();
    workflows.insert("my_workflow".to_string(), "workflow.yaml".to_string());

    serve(config, workflows).await?;
    Ok(())
}
```

## YAML Configuration

### Direct Workflow

```yaml
kind: Direct
name: SimpleCalculator
stream: false
description: "Simple calculator agent for basic math"

workflow:
  agent:
    name: MathAgent
    description: "A helpful math calculator agent"
    instructions: |
      You are a math expert. Solve the given math problem step by step.
      Provide a clear and concise answer.
    executor: ReAct
    memory:
      kind: sliding_window
      parameters:
        window_size: 10
    model:
      kind: llm
      backend:
        kind: Cloud
      provider: OpenAI
      model_name: gpt-4o-mini
      parameters:
        temperature: 0.1
        max_tokens: 500
    tools: []
    output:
      type: text
  output:
    type: text
```

## Architecture

### Workflow Types

- **Direct**: Uses `DirectAgent` for simple, synchronous execution
- **Sequential**: Uses `ActorAgent` with topic-based pub/sub for chaining
- **Parallel**: Uses `DirectAgent` with tokio tasks for concurrent execution
- **Routing**: Uses `ActorAgent` for routing and handler delegation

### LLM Providers

Configure LLM providers through environment variables:

- `OPENAI_API_KEY` - For OpenAI
- Base URLs can be configured in YAML for local providers (Ollama)

Use Mistral-rs for running llm on local with CUDA, Metal

### Tools

Tools are configured in the YAML and instantiated through the `ToolRegistry`:

- `brave_search` - Web search (requires `BRAVE_SEARCH_API_KEY`)
- Extensible for custom tools

## REST API

When using the `http-serve` feature, the following endpoints are available:

- `GET /health` - Health check
- `GET /api/v1/workflows` - List loaded workflows
- `POST /api/v1/workflows/:name/execute` - Execute a workflow

Example request:

```bash
curl -X POST http://localhost:8080/api/v1/workflows/my_workflow/execute \
  -H "Content-Type: application/json" \
  -d '{"input": "What is Rust?"}'
```

## Features

- `default` - Core functionality only
- `http-serve` - Enable HTTP REST API server
- `search` - Enable search tools (Brave Search)
- `cuda` - CUDA support for local models
- `metal` - Metal support for local models on macOS

## Examples

See `examples/serve/` for a complete working example.

## Setting Log Level

### Using Environment Variable

```bash
# Set log level for all components
export RUST_LOG=info

# Set different levels for different modules
export RUST_LOG=autoagents_serve=debug,autoagents_cli=info

# Enable debug for server, info for everything else
export RUST_LOG=debug,autoagents_serve::server=trace
```
