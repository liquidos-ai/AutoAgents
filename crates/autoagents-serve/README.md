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

workflow:
  agent:
    name: ResearchAgent
    description: A research agent for web searches
    instructions: |
      You are a helpful research agent. Search the internet
      for accurate information and provide concise answers.
    model:
      kind: llm
      backend:
        kind: Cloud
      provider: OpenAI
      model_name: gpt-4o
      parameters:
        temperature: 0.2
        max_tokens: 1024
    tools:
      - name: brave_search
        options: {}
    output:
      type: text
```

### Sequential Workflow

```yaml
kind: Sequential

workflow:
  agents:
    - name: ExtractorAgent
      description: Extract specifications from text
      model:
        kind: llm
        backend:
          kind: Cloud
        provider: OpenAI
        model_name: gpt-4
      tools: []

    - name: TransformerAgent
      description: Transform specs to JSON
      model:
        kind: llm
        backend:
          kind: Cloud
        provider: OpenAI
        model_name: gpt-4
      tools: []
```

### Parallel Workflow

```yaml
kind: Parallel

workflow:
  agents:
    - name: Summarizer1
      description: Summarize from perspective A
      model:
        kind: llm
        backend:
          kind: Cloud
        provider: OpenAI
        model_name: gpt-4
      tools: []

    - name: Summarizer2
      description: Summarize from perspective B
      model:
        kind: llm
        backend:
          kind: Cloud
        provider: OpenAI
        model_name: gpt-4
      tools: []
```

### Routing Workflow

```yaml
kind: Routing

workflow:
  router:
    name: RouterAgent
    description: |
      Route requests to appropriate handler.
      Output 'booker' for booking requests, 'info' for information.
    model:
      kind: llm
      backend:
        kind: Cloud
      provider: OpenAI
      model_name: gpt-4
    tools: []

  handlers:
    - condition: booker
      agent:
        name: BookingAgent
        description: Handle booking requests
        model:
          kind: llm
          backend:
            kind: Cloud
          provider: OpenAI
          model_name: gpt-4
        tools: []

    - condition: info
      agent:
        name: InfoAgent
        description: Provide information
        model:
          kind: llm
          backend:
            kind: Cloud
          provider: OpenAI
          model_name: gpt-4
        tools: []
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
- `ANTHROPIC_API_KEY` - For Anthropic
- `GROQ_API_KEY` - For Groq
- Base URLs can be configured in YAML for local providers (Ollama)

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
