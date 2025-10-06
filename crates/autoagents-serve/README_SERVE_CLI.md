# AutoAgents Serve & CLI - Complete Implementation

## 🎯 Overview

Production-quality implementation of YAML-based workflow execution system for AutoAgents with HTTP REST API and CLI
interface.

## ✨ Key Features

### Workflow Types

- ✅ **Direct**: Single agent execution with DirectAgent
- ✅ **Sequential**: Chain agents using ActorAgent with pub/sub
- ✅ **Parallel**: Concurrent execution with DirectAgent
- ✅ **Routing**: Actor-based routing with conditional handlers

### Configuration

- ✅ **Streaming**: Configurable per workflow
- ✅ **Memory**: Sliding window with configurable size (n_slide/window_size)
- ✅ **Executors**: Basic or ReAct
- ✅ **Structured Output**: JSON schema support for agents
- ✅ **LLM Providers**: OpenAI, Anthropic, Ollama, Groq
- ✅ **Tools**: Extensible tool registry (BraveSearch included)

### HTTP Server

- ✅ REST API with JSON responses
- ✅ Health checks and workflow listing
- ✅ Execution time tracking
- ✅ CORS support
- ✅ HTTP request tracing
- ✅ Comprehensive logging

### CLI

- ✅ Run workflows from YAML
- ✅ Serve workflows over HTTP
- ✅ Environment variable support
- ✅ Detailed logging

## 🚀 Quick Start

### Installation

```bash
# Build serve library
cargo build --package autoagents-serve --features http-serve,search --release

# Build CLI
cargo build --package autoagents-cli --release

# Binary location
./target/release/autoagents
```

### Basic Usage

```bash
# Set up environment
export OPENAI_API_KEY="sk-..."
export BRAVE_SEARCH_API_KEY="BSA..."

# Run a workflow
autoagents run -w workflow.yaml -i "What is Rust?"

# Serve workflows over HTTP
autoagents serve -w workflow.yaml -p 8080 --name research

# With logging
RUST_LOG=info autoagents serve -w workflow.yaml
```

## 📝 YAML Configuration

### Complete Example

```yaml
kind: Direct
name: Research_Workflow
stream: true
description: "AI-powered research workflow"
version: "1.0"

workflow:
  agent:
    name: ResearchWeb
    description: "Web research agent with search capabilities"
    instructions: |
      You are a helpful research agent. Search the web
      for accurate information and provide detailed answers.
    executor: ReAct
    memory:
      kind: sliding_window
      parameters:
        n_slide: 20
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
      - type: brave_search
        options: { }
    output:
      type: structured
      schema:
        type: object
        properties:
          answer: { type: string }
          sources: { type: array, items: { type: string } }
        required: [ answer ]
  output:
    type: structured
```

### Workflow Types

#### Direct Workflow

```yaml
kind: Direct
workflow:
  agent:
    name: agent_name
    # ... agent config
```

#### Sequential Workflow

```yaml
kind: Sequential
workflow:
  agents:
    - name: agent1
      # ... config
    - name: agent2
      # ... config
```

#### Parallel Workflow

```yaml
kind: Parallel
workflow:
  agents:
    - name: parallel1
      # ... config
    - name: parallel2
      # ... config
```

#### Routing Workflow

```yaml
kind: Routing
workflow:
  router:
    name: router_agent
    # ... config
  handlers:
    - condition: "booking"
      agent:
      # ... config
    - condition: "info"
      agent:
      # ... config
```

## 🌐 HTTP API

### Endpoints

#### Execute Workflow

```bash
POST /api/v1/workflows/:name/execute
Content-Type: application/json

{
  "input": "Your query here"
}
```

Response:

```json
{
  "success": true,
  "output": {
    "Single": "Result text"
  },
  "execution_time_ms": 2340,
  "error": null
}
```

#### List Workflows

```bash
GET /api/v1/workflows
```

Response:

```json
{
  "workflows": [
    "research",
    "analysis"
  ]
}
```

#### Health Check

```bash
GET /health
```

Response:

```json
{
  "status": "healthy"
}
```

### cURL Examples

```bash
# Execute workflow
curl -X POST http://localhost:8080/api/v1/workflows/research/execute \
  -H "Content-Type: application/json" \
  -d '{"input": "What is Rust programming?"}'

# List workflows
curl http://localhost:8080/api/v1/workflows

# Health check
curl http://localhost:8080/health
```

## 🔧 Configuration

### Agent Configuration

```yaml
agent:
  name: string              # Required
  description: string       # Required
  instructions: string      # Optional, defaults to description
  executor: ReAct | Basic   # Optional, default: ReAct

  memory: # Optional
    kind: sliding_window
    parameters:
      n_slide: 20          # Or window_size

  model:
    kind: llm
    backend:
      kind: Cloud | Local
      base_url: string     # For local models
    provider: OpenAI | Anthropic | Ollama | Groq
    model_name: string
    parameters:
      temperature: float
      max_tokens: int
      top_p: float
      top_k: int

  tools:
    - type: tool_name
      options: { }

  output:
    type: text | json | structured
    schema: { }             # For structured output
```

### Workflow Configuration

```yaml
kind: Direct | Sequential | Parallel | Routing
name: string               # Optional
stream: bool              # Optional, default: false
description: string       # Optional
version: string          # Optional

workflow:
# Workflow-specific configuration

environment: # Optional
  working_directory: string
  timeout_seconds: int

runtime: # Optional
  type: single_threaded | multi_threaded
  max_concurrent: int
```

## 📊 Logging

### Log Levels

```bash
# Production
RUST_LOG=info autoagents serve -w workflow.yaml

# Development
RUST_LOG=debug autoagents serve -w workflow.yaml

# Debugging
RUST_LOG=trace autoagents serve -w workflow.yaml

# Module-specific
RUST_LOG=info,autoagents_serve::server=debug autoagents serve -w workflow.yaml
```

### Log Output

```
[INFO] Initializing AutoAgents HTTP server
[INFO] Loading workflow 'research' from 'workflow.yaml'
[INFO] ✓ Workflow 'research' loaded successfully
[INFO] 🚀 Server started successfully!
[INFO] Executing workflow 'research' with input: 'query'
[INFO] Workflow 'research' completed successfully in 2.34s
```

## 🔐 Environment Variables

```bash
# LLM Providers
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GROQ_API_KEY="gsk_..."

# Tools
export BRAVE_SEARCH_API_KEY="BSA..."

# Logging
export RUST_LOG=info
```

## 📦 Features

### autoagents-serve

- `default` - Core functionality
- `http-serve` - HTTP REST API
- `search` - Search tools (BraveSearch)
- `cuda` - CUDA support
- `metal` - Metal support (macOS)

### autoagents-cli

- Includes `http-serve` and `search` by default

## 🏗️ Architecture

### Module Structure

```
autoagents-serve/
├── config/          # YAML schema, parser, validator
├── workflow/        # Workflow implementations
│   ├── direct.rs
│   ├── sequential.rs
│   ├── parallel.rs
│   └── routing.rs
├── tools/           # Tool registry
├── server/          # HTTP server (feature-gated)
│   ├── api.rs
│   ├── state.rs
│   └── mod.rs
├── builder.rs       # WorkflowBuilder
├── error.rs         # Error types
└── lib.rs           # Public API
```

### Design Patterns

- **Builder Pattern**: WorkflowBuilder for fluent configuration
- **Factory Pattern**: LLMFactory for provider creation
- **Registry Pattern**: ToolRegistry for tool management
- **Actor Pattern**: ActorAgent for routing and sequential workflows
- **Direct Pattern**: DirectAgent for simple and parallel workflows

## 📚 Examples

### Programmatic Usage

```rust
use autoagents_serve::WorkflowBuilder;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load and build workflow
    let workflow = WorkflowBuilder::from_yaml_file("workflow.yaml")?
        .build()?;

    // Execute workflow
    let result = workflow.run("What is Rust?".to_string()).await?;

    // Handle result
    match result {
        WorkflowOutput::Single(text) => println!("{}", text),
        WorkflowOutput::Multiple(results) => {
            for (i, r) in results.iter().enumerate() {
                println!("Agent {}: {}", i + 1, r);
            }
        }
    }

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
        host: "0.0.0.0".to_string(),
        port: 8080,
    };

    let mut workflows = HashMap::new();
    workflows.insert("research".to_string(), "workflow.yaml".to_string());

    serve(config, workflows).await?;
    Ok(())
}
```

## 🧪 Testing

```bash
# Build all packages
cargo build --package autoagents-serve --features http-serve,search
cargo build --package autoagents-cli
cargo build --package serve

# Run example
cargo run --package serve

# Test CLI
cargo run --package autoagents-cli -- run -w workflow.yaml -i "test"
cargo run --package autoagents-cli -- serve -w workflow.yaml -p 8080
```

## 📖 Documentation

- [Implementation Summary](IMPLEMENTATION_SUMMARY.md)
- [Features Summary](FEATURES_SUMMARY.md)
- [Logging Guide](LOGGING_GUIDE.md)
- [Library README](README.md)
- [CLI README](../autoagents-cli/README.md)

## 🎯 Key Achievements

✅ Full YAML-based configuration
✅ Multiple workflow patterns (Direct, Sequential, Parallel, Routing)
✅ Actor-based routing with pub/sub
✅ Structured output support
✅ Streaming capability
✅ HTTP REST API with detailed logging
✅ CLI interface
✅ Memory configuration
✅ Executor selection (Basic/ReAct)
✅ Production-quality error handling
✅ Comprehensive logging with tracing
✅ Performance metrics (execution time)
✅ Extensible architecture

## 🚀 Production Deployment

```bash
# Build release binary
cargo build --package autoagents-cli --release

# Run with systemd
cat > /etc/systemd/system/autoagents.service << 'SYSTEMD'
[Unit]
Description=AutoAgents Workflow Server
After=network.target

[Service]
Type=simple
User=autoagents
WorkingDirectory=/opt/autoagents
Environment="RUST_LOG=info"
Environment="OPENAI_API_KEY=sk-..."
ExecStart=/opt/autoagents/bin/autoagents serve -w /opt/autoagents/workflows/main.yaml -p 8080
Restart=always

[Install]
WantedBy=multi-user.target
SYSTEMD

# Start service
systemctl enable autoagents
systemctl start autoagents
```

## 📄 License

MIT OR Apache-2.0
