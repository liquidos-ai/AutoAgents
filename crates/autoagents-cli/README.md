# AutoAgents CLI

Command-line interface for running and serving AutoAgents workflows.

## Installation

```bash
cargo build --package autoagents-cli --release
```

The binary will be available at `target/release/autoagents`.

## Usage

### Run a Workflow

Execute a workflow from a YAML file:

```bash
autoagents run --workflow workflow.yaml --input "What is Rust?"
```

### Serve Workflows over HTTP

Start an HTTP server to serve workflows via REST API:

```bash
autoagents serve --workflow workflow.yaml --port 8080
```

Optional arguments:

- `--name <NAME>` - Custom name for the workflow (defaults to filename)
- `--host <HOST>` - Host to bind to (default: 127.0.0.1)
- `--port <PORT>` - Port to bind to (default: 8080)

### Examples

```bash
# Run a direct workflow
autoagents run -w examples/serve/workflow.yaml -i "Tell me about AI"

# Serve a workflow on custom port
autoagents serve -w workflow.yaml -p 9000 --name research

# Serve with custom name
autoagents serve -w workflow.yaml --name my_agent --host 0.0.0.0 --port 3000
```

## REST API Endpoints

When using the `serve` command, the following endpoints are available:

### Health Check

```bash
GET /health
```

### List Workflows

```bash
GET /api/v1/workflows
```

### Execute Workflow

```bash
POST /api/v1/workflows/:name/execute
Content-Type: application/json

{
  "input": "Your query here"
}
```

Example:

```bash
curl -X POST http://localhost:8080/api/v1/workflows/research/execute \
  -H "Content-Type: application/json" \
  -d '{"input": "What is Rust programming language?"}'
```

## Environment Variables

Set up environment variables for LLM providers:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GROQ_API_KEY="your-key"
export BRAVE_SEARCH_API_KEY="your-key"  # For search tools
```

## With Local Model Acceleration

```shell
cargo run -p autoagents-cli --release --features cuda serve --directory ./examples/serve/workflows
```
