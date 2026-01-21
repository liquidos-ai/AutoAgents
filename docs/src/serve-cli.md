# Serve and CLI

AutoAgents includes a serving library (`autoagents-serve`) and a CLI (`autoagents-cli`) for running agent workflows defined in YAML.

Repository: https://github.com/liquidos-ai/AutoAgents-CLI

To install the CLI, clone the AutoAgents repository and run:

```rs
cargo install --path ./crates/autoagents-cli.
```

## CLI

Binary: `autoagents`

Subcommands:

- `run --workflow <file> --input <text>`
  - Loads a workflow YAML and executes it once with the given input.
- `serve [--workflow <file> | --directory <dir>] [--name <name>] [--host <host>] [--port <port>]`
  - Serves one or many workflows over HTTP.

Examples:

```bash
# Run a single workflow once
OPENAI_API_KEY=... cargo run -p autoagents-cli -- run --workflow examples/serve/workflows/direct.yaml --input "2 + 2"

# Serve a single workflow
OPENAI_API_KEY=... cargo run -p autoagents-cli -- serve --workflow examples/serve/workflows/direct.yaml --host 127.0.0.1 --port 8080

# Serve all workflows in a directory
OPENAI_API_KEY=... cargo run -p autoagents-cli -- serve --directory examples/serve/workflows --host 127.0.0.1 --port 8080
```

## Workflow YAML

Top-level fields:

- `kind`: `Direct` | `Sequential` | `Parallel` | `Routing`
- `name`: Optional workflow name
- `description`: Optional description
- `version`: Optional version
- `stream`: `true`/`false` to enable streaming
- `memory_persistence`: Optional persistence policy `{ mode: "memory" | "file" }`
- `workflow`: Workflow spec (see below)
- `environment`: Optional `{ working_directory, timeout_seconds }`
- `runtime`: Optional `{ type: "single_threaded", max_concurrent }`

Workflow spec variants:

- `Direct`:
  - `workflow.agent`: Single agent to run
  - `workflow.output`: Output type
- `Sequential`/`Parallel`:
  - `workflow.agents`: List of agents to run (in order or concurrently)
  - `workflow.output`: Output type
- `Routing`:
  - `workflow.router`: Router agent
  - `workflow.handlers`: `[{ condition, agent }, ...]` routing targets
  - `workflow.output`: Output type

### Agent

```yaml
agent:
  name: MathAgent
  description: "Helpful math agent"
  instructions: |
    You are a math expert. Solve the problem step by step.
  executor: ReAct   # or Basic
  memory:
    kind: sliding_window
    parameters:
      window_size: 10  # or `n_slide`
  model:
    kind: llm
    backend:
      kind: Cloud       # or Local
      base_url: ...     # optional
    provider: OpenAI    # e.g., OpenAI, Anthropic, Groq, OpenRouter, Ollama
    model_name: gpt-4o-mini
    parameters:
      temperature: 0.1
      max_tokens: 500
      top_p: 0.95
      top_k: 40
  tools: [ ]           # list of tool names (see Tools)
  output:
    type: text         # text | json | structured
```

### Tools

Tool entries are resolved by the server’s registry; for built-in toolkits (e.g., search), enable features like `search-tools` on `autoagents-serve`.

```yaml
tools:
  - name: brave_search
    options: { }
    config: { }
```

### Output

Output config allows specifying the format:

- `type: text` — plain text
- `type: json` — JSON value
- `type: structured` — JSON Schema via `schema`

### Examples

Study the sample workflows in `examples/serve/workflows/`:

- Direct: `examples/serve/workflows/direct.yaml`
- Sequential: `examples/serve/workflows/sequential.yaml`
- Parallel: `examples/serve/workflows/parallel.yaml`
- Routing: `examples/serve/workflows/routing.yaml`
- Research (with search tool): `examples/serve/workflows/research.yaml`

## Library Usage (Serve)

For embedding in your app, use `WorkflowBuilder` and (optionally) `HTTPServer` from `autoagents-serve`:

```rust
use autoagents_serve::{WorkflowBuilder, HTTPServer, ServerConfig};

// Build and run in-process
let built = WorkflowBuilder::from_yaml_file("examples/serve/workflows/direct.yaml")?
    .build()?;
let result = built.run("2 + 2".to_string()).await?;

// Serve over HTTP
let server = HTTPServer::new(ServerConfig { host: "127.0.0.1".into(), port: 8080 },
                             std::collections::HashMap::from([
                               ("calc".into(), "examples/serve/workflows/direct.yaml".into())
                             ]));
server.serve().await?;
```

See crate docs for endpoint details and streaming support.
