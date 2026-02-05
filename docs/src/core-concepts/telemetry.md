# Telemetry (OpenTelemetry)

AutoAgents ships with an optional OpenTelemetry integration via the `autoagents-telemetry` crate. It consumes the runtime protocol `Event` stream to produce spans and metrics with minimal changes to your executors or agents.

## What gets captured

- Task lifecycle: `TaskStarted`, `TaskComplete`, `TaskError`
- Turn lifecycle: `TurnStarted`, `TurnCompleted`
- Tool calls: `ToolCallRequested`, `ToolCallCompleted`, `ToolCallFailed`

Each span is correlated using `submission_id` + `actor_id` so it works for both direct agents and actor-based runtimes.

## Direct agents

```rust
use autoagents_telemetry::{ExporterConfig, OtlpConfig, TelemetryConfig, attach_to_direct};

let mut telemetry_config = TelemetryConfig::new("my-app");
telemetry_config.exporter = ExporterConfig {
    otlp: Some(OtlpConfig::new("http://localhost:4318")),
    stdout: false,
};

let _telemetry = attach_to_direct(&mut agent_handle, telemetry_config)?;
```

## Actor runtimes

```rust
use autoagents_telemetry::{ExporterConfig, OtlpConfig, TelemetryConfig, attach_to_environment};

let mut telemetry_config = TelemetryConfig::new("my-app");
telemetry_config.exporter = ExporterConfig {
    otlp: Some(OtlpConfig::new("http://localhost:4318")),
    stdout: false,
};

let _telemetry = attach_to_environment(&mut env, None, telemetry_config).await?;
```

## Provider configuration (Langfuse, Honeycomb, Jaeger, etc.)

Most providers accept OTLP over HTTP. Use their OTLP endpoint and pass any required headers (API keys, org IDs, etc.) via `OtlpConfig::headers`.

```rust
use autoagents_telemetry::{ExporterConfig, OtlpConfig, TelemetryConfig};
use std::collections::HashMap;

let mut otlp = OtlpConfig::new("https://provider.example.com/otlp");
otlp.headers = HashMap::from([
    ("x-api-key".to_string(), "YOUR_KEY".to_string()),
]);

let mut config = TelemetryConfig::new("my-app");
config.exporter = ExporterConfig {
    otlp: Some(otlp),
    stdout: false,
};
```

## Redaction

For production safety, you can redact prompts, tool arguments, and tool results:

```rust
use autoagents_telemetry::{RedactionConfig, TelemetryConfig};

let mut config = TelemetryConfig::new("my-app");
config.redaction = RedactionConfig {
    redact_task_inputs: true,
    redact_task_outputs: true,
    redact_tool_arguments: true,
    redact_tool_results: true,
};
```

## Metrics

The telemetry pipeline emits counters and histograms:

- `autoagents.tasks.total`
- `autoagents.tool_calls.total`
- `autoagents.errors.total`
- `autoagents.task.duration.seconds`
- `autoagents.turn.duration.seconds`
- `autoagents.tool.duration.seconds`

Metrics are exported via OTLP when configured.
