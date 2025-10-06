# AutoAgents Serve - Logging Guide

## Overview

The AutoAgents serve implementation includes comprehensive logging at all levels to help with debugging, monitoring, and understanding workflow execution.

## Logging Levels

The implementation uses standard Rust logging levels via the `log` crate:

- **ERROR**: Critical errors that prevent execution
- **WARN**: Warning conditions
- **INFO**: General informational messages
- **DEBUG**: Detailed information for debugging
- **TRACE**: Very detailed trace information

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

### CLI Usage

```bash
# Run with info logging
RUST_LOG=info autoagents run -w workflow.yaml -i "query"

# Serve with debug logging
RUST_LOG=debug autoagents serve -w workflow.yaml -p 8080

# Detailed trace logging
RUST_LOG=trace autoagents serve -w workflow.yaml
```

## Server Logging

### Startup Logs

When starting the server, you'll see:

```
[INFO] Initializing AutoAgents HTTP server
[DEBUG] Server configuration: ServerConfig { host: "127.0.0.1", port: 8080 }
[INFO] Loading 1 workflow(s)
[INFO] Loading workflow 'research' from 'workflow.yaml'
[INFO] ✓ Workflow 'research' loaded successfully
[INFO] Total workflows loaded: 1
[INFO]   - research
[INFO] Creating API router with endpoints:
[INFO]   GET  /health
[INFO]   GET  /api/v1/workflows
[INFO]   POST /api/v1/workflows/:name/execute
[INFO] Starting HTTP server on 127.0.0.1:8080
[INFO] Server is ready to accept connections
[INFO] Available endpoints:
[INFO]   - GET  http://127.0.0.1:8080/health
[INFO]   - GET  http://127.0.0.1:8080/api/v1/workflows
[INFO]   - POST http://127.0.0.1:8080/api/v1/workflows/:name/execute
[INFO] 🚀 Server started successfully!
```

### Request Logs

For each HTTP request:

```
[INFO] Executing workflow 'research' with input: 'What is Rust?'
[DEBUG] Workflow 'research' found, starting execution
[INFO] Workflow 'research' completed successfully in 2.34s
[DEBUG] Workflow 'research' output: Single("Rust is a systems programming language...")
```

### Error Logs

When errors occur:

```
[ERROR] Workflow 'nonexistent' not found
[ERROR] Workflow 'research' failed after 1.23s: LLM error: API key not set
```

### Performance Logs

The server includes execution time logging:

```
[INFO] Workflow 'research' completed successfully in 2.34s
```

Response includes `execution_time_ms` field:
```json
{
  "success": true,
  "output": {"Single": "result"},
  "execution_time_ms": 2340
}
```

## Workflow Execution Logs

### Direct Workflow

```
[DEBUG] Creating LLM provider for OpenAI with model gpt-4o
[DEBUG] Creating 1 tool(s)
[DEBUG] Creating dynamic agent 'ResearchWeb'
[DEBUG] Configuring memory with window size: 20
[DEBUG] Building DirectAgent with ReAct executor
[DEBUG] Agent execution started
[INFO] Tool 'brave_search' executed successfully
[DEBUG] Agent execution completed
```

### Sequential Workflow

```
[INFO] Starting sequential workflow with 2 agents
[DEBUG] Building chain agent 0: 'Extractor'
[DEBUG] Building chain agent 1: 'Transformer'
[DEBUG] Publishing initial task to agent_0
[DEBUG] Agent 'Extractor' received task
[DEBUG] Agent 'Extractor' forwarding to agent_1
[DEBUG] Agent 'Transformer' received task
[INFO] Sequential workflow completed
```

### Parallel Workflow

```
[INFO] Starting parallel workflow with 3 agents
[DEBUG] Spawning task for agent 'Analyzer1'
[DEBUG] Spawning task for agent 'Analyzer2'
[DEBUG] Spawning task for agent 'Analyzer3'
[INFO] Waiting for 3 parallel executions
[INFO] Parallel workflow completed in 3.45s
```

### Routing Workflow

```
[INFO] Starting routing workflow
[DEBUG] Router agent processing request
[DEBUG] Router decision: 'booking'
[DEBUG] Publishing to handler_booking
[DEBUG] Handler 'BookingAgent' received task
[INFO] Routing workflow completed
```

## Logging Configuration Examples

### Production (Minimal Logging)

```bash
RUST_LOG=warn autoagents serve -w workflow.yaml
```

Logs only warnings and errors.

### Development (Detailed Logging)

```bash
RUST_LOG=debug autoagents serve -w workflow.yaml
```

Includes debug information for troubleshooting.

### Debugging (Maximum Detail)

```bash
RUST_LOG=trace autoagents serve -w workflow.yaml
```

Includes all trace-level information.

### Module-Specific Logging

```bash
# Debug server, info for everything else
RUST_LOG=info,autoagents_serve::server=debug autoagents serve -w workflow.yaml

# Debug workflows, trace API calls
RUST_LOG=info,autoagents_serve::workflow=debug,autoagents_serve::server::api=trace autoagents serve -w workflow.yaml
```

## HTTP Request Tracing

The server includes HTTP request tracing via `tower-http::trace::TraceLayer`:

### Request Logs (with TRACE level)

```
[TRACE] Started processing request GET /health
[TRACE] Finished processing request GET /health in 0.05ms
[TRACE] Started processing request POST /api/v1/workflows/research/execute
[DEBUG] Health check endpoint called
[INFO] Executing workflow 'research' with input: 'query'
[TRACE] Finished processing request POST /api/v1/workflows/research/execute in 2340ms
```

## Logging Best Practices

### 1. Use Appropriate Levels

```rust
log::error!("Critical failure: {}", error);  // Errors
log::warn!("Deprecated feature used");       // Warnings
log::info!("Workflow completed");            // Info
log::debug!("Agent state: {:?}", state);     // Debug
log::trace!("Variable value: {}", val);      // Trace
```

### 2. Production Deployment

```bash
# Use INFO level in production
RUST_LOG=info autoagents serve -w workflow.yaml

# Or write to file
RUST_LOG=info autoagents serve -w workflow.yaml 2>&1 | tee server.log
```

### 3. Development

```bash
# Use DEBUG during development
RUST_LOG=debug autoagents serve -w workflow.yaml
```

### 4. Troubleshooting

```bash
# Use TRACE when debugging issues
RUST_LOG=trace autoagents serve -w workflow.yaml 2>&1 | tee debug.log
```

## Log Output Examples

### Successful Workflow Execution

```
[2025-10-06T21:30:00Z INFO  autoagents_serve::server] Initializing AutoAgents HTTP server
[2025-10-06T21:30:00Z INFO  autoagents_serve::server] Loading 1 workflow(s)
[2025-10-06T21:30:00Z INFO  autoagents_serve::server] Loading workflow 'research' from 'workflow.yaml'
[2025-10-06T21:30:00Z INFO  autoagents_serve::server] ✓ Workflow 'research' loaded successfully
[2025-10-06T21:30:00Z INFO  autoagents_serve::server] 🚀 Server started successfully!
[2025-10-06T21:30:15Z INFO  autoagents_serve::server::api] Executing workflow 'research' with input: 'What is Rust?'
[2025-10-06T21:30:17Z INFO  autoagents_serve::server::api] Workflow 'research' completed successfully in 2.34s
```

### Failed Workflow Execution

```
[2025-10-06T21:31:00Z INFO  autoagents_serve::server::api] Executing workflow 'research' with input: 'query'
[2025-10-06T21:31:01Z ERROR autoagents_serve::server::api] Workflow 'research' failed after 1.23s: LLM error: OPENAI_API_KEY not set
```

### Workflow Not Found

```
[2025-10-06T21:32:00Z INFO  autoagents_serve::server::api] Executing workflow 'invalid' with input: 'query'
[2025-10-06T21:32:00Z ERROR autoagents_serve::server::api] Workflow 'invalid' not found
```

## Monitoring and Metrics

The server logs include:
- **Execution time**: For performance monitoring
- **Error tracking**: For debugging failures
- **Request logging**: For access tracking

### Example Monitoring Setup

```bash
# Pipe logs to monitoring tool
RUST_LOG=info autoagents serve -w workflow.yaml 2>&1 | \
  grep -E "(ERROR|completed|failed)" | \
  tee -a /var/log/autoagents.log
```

### Extract Performance Metrics

```bash
# Extract execution times
grep "completed successfully" server.log | \
  sed 's/.*in \([0-9.]*\)s/\1/' | \
  awk '{sum+=$1; count++} END {print "Average:", sum/count, "s"}'
```

## Integration with Logging Tools

### JSON Logging (for structured logging)

Add to Cargo.toml:
```toml
env_logger = { version = "0.11", features = ["json"] }
```

### Syslog Integration

```bash
RUST_LOG=info autoagents serve -w workflow.yaml 2>&1 | \
  logger -t autoagents
```

### Cloud Logging

```bash
# AWS CloudWatch
RUST_LOG=info autoagents serve -w workflow.yaml 2>&1 | \
  aws logs put-log-events --log-group-name autoagents ...

# Google Cloud Logging
RUST_LOG=info autoagents serve -w workflow.yaml 2>&1 | \
  gcloud logging write autoagents ...
```

## Summary

The AutoAgents serve implementation provides:
- ✅ Comprehensive logging at all levels
- ✅ HTTP request tracing
- ✅ Performance metrics (execution time)
- ✅ Error tracking with context
- ✅ Flexible log level configuration
- ✅ Production-ready logging infrastructure
