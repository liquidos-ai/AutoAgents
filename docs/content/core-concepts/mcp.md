# MCP (Model Context Protocol)

AutoAgents integrates with [MCP](https://modelcontextprotocol.io) servers through `autoagents-toolkit::mcp`, exposing remote tools as native `ToolT` implementations for agents.

## Trust model

**MCP configuration is a code-execution boundary.**

- **Stdio transport** spawns a host subprocess using the `command`, `args`, `cwd`, and `env` fields from your config. Anyone who can supply or modify that config can run processes as the AutoAgents process user.
- **HTTP / SSE transport** connects to a remote endpoint. The URL and headers define where your agent sends MCP protocol traffic.

Only load MCP configuration from **trusted sources** (your own deployment manifests, admin-controlled files, or explicitly user-approved paths). Do not auto-load project-shipped MCP configs without a trust prompt in production UIs.

## Security policy

By default, `McpSecurityPolicy::secure_default()` enforces:

- A stdio launcher allowlist aligned with production MCP clients (LiteLLM, CrewAI): `npx`, `uvx`, `python`, `python3`, `node`, `docker`, `deno`
- Bare launcher names only (no `/usr/bin/...` paths)
- Dangerous launcher arguments (`-c`, `-e`, `--eval`, shell metacharacters, privileged `docker` flags)
- A denylist of high-risk environment variables (`LD_PRELOAD`, `NODE_OPTIONS`, `PYTHONPATH`, …)
- Working directories and script paths constrained to the config file directory when loading from disk
- SSRF protection for HTTP URLs (blocks localhost, link-local, and private-network hosts)

Extend the allowlist at deploy time (entries must be bare launcher names):

```bash
export AUTOAGENTS_MCP_STDIO_EXTRA_COMMANDS="bun,ruby"
```

For local development and tests only:

```rust
use autoagents_toolkit::mcp::McpSecurityPolicy;

let policy = McpSecurityPolicy::permissive();
let tools = McpTools::from_config_with_policy("config.toml", policy).await?;
```

`permissive()` disables the allowlist and permits private-network HTTP endpoints. **Do not use in production.**

To allow local HTTP MCP servers under `secure_default()`:

```rust
let policy = McpSecurityPolicy::secure_default()
    .with_allow_private_http_endpoints(true);
```

### Approval hook

For commands outside the allowlist, provide an explicit approver. When an approver is configured, config validation defers the allowlist check to spawn time so the hook can run:

```rust
use std::sync::Arc;
use autoagents::async_trait;
use autoagents_toolkit::mcp::{
    McpProcessApprover, McpProcessLaunchSpec, McpSecurityPolicy, McpSecurityError,
};

struct MyApprover;

#[async_trait]
impl McpProcessApprover for MyApprover {
    async fn approve(&self, launch: &McpProcessLaunchSpec) -> Result<(), McpSecurityError> {
        // Present `launch` to the user and require confirmation.
        Ok(())
    }
}

let policy = McpSecurityPolicy::secure_default()
    .with_approver(Arc::new(MyApprover));
```

Stdio commands are validated at **config load** and again immediately **before spawn** (defense in depth).

### Residual risk

Allowlisted launchers such as `npx` and `docker` can still execute code via arguments (for example `npx -y package` or `docker run`). Treat MCP config as trusted input and prefer fixed, reviewed server definitions in production.

## Configuration

### Stdio server

```toml
[[mcp.server]]
name = "echo"
protocol = "stdio"
command = "python3"          # bare launcher name only — no paths
args = ["servers/echo_server.py"]
timeout = 30                 # seconds; applies to connect, list_tools, call_tool

[mcp.server.env]
MY_VAR = "value"

# optional working directory (resolved relative to the config file)
cwd = "."
```

Relative paths in `args` and `cwd` are resolved against the config file directory and must remain inside that directory.

### HTTP server (Streamable HTTP)

```toml
[[mcp.server]]
name = "remote"
protocol = "http"
url = "https://api.example.com/mcp"
timeout = 60

[mcp.server.headers]
Authorization = "Bearer YOUR_TOKEN"
X-Custom = "value"
```

The legacy `sse` protocol value is accepted as an alias for `http` and uses the same Streamable HTTP client (a deprecation warning is logged). Legacy HTTP+SSE-only servers that do not implement Streamable HTTP may not connect.

## Usage

```rust
use autoagents_toolkit::mcp::{McpConfig, McpSecurityPolicy, McpTools};

// secure default
let mcp_tools = McpTools::from_config("config.toml").await?;

// custom policy
let (config, base_dir) = McpConfig::load_from_file("config.toml")?;
config.validate_all(&policy, Some(&base_dir))?;
```

See the [`mcp-example`](../../../examples/mcp/) crate for a full agent integration with a local stdio echo server.

## Timeouts

`timeout` (default: 30 seconds) is enforced for:

1. Server connect / MCP handshake
2. `tools/list`
3. Each `tools/call` through an adapter

Operations that exceed the limit return `McpError::Timeout` with millisecond precision.
