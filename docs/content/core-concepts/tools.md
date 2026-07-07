# Tools

Tools let agents act on the world. Each tool provides a name, description, JSON schema for its input, and an async `execute` implementation.

## Custom Tools

Define tool inputs and the tool with derive macros, and implement `ToolRuntime`:

```rust
use autoagents::async_trait;
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents_derive::{tool, ToolInput};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize, ToolInput, Debug)]
struct AddArgs { left: i64, right: i64 }

#[tool(name = "add", description = "Add two integers", input = AddArgs)]
struct Add;

#[async_trait]
impl ToolRuntime for Add {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let a: AddArgs = serde_json::from_value(args)?;
        Ok(serde_json::json!({ "result": a.left + a.right }))
    }
}
```

Attach tools in the `#[agent(..., tools = [ .. ])]` macro. Tools can also be built dynamically; when sharing `Arc<dyn ToolT>` across agents use `shared_tools_to_boxes`.

## Toolkit

Reusable tools are in `autoagents-toolkit`:

- Filesystem tools: `ListDir`, `ReadFile`, `WriteFile`, `CopyFile`, `MoveFile`, `DeleteFile`, `SearchFile` (feature: `filesystem`)
- Web search: `BraveSearch` (feature: `search`, requires `BRAVE_SEARCH_API_KEY` or `BRAVE_API_KEY`)

Enable features in your `Cargo.toml` as needed.

Filesystem tools should be scoped to an explicit workspace root when exposed to agents:

```rust
use autoagents_toolkit::tools::filesystem::{DeleteFile, ListDir, ReadFile, WriteFile};

let workspace_root = std::env::current_dir()?.canonicalize()?;

let read_file = ReadFile::new_with_root_dir(workspace_root.display().to_string());
let write_file = WriteFile::new_with_root_dir(workspace_root.display().to_string());
let delete_file = DeleteFile::new_with_root_dir(workspace_root.display().to_string());
let list_dir = ListDir::new_with_root_dir(workspace_root.display().to_string());
```

Root-scoped tools reject absolute paths, `..` traversal outside the root, and symlink escapes. Directory deletion is non-recursive by default; pass `recursive: true` only when recursive deletion is explicitly intended.

The filesystem constructors named `new()` are also sandboxed to the process current directory by default. Use `new_unrestricted()` only for trusted local workflows where unrestricted host filesystem access is intentional.

## MCP

Model Context Protocol (MCP) integrations are available via `autoagents-toolkit::mcp` — load tool definitions from MCP servers and expose them as `ToolT`. AutoAgents supports local stdio servers and remote Streamable HTTP servers.

Local MCP servers execute host processes. Treat MCP config as trusted code and pass an explicit `McpProcessPolicy` before local servers can run:

```rust
use autoagents_toolkit::mcp::{McpProcessPolicy, McpTools};

let server_bin = std::fs::canonicalize("./mcp-servers/my-server")?;
let process_policy = McpProcessPolicy::allow_paths([server_bin])?;
let mcp_tools = McpTools::from_config_with_process_policy("mcp.toml", process_policy).await?;
```

Remote MCP servers use `type = "remote"` and may include HTTP headers for API-key style authentication. OAuth is intentionally not handled by the toolkit MCP client yet; store and inject credentials from your application boundary.

```toml
[mcp]
[[mcp.server]]
name = "docs"
type = "remote"
url = "https://example.com/mcp"
timeout_ms = 30000

[mcp.server.headers]
Authorization = "Bearer ${TOKEN}"
```
