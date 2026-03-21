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
struct WriteArgs { path: String, contents: String }

#[tool(name = "write_file", description = "Write text to a file", input = WriteArgs)]
struct WriteFile;

#[async_trait]
impl ToolRuntime for WriteFile {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let a: WriteArgs = serde_json::from_value(args)?;
        std::fs::write(a.path, a.contents).map_err(|e| ToolCallError::RuntimeError(e.into()))?;
        Ok(serde_json::json!({"ok": true}))
    }
}
```

Attach tools in the `#[agent(..., tools = [ .. ])]` macro. Tools can also be built dynamically; when sharing `Arc<dyn ToolT>` across agents use `shared_tools_to_boxes`.

## Toolkit

Reusable tools are in `autoagents-toolkit`:

- Filesystem tools: `ListDir`, `ReadFile`, `WriteFile`, `CopyFile`, `MoveFile`, `DeleteFile`, `SearchFile` (feature: `filesystem`)
- Web search: `BraveSearch` (feature: `search`, requires `BRAVE_SEARCH_API_KEY` or `BRAVE_API_KEY`)

Enable features in your `Cargo.toml` as needed.

## MCP

Model Context Protocol (MCP) integrations are available via `autoagents-toolkit::mcp` — load tool definitions from MCP servers and expose them as `ToolT`. See `McpTools` for managing connections and wrapping MCP tools for use by agents.
