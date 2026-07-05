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

Model Context Protocol (MCP) integrations are available via `autoagents-toolkit::mcp` — load tool definitions from MCP servers and expose them as `ToolT`. See `McpTools` for managing connections and wrapping MCP tools for use by agents.
