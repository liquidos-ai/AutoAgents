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

### Filesystem sandboxing (v0.4.0)

Filesystem tools require an explicit workspace root at construction. Unscoped constructors were removed in v0.4.0.

```rust
use autoagents_toolkit::tools::filesystem::{FilesystemSandbox, ReadFile, WriteFile};

let workspace = std::env::current_dir()?;
let read = ReadFile::new(&workspace)?;
let write = WriteFile::new(&workspace)?;
```

Rules enforced by `FilesystemSandbox`:

- User paths must be **relative** to the workspace root (absolute paths are rejected).
- `..` traversal components are rejected.
- Existing paths are canonicalized and verified to stay within the root (symlink escapes fail closed).
- Directory walks use `follow_links(false)`.
- `delete_file` deletes directories non-recursively by default; pass `"recursive": true` to remove a directory tree.
- Mutating tools re-validate paths after creating parent directories to close TOCTOU races.

Symlink-escape regression tests run on Unix in CI. On Windows, the same canonicalization checks apply at runtime, but dedicated Windows junction fixtures are not yet in the test suite.

### v0.4.0 migration

| Before | After |
| ------ | ----- |
| `ReadFile::new()` | `ReadFile::new(&workspace)?` |
| `new_with_root_dir(s)` | `new(&s)?` or `with_sandbox(sandbox)` |
| `SearchFile::new(100)` | `SearchFile::new(&workspace, 100)?` |
| Recursive directory delete (implicit) | Pass `"recursive": true` to `delete_file` |

See [CHANGELOG.md](../../../CHANGELOG.md) for the full migration guide.

For dynamic tool wiring (workspace known only at runtime), see `examples/coding_agent` and `examples/basic/src/manual_tool_agent.rs`.

## MCP

Model Context Protocol (MCP) integrations are available via `autoagents-toolkit::mcp` — load tool definitions from MCP servers and expose them as `ToolT`. See `McpTools` for managing connections and wrapping MCP tools for use by agents.
