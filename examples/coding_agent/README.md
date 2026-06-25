# Coding Agent Example

A ReAct (Reasoning + Acting) based coding agent that demonstrates **sandboxed** file manipulation capabilities. All filesystem tools are scoped to an explicit workspace root; relative paths only, with traversal and symlink escapes rejected.

## Workspace

By default the agent uses the current working directory as the workspace sandbox. Override it with `--workspace`:

```sh
export OPENAI_API_KEY=your_openai_api_key_here
cargo run --package coding_agent -- --usecase interactive --workspace /path/to/project
```

At startup the canonical workspace path is printed to the terminal.

## Interactive session

```sh
export OPENAI_API_KEY=your_openai_api_key_here
cargo run --package coding_agent -- --usecase interactive
```

## Sandboxed tool wiring

The example builds filesystem tools with a required workspace root (see `examples/coding_agent/src/agent.rs`):

```rust
ReadFile::new(&workspace_root)?;
WriteFile::new(&workspace_root)?;
DeleteFile::new(&workspace_root)?;
```

Custom tools (`GrepTool`, `AnalyzeCodeTool`) use the same `FilesystemSandbox` from `autoagents-toolkit`.

## Safety notes

- Paths must be relative to the workspace root (`..` and absolute paths are rejected).
- Directory deletion is non-recursive by default; pass `"recursive": true` to `delete_file` when needed.
- Directory walks do not follow symlinks.
