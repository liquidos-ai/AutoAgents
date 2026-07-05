# Coding Agent Example

A ReAct (Reasoning + Acting) based coding agent that demonstrates file manipulation capabilities similar to AI coding assistants. This agent systematically reasons through tasks and uses tools to search, read, write, and analyze code files in a project.

## Use Cases

### Interactive Coding Agent
```sh
export OPENAI_API_KEY=your_openai_api_key_here
cargo run --package coding_agent -- --usecase interactive --workspace-root .
```
Provides an interactive session where you can ask the agent to perform various coding tasks.

All file operations are sandboxed to `--workspace-root`. The agent must use workspace-relative paths; absolute paths, `..` traversal outside the workspace, and symlink escapes are rejected by the filesystem tools.

Directory deletion is non-recursive unless the tool call explicitly sets `recursive` to `true`.
