# Coding Agent Example

A ReAct (Reasoning + Acting) based coding agent that demonstrates file manipulation capabilities similar to AI coding assistants. This agent systematically reasons through tasks and uses tools to search, read, write, and analyze code files in a project.

## Use Cases

### Basic Coding Agent
```sh
export OPENAI_API_KEY=your_openai_api_key_here
cargo run --package coding_agent -- --usecase simple
```
Shows a basic coding agent that can help with file operations.

### Interactive Coding Agent
```sh
export OPENAI_API_KEY=your_openai_api_key_here
cargo run --package coding_agent -- --usecase interactive
```
Provides an interactive session where you can ask the agent to perform various coding tasks.
