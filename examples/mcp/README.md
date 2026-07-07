# MCP (Model Context Protocol) Integration Example

This example demonstrates how to integrate MCP servers with AutoAgents, allowing your agents to use tools from external MCP-compatible servers.

MCP local servers execute host processes. This example uses the globally installed Brave Search MCP server binary and explicitly allowlists that command name in code; do not load MCP configs from untrusted sources, and keep your `PATH` trusted.

### Setup

```sh
export OPENAI_API_KEY=your_openai_api_key_here
export BRAVE_API_KEY=your_brave_api_key_here
npm install -g @modelcontextprotocol/server-brave-search
cargo run --package mcp-example
```
