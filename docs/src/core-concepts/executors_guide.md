## Executors: When To Use What

AutoAgents ships two execution strategies out of the box. Pick based on your task shape and integration needs.

### BasicAgent

- Single-turn request with no intermediate tool calls.
- Great for:
  - Prompt → response without orchestration
  - Simple chat replies
  - Fast endpoints or CLI utilities
- Streaming: Yes (`execute_stream` yields delta content)
- Output: `BasicAgentOutput { response, done }` — can be converted to your structured type with `parse_or_map`.

Use if you need a quick, single-shot LLM call.

### ReActAgent

- Multi-turn loop with tool calling support and memory integration.
- Great for:
  - Tool-augmented tasks (retrieval, search, filesystem, MCP)
  - Step-by-step reasoning with intermediate state
  - Streaming UI with turn events
- Streaming: Yes (delta content and tool-call accumulation)
- Output: `ReActAgentOutput { response, tool_calls, done }` — parse via `try_parse`/`parse_or_map` or `extract_agent_output`.

Use if your agent must call tools or orchestrate multiple turns.

### Structured Output Helpers

Both executors provide helpers to reduce parsing boilerplate:

```rust
// Best-effort parse with fallback to raw text mapping
let my_struct = output.parse_or_map(|raw| MyStruct { text: raw.to_string() });

// Strict parse
let parsed: MyStruct = output.try_parse()?;
```
