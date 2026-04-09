# Executors: When To Use What

AutoAgents ships three execution strategies out of the box. Pick based on your task shape and integration needs.

### BasicAgent

- Single-turn request with no intermediate tool calls.
- Great for:
  - Prompt → response without orchestration
  - Simple chat replies
  - Fast endpoints or CLI utilities
- Streaming: Yes (`execute_stream` yields delta content)
- Output: `BasicAgentOutput { response, done }` — can be converted to your structured type with `parse_or_map`.
- Reasoning/thinking deltas, when supported by the model/provider, are emitted via stream events (`Event::StreamChunk(StreamChunk::ReasoningContent)`).

Use if you need a quick, single-shot LLM call.

### ReActAgent

- Multi-turn loop with tool calling support and memory integration.
- Great for:
  - Tool-augmented tasks (retrieval, search, filesystem, MCP)
  - Step-by-step reasoning with intermediate state
  - Streaming UI with turn events
- Streaming: Yes (delta content and tool-call accumulation)
- Output: `ReActAgentOutput { response, tool_calls, done }` — parse via `try_parse`/`parse_or_map` or `extract_agent_output`.
- Reasoning/thinking deltas are available through stream events (`StreamChunk::ReasoningContent`).

Use if your agent must call tools or orchestrate multiple turns.

### CodeActAgent

- Multi-turn loop where the model writes sandboxed TypeScript to compose tools.
- Great for:
  - Coding and data-processing workflows where tool composition matters
  - Tasks that benefit from local computation inside the executor instead of repeated model turns
  - Rich observability around code execution, console output, and nested tool calls
- Streaming: Yes (delta content plus code-execution lifecycle events)
- Output: `CodeActAgentOutput { response, executions, done }` — parse via `try_parse`/`parse_or_map` or `extract_agent_output`.
- Requires the `codeact` cargo feature and a native target.

Use if your agent should plan with tools by writing small TypeScript programs instead of issuing direct tool calls.

### Structured Output Helpers

All prebuilt executors provide helpers to reduce parsing boilerplate:

```rust
// Best-effort parse with fallback to raw text mapping
let my_struct = output.parse_or_map(|raw| MyStruct { text: raw.to_string() });

// Strict parse
let parsed: MyStruct = output.try_parse()?;
```
