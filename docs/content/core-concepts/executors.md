# Executors

AutoAgents ships with three primary executors. You can also implement your own by conforming to `AgentExecutor`.

- Basic: single‑turn chat without tool calls
- ReAct: multi‑turn reasoning with tool calls and streaming
- CodeAct: multi‑turn reasoning where the model writes sandboxed TypeScript that composes tools

## Basic

Use `BasicAgent<T>` to run a single prompt/response cycle. Suitable for simple Q&A and when tools are not needed.

Key points:

- No tool calls
- Optional streaming variant
- Output type: `BasicAgentOutput` → convert to your agent output via `From<...>`
- Reasoning/thinking chunks are emitted in stream events (`StreamChunk::ReasoningContent`) when supported.

```rust
use autoagents::core::agent::prebuilt::executor::BasicAgent;
let agent = BasicAgent::new(MyAgent);
```

## ReAct

Use `ReActAgent<T>` for iterative reasoning + acting. Supports tool calls, multi‑turn loops (with `max_turns`), and streaming with intermediate events.

Key points:

- Tool calls are serialized, executed, and their results fed back to the LLM
- Emits events: tool requested/completed, turn started/completed, stream chunks
- Output type: `ReActAgentOutput` → convert to your agent output via `From<...>`
- Reasoning/thinking remains event-level (`StreamChunk::ReasoningContent`) and is not included in final output fields by default.

```rust
use autoagents::core::agent::prebuilt::executor::ReActAgent;
let agent = ReActAgent::new(MyAgent);
```

Tip: `ReActAgentOutput::extract_agent_output<T>` can deserialize a structured JSON response into your type when you expect strict JSON.

## CodeAct

Use `CodeActAgent<T>` when you want the model to compose tools through sandboxed TypeScript instead of issuing direct tool calls one by one.

Key points:

- The model only sees one tool: `execute_typescript`
- Your registered tools are exposed inside the sandbox as typed `external_*` functions
- Each execution runs in a fresh native sandbox with explicit resource limits
- Output type: `CodeActAgentOutput { response, executions, done }`
- Emits additional events for code execution lifecycle and console output
- Available on native targets with the `codeact` feature enabled

```rust
use autoagents::core::agent::prebuilt::executor::CodeActAgent;
let agent = CodeActAgent::new(MyAgent);
```
