# Executors

AutoAgents ships with two primary executors. You can also implement your own by conforming to `AgentExecutor`.

- Basic: single‑turn chat without tool calls
- ReAct: multi‑turn reasoning with tool calls and streaming

## Basic

Use `BasicAgent<T>` to run a single prompt/response cycle. Suitable for simple Q&A and when tools are not needed.

Key points:

- No tool calls
- Optional streaming variant
- Output type: `BasicAgentOutput` → convert to your agent output via `From<...>`

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

```rust
use autoagents::core::agent::prebuilt::executor::ReActAgent;
let agent = ReActAgent::new(MyAgent);
```

Tip: `ReActAgentOutput::extract_agent_output<T>` can deserialize a structured JSON response into your type when you expect strict JSON.

