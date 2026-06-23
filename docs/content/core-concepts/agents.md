# Agents

Agents are the core building blocks of AutoAgents. They understand tasks, apply an execution strategy (for example Basic, ReAct, or CodeAct), optionally call tools, and produce outputs (string or structured JSON).

## What Is an Agent?

An agent in AutoAgents typically:

- Defines metadata (name, description) and available tools
- Chooses an executor (Basic, ReAct, or CodeAct)
- Uses an LLM provider
- Optionally has memory for context
- Emits events (task started/completed, tool calls, streaming chunks)

## Agent Lifecycle

1. Build: `AgentBuilder` wraps your agent into a runnable `BaseAgent` with an LLM, optional memory, and event channel
2. Run: call `run(Task)` (or `run_stream`) to execute
3. Hooks: optional `AgentHooks` fire on create, run start/complete, per‑turn, and tool activity
4. Output: executor output is converted into your agent output type (`From<...>`)

## Direct Agents

Direct agents expose simple `run`/`run_stream` APIs and return results to the caller. `AgentBuilder::build()` returns a `DirectAgentHandle` with the runnable agent and an event receiver (`handle.rx`).

```rust
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, DirectAgent};

let mut handle = AgentBuilder::<_, DirectAgent>::new(my_executor)
    .llm(llm)
    .build()
    .await?;

// One-shot: the final output is available from the return value and as TaskComplete on handle.rx
let result = handle.agent.run(Task::new("Prompt")).await?;
```

### Direct agent event contract

Direct agents emit the same terminal protocol events as actor agents on success and failure. Use the `Result` from `run()` or items from `run_stream()` for output; subscribe to `handle.rx` when multiple consumers need lifecycle events.

| Outcome | Return value | `handle.rx` |
|--------|--------------|-------------|
| Success (`run`) | `Ok(output)` | `TaskComplete` |
| Success (`run_stream`) | `Ok(stream)` of output items | `TaskComplete` when the output stream is fully drained (last successful item) |
| Hook abort | `Err(Abort)` | `TaskError` |
| Executor error (`run`) | `Err(...)` | `TaskError` |
| Stream setup error (`run_stream`) | `Err(...)` | `TaskError` |
| Empty stream (`run_stream`) | `Ok(empty stream)` | `TaskError` when the output stream is fully drained |
| In-stream item error (`run_stream`) | `Err(...)` on the output stream | `TaskError` when the stream item is polled |

**Listening for terminal events:** If your code waits on `handle.rx` (for example in a `select!`), subscribe before or concurrently with execution. Both success and failure emit a terminal event so listeners do not hang indefinitely.

**Streaming caveat:** For `run_stream()`, in-stream failures and `TaskComplete` are emitted when the **returned output stream is polled**. If you only listen on `handle.rx` and never poll the output stream, item-level errors and stream completion will not surface on the event channel.

**Mid-run events:** Executors may still emit non-terminal events (for example `TurnStarted`, `ToolCallRequested`, `StreamChunk`) on `handle.rx` during execution. Use `handle.subscribe_events()` when multiple consumers need the same stream.

For full protocol event shapes, see [Actor Agents — Protocol Events Reference](./actor_agents.md#protocol-events-reference). Direct agents and actor agents both emit `TaskComplete` / `TaskError` on the terminal paths above; see [Actor streaming APIs](./actor_agents.md#actor-streaming-apis) for the actor-only `run_stream()` footgun (no terminal events on the public streaming API).

## Actor Based Agents

Actor agents integrate with a runtime for pub/sub and cross‑agent messaging. Use `AgentBuilder::<_, ActorAgent>` with `.runtime(...)` and optional `.subscribe(topic)`.

High‑level flow:

- Create a `SingleThreadedRuntime`
- Build the agent with `.runtime(runtime.clone())`
- Register runtime in an `Environment`, call `run()`, then `wait()` or `shutdown()`
- Publish `Task`s to topics or send direct messages

See [Actor Agents](./actor_agents.md#environment-lifecycle) for lifecycle details (`run`, `wait`, `shutdown`).

Use actor agents for multi‑agent collaboration, routing, or background workflows.

## Hooks

Agents can implement the `AgentHooks` trait to observe and customize execution at well‑defined points. This is useful for logging, metrics, guardrails, or side‑effects.

Lifecycle hooks:

- `on_agent_create(&self)` — called when the agent is constructed
- `on_run_start(&self, task, ctx) -> HookOutcome` — called before execution; return `Abort` to cancel
- `on_run_complete(&self, task, result, ctx)` — called after execution succeeds
- `on_turn_start(&self, turn_index, ctx)` — called at the start of each executor turn (e.g., ReAct or CodeAct)
- `on_turn_complete(&self, turn_index, ctx)` — called when a turn completes
- `on_tool_call(&self, tool_call, ctx) -> HookOutcome` — gate a tool call before it executes
- `on_tool_start(&self, tool_call, ctx)` — tool execution started
- `on_tool_result(&self, tool_call, result, ctx)` — tool execution completed successfully
- `on_tool_error(&self, tool_call, err, ctx)` — tool execution failed
- `on_agent_shutdown(&self)` — actor shutdown hook (actor agents only)

Example:

```rust
use autoagents::prelude::*;

#[derive(Clone, Default, AgentHooks)]
struct MyAgent;

#[autoagents::agent(name = "my_agent", description = "Example agent")]
impl MyAgent {}

#[autoagents::async_trait]
impl AgentHooks for MyAgent {
    async fn on_run_start(&self, task: &Task, _ctx: &Context) -> HookOutcome {
        if task.prompt.len() > 2_000 { HookOutcome::Abort } else { HookOutcome::Continue }
    }
    async fn on_tool_error(&self, _call: &autoagents::llm::ToolCall, err: serde_json::Value, _ctx: &Context) {
        eprintln!("tool failed: {}", err);
    }
}
```

Tips:

- Hooks should be fast and side‑effect aware, particularly in streaming contexts.
- `on_agent_shutdown` only fires for actor agents (not for direct agents).
- Use `Context` to access config, memory, and event sender.
