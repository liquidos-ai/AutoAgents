# Agents

Agents are the core building blocks of AutoAgents. They understand tasks, apply an execution strategy (e.g., ReAct or Basic), optionally call tools, and produce outputs (string or structured JSON).

## What Is an Agent?

An agent in AutoAgents typically:

- Defines metadata (name, description) and available tools
- Chooses an executor (Basic or ReAct)
- Uses an LLM provider
- Optionally has memory for context
- Emits events (task started/completed, tool calls, streaming chunks)

## Agent Lifecycle

1. Build: `AgentBuilder` wraps your agent into a runnable `BaseAgent` with an LLM, optional memory, and event channel
2. Run: call `run(Task)` (or `run_stream`) to execute
3. Hooks: optional `AgentHooks` fire on create, run start/complete, per‑turn, and tool activity
4. Output: executor output is converted into your agent output type (`From<...>`)

## Direct Agents

Direct agents expose simple `run`/`run_stream` APIs and return results to the caller.

```rust
use autoagents::core::agent::{AgentBuilder, DirectAgent};
let handle = AgentBuilder::<_, DirectAgent>::new(my_executor)
    .llm(llm)
    .build()
    .await?;
let result = handle.agent.run(Task::new("Prompt")) .await?;
```

## Actor Based Agents

Actor agents integrate with a runtime for pub/sub and cross‑agent messaging. Use `AgentBuilder::<_, ActorAgent>` with `.runtime(...)` and optional `.subscribe(topic)`.

High‑level flow:

- Create a `SingleThreadedRuntime`
- Build the agent with `.runtime(runtime.clone())`
- Register runtime in an `Environment` and run it
- Publish `Task`s to topics or send direct messages

Use actor agents for multi‑agent collaboration, routing, or background workflows.
