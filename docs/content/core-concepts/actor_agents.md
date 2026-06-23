# Actor Agents

Actor-based agents run inside a runtime and communicate via typed messages and protocol events. Use them when you need:
- Event streaming for UI updates (turn started/completed, tool calls, streaming chunks)
- Pub/Sub between agents or external actors
- Long-running orchestrations rather than a single direct call

## Core Building Blocks

- `Runtime` and `Environment`: Manage event routing and lifecycle of actor systems.
- `Topic<M>`: Typed pub/sub channels for messages of type `M`.
- `ActorAgent` and `ActorAgentHandle`: Wrap your agent with an actor reference so it can receive `Task`s via pub/sub/direct messaging.
- `Event`: Streamed protocol events (task start/complete, tool calls, streaming chunks) published by agents during execution.

## Typical Wiring Pattern

1) Create a runtime and register it in an `Environment`.
2) Spawn your agent as `ActorAgent` and subscribe it to one or more `Topic<Task>`.
3) Take the environment's event receiver and forward events to your UI/log sink.
4) Publish `Task` messages to the relevant topic to trigger work.
5) Call [`Environment::run`](https://docs.rs/autoagents-core/latest/autoagents_core/environment/struct.Environment.html#method.run), then [`wait`](https://docs.rs/autoagents-core/latest/autoagents_core/environment/struct.Environment.html#method.wait) or [`shutdown`](https://docs.rs/autoagents-core/latest/autoagents_core/environment/struct.Environment.html#method.shutdown) to manage runtime lifecycle.

## Environment Lifecycle

After wiring agents and publishing work, start the registered runtimes:

- **`run()`** — spawns a background task that runs all registered runtimes. Returns `Err(EnvironmentError::AlreadyRunning)` if a run task is already in progress. Does **not** block until work completes.
- **`wait().await`** — joins the background task started by `run()`. Clears the stored handle when the task finishes, so later calls return immediately with `Ok(Ok(()))`. Use this in short-lived programs once messages/tasks have been published.
- **`shutdown().await`** — requests shutdown on all runtimes and joins the run handle. Returns `Result<(), EnvironmentError>` so runtime or join failures are visible. Use for graceful exit (for example on `Ctrl+C`).
- **`run_background().await`** — starts runtimes without storing a join handle on the environment. Useful when you manage lifecycle elsewhere. Cannot be combined with `run()` on the same `Environment` until `shutdown().await` is called.
- **`is_running()`** — returns whether a background run task is currently active.

If a managed run task finishes without calling `wait()` or `shutdown()`, a subsequent `run()` joins the finished task first and returns any runtime or join error before spawning a new run.

Calling `wait()` inside `tokio::select!` is safe: if another branch wins, the join handle stays on the environment so `shutdown()` can still drain the run task.

`SingleThreadedRuntime` is single-cycle: after its event loop exits, register a **new** runtime instance to run again on the same `Environment`.

### Migration from earlier releases

`Environment::run()` no longer returns a `JoinHandle`. Use the stored lifecycle instead:

```rust
// Before
let handle = environment.run();
handle.await??;

// After
environment.run()?;
let run_result = environment.wait().await?;
run_result?;
```

For long-running programs that previously dropped the join handle, call `environment.shutdown().await?` on exit.

Common patterns:

```rust
// Fire-and-wait: publish work, start runtimes, block until they finish
environment.run()?;
let run_result = environment.wait().await?;
if let Err(runtime_err) = run_result {
    eprintln!("runtime error: {runtime_err}");
}
```

```rust
// Interactive or long-running: start runtimes, handle Ctrl+C gracefully
environment.run()?;
// `wait()` is select-safe — if Ctrl+C wins, the join handle is restored
// so shutdown can still drain runtimes. See design_patterns::environment_lifecycle.
tokio::select! {
    result = environment.wait() => {
        if let Ok(Err(e)) = result {
            eprintln!("runtime error: {e}");
        } else if let Err(e) = result {
            eprintln!("run task join error: {e}");
        }
    }
    _ = tokio::signal::ctrl_c() => {
        if let Err(err) = environment.shutdown().await {
            eprintln!("shutdown failed: {err}");
        }
    }
}
```

Call `run()` **after** registering runtimes and **before** or **after** publishing tasks depending on your program. Actor agents process messages only while runtimes are running.

## Minimal Example

```rust
use autoagents::core::{
  agent::prebuilt::executor::ReActAgent,
  agent::{AgentBuilder, ActorAgent},
  agent::task::Task,
  environment::Environment,
  runtime::{SingleThreadedRuntime, TypedRuntime},
  actor::Topic,
};
use std::sync::Arc;

// 1) Create runtime and environment
let runtime = SingleThreadedRuntime::new(None);
let mut env = Environment::new(None);
env.register_runtime(runtime.clone()).await?;

// 2) Build actor agent and subscribe to a topic
let chat_topic = Topic::<Task>::new("chat");
let handle = AgentBuilder::<_, ActorAgent>::new(ReActAgent::new(MyAgent {}))
    .llm(my_llm)
    .runtime(runtime.clone())
    .subscribe(chat_topic.clone())
    .build()
    .await?;

// 3) Consume events (UI updates, tool calls, streaming)
let receiver = env.take_event_receiver(None).await?;
tokio::spawn(async move { /* forward events to UI */ });

// 4) Publish tasks
runtime.publish(&chat_topic, Task::new("Hello!"))
    .await?;

// 5) Start runtimes and wait for the background run task to finish
env.run()?;
let _ = env.wait().await?;
```

## Event Handling Patterns

- Pub/Sub: Publish `Task` to `Topic<Task>`; all subscribed agents receive the message.
- Direct send: Use `TypedRuntime::send_message` to deliver a message directly to a specific actor.
- Protocol events: `Event::TaskStarted`, `Event::TurnStarted`, `Event::ToolCallRequested`, `Event::StreamChunk`, etc. are emitted by agents while running.

## Protocol Events Reference

These map to `autoagents::core::protocol::Event` variants emitted by actor agents and the runtime:

- `TaskStarted { sub_id, actor_id, actor_name, task_description }`
  - Emitted when an agent begins processing a task.
- `TaskComplete { sub_id, actor_id, actor_name, result }`
  - Final result for a task. `result` is a pretty JSON string; parse into your agent output type when needed.
- `TaskError { sub_id, actor_id, error }`
  - Any executor/provider error surfaced during execution.
- `TurnStarted { sub_id, actor_id, turn_number, max_turns }`
  - Multi-turn executors (e.g., ReAct) mark each turn start.
- `TurnCompleted { sub_id, actor_id, turn_number, final_turn }`
  - Marks turn completion; `final_turn` is true when the loop ends.
- `ToolCallRequested { sub_id, actor_id, id, tool_name, arguments }`
  - The LLM requested a tool call with JSON arguments (as string).
- `ToolCallCompleted { sub_id, actor_id, id, tool_name, result }`
  - Tool finished successfully; `result` is JSON.
- `ToolCallFailed { sub_id, actor_id, id, tool_name, error }`
  - Tool failed; error string is included.
- `StreamChunk { sub_id, chunk }`
  - Streaming delta content; `chunk` matches provider’s streaming shape.
- `StreamToolCall { sub_id, tool_call }`
  - Streaming tool call delta (when provider emits incremental tool-call info).
- `StreamComplete { sub_id }`
  - Streaming finished for the current task.

Internally, the runtime also routes `PublishMessage` for typed pub/sub (`Topic<M>`), but that variant is skipped in serde and used only inside the runtime.

## Actor streaming APIs

Actor agents expose two streaming entry points with different event contracts:

| API | Terminal events (`TaskComplete` / `TaskError`) | Hooks | Typical use |
|-----|-----------------------------------------------|-------|-------------|
| `run_stream()` | **No** — failures are `Err` items on the returned stream only | Skipped | Incremental output; poll the stream directly |
| `run_stream_to_completion()` | **Yes** — full task lifecycle on the event channel | Run | Runtime pub/sub dispatch, event subscribers, `select!` on `TaskError` |

**Footgun:** If you subscribe to runtime or agent events and call `run_stream()` directly, terminal failures will **not** emit `TaskError` on the event channel even though mid-run events (`StreamChunk`, tool calls) may still arrive. Listeners waiting only for `TaskComplete` / `TaskError` can hang. Use `run_stream_to_completion()` instead, or handle errors on the returned output stream.

Non-streaming `run()` always emits `TaskComplete` or `TaskError`. This matches `run_stream_to_completion()`, not `run_stream()`.

For the direct-agent variant of this contract, see [Agents — Direct agent event contract](./agents.md#direct-agent-event-contract).

## When To Use Actor Agents vs Direct Agents

- Use Direct agents for one-shot calls (no runtime, minimal wiring).
- Use Actor agents when you need: real-time events, multiple agents, pub/sub routing, or running agents as durable tasks.
