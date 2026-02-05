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

## When To Use Actor Agents vs Direct Agents

- Use Direct agents for one-shot calls (no runtime, minimal wiring).
- Use Actor agents when you need: real-time events, multiple agents, pub/sub routing, or running agents as durable tasks.
