# Memory

Memory lets agents retain context across turns. AutoAgents exposes a `MemoryProvider` trait and includes a simple sliding‑window implementation.

## Sliding Window Memory

`SlidingWindowMemory` stores the most recent N messages (FIFO). Use it when recent context is sufficient and you want predictable memory usage.

```rust
use autoagents::core::agent::memory::SlidingWindowMemory;
let memory = Box::new(SlidingWindowMemory::new(10));
```

Attach memory via `AgentBuilder`:

```rust
let handle = AgentBuilder::<_, DirectAgent>::new(agent)
    .llm(llm)
    .memory(Box::new(SlidingWindowMemory::new(10)))
    .build()
    .await?;
```

The ReAct executor automatically stores user/assistant messages and tool interactions in memory each turn.

## Custom Memory

Implement `MemoryProvider` to support alternate strategies (e.g., vector‑store, summaries, persistence). Key methods:

- `remember(&ChatMessage)` — store message
- `recall(query, limit)` — retrieve relevant context
- `clear()` — reset
- `size()` / `memory_type()` — diagnostics

The trait includes convenience hooks for summarization and export/import if you need persistence.
