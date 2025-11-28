## Advanced Patterns (Core)

This chapter collects practical patterns using the core library APIs.

### 1) Shared Tools Across Agents

Use `SharedTool` or `shared_tools_to_boxes` to reuse `Arc<dyn ToolT>` across many agents without cloning tool instances.

### 2) Memory Strategies

- Start with `SlidingWindowMemory` for compact histories.
- Persist memory using your own storage if you want durable conversations.
- Mix recalled messages from memory with system instructions in `Context`.

### 3) Streaming UI Integration

- Prefer `execute_stream` and consume stream items directly.
- Subscribe to protocol `Event`s for fine-grained updates like `TurnStarted`, `ToolCallRequested`, and `StreamChunk`.

### 4) Multi-Agent Topologies (Pub/Sub)

- Use `Topic<M>` to broadcast tasks to a group of actor agents.
- Combine with `Environment` + `Runtime` to route events and messages.

<!-- Serve and CLI topics moved to the Serve & CLI chapter. -->
