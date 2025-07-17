# Implementing Separate Streaming and Non-Streaming Execution Paths

This document outlines the architectural approach to implementing separate execution paths for standard (non-streaming) and streaming agent behavior, as per maintainer feedback. The goal is to create two distinct, parallel code paths for clarity and maintainability.

## Core Principle

The fundamental idea is to avoid mixing streaming and non-streaming logic within the same functions. Instead, we create a dedicated set of methods for each execution model, from the highest-level entry point down to the specific executor implementation.

This is particularly important for use cases like voice agents, where receiving and processing tokens in real-time is critical, versus batch processing where a complete response is awaited.

## Implementation Steps

Here is a step-by-step guide to achieving this separation:

### 1. Create a Dedicated Streaming Method in the Executor (`ReActExecutor`)

The `ReActExecutor` in `crates/core/src/agent/prebuilt/react.rs` is responsible for handling the logic of a single agent turn. This is where the separation begins.

-   **`process_turn` (Non-Streaming):** This method should handle a single, synchronous-style turn. It calls the LLM, gets a complete response back (including any tool calls), processes it, and returns a `TurnResult`. It should **not** deal with streaming chunks.

-   **`process_stream_turn` (New Streaming Method):** A new method with a nearly identical signature to `process_turn`. Its responsibilities are:
    1.  Call the LLM's streaming endpoint (e.g., `chat_stream_with_tools`).
    2.  Asynchronously iterate through the stream of `ChatResponseChunk`s.
    3.  Accumulate the text and tool call data from the chunks.
    4.  Reconstruct full tool calls once the stream is complete.
    5.  Execute the tools and return a `TurnResult`.

### 2. Define Separate Top-Level Execution Paths in the `AgentExecutor` Trait

The `AgentExecutor` trait in `crates/core/src/agent/executor.rs` defines the main public entry points for running an agent. We will mirror the separation here.

-   **`execute` (Non-Streaming):** This existing method will serve as the entry point for standard execution. It will contain the loop that calls `process_turn` repeatedly until the task is complete.

-   **`execute_stream` (New Streaming Method):** A new `execute_stream` method will be added to the `AgentExecutor` trait.
    -   It will have the same signature as `execute`.
    -   Its implementation for `ReActExecutor` will contain a loop that calls the new `process_stream_turn` method.

### 3. Expose Streaming Execution in the Environment

The "environment" represents the highest-level API that end-users will interact with (e.g., in the `examples` directory or a library crate).

-   **`run()`:** This function will call the agent's `execute()` method for standard, non-streaming behavior.
-   **`run_stream()`:** This new function will call the agent's `execute_stream()` method to initiate the streaming execution path.

This provides a clear and explicit choice for the developer using the agent.

## Summary of the Flow

The resulting architecture creates two parallel flows:

| Layer                   | Non-Streaming Path              | Streaming Path (New)                     |
| :---------------------- | :------------------------------ | :--------------------------------------- |
| **Environment**         | `run()`                         | `run_stream()`                           |
| **`AgentExecutor` Trait** | `execute()`                     | `execute_stream()`                       |
| **`ReActExecutor` Impl**  | `process_turn()`                | `process_stream_turn()`                  |
| **LLM Call**            | `llm.chat_with_tools()`         | `llm.chat_stream_with_tools()`           |
