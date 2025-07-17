# Streaming Agent Example

This example demonstrates how to use the AutoAgents framework with streaming execution. It showcases a simple agent that can interact with tools and stream its output back to the user in real-time.

## Prerequisites

Before running the example, you need to set your OpenAI API key as an environment variable.

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Running the Example

You can run the agent in both streaming and non-streaming modes to compare the behavior.

### Non-Streaming Mode

To run the agent without streaming, execute the following command from the `examples/streaming_agent` directory:

```bash
cargo run -- --usecase simple
```

The agent will process the entire task and print the final result at the end.

### Streaming Mode

To run the agent in streaming mode, add the `--stream` flag:

```bash
cargo run -- --usecase simple --stream
```

In this mode, you will see the agent's output, including tool calls and intermediate thoughts, as they are generated.

### Other Use Cases

This example also includes an `events` use case that demonstrates how to subscribe to and handle agent events. To run it, use:

```bash
cargo run -- --usecase events
```
