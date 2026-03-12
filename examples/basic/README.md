# Basic Example

## Use Cases

### Basic Agent

```sh
export OPENAI_API_KEY=your_openai_api_key_here
cargo run --package basic-example -- --usecase basic
```

### Simple ReAct Agent

```sh
export OPENAI_API_KEY=your_openai_api_key_here
cargo run --package basic-example -- --usecase simple
```

Shows a basic event-driven agent.

### Add hooks to agents

```sh
export OPENAI_API_KEY=your_openai_api_key_here
cargo run --package basic-example -- --usecase hooks
```

### Create Agents and Tools without Macro

This is useful when dynamic tool and agent instantiation is required

```sh
export OPENAI_API_KEY=your_openai_api_key_here
cargo run --package basic-example -- --usecase manual-tool-agent
```

### Generic Actors

```sh
export OPENAI_API_KEY=your_openai_api_key_here
cargo run --package basic-example -- --usecase actor
```

#### Edge Runtime using LiquidEdge

Download the tinyllama model from ONNX file and models weights

```sh
optimum-cli export onnx \
      --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --task text-generation ./models/tinyllama
```

### Reasoning / Thinking Output

Demonstrates agent-level `reasoning_content` for both:
- stream `run_stream()` via event chunks

```sh
cargo run --package basic-example -- --usecase thinking
```

This use case uses the OpenAI-compatible backend pointed at a vLLM server.
Set:
- `VLLM_BASE_URL` (or `OPENAI_BASE_URL`) default: `http://127.0.0.1:8000/v1`
- `VLLM_MODEL` default: `Qwen/Qwen3-8B`
- `OPENAI_API_KEY` (vLLM typically accepts any non-empty value)

and prints:
- response text events
- reasoning events from `Event::StreamChunk(StreamChunk::ReasoningContent)`

This avoids adding reasoning fields to the agent output schema.

Note: `reasoning_content` chunks are provider/model dependent.


### ToolKit

```sh
export OPENAI_API_KEY=your_openai_api_key_here
export BRAVE_API_KEY=your_brave_api_key_here
cargo run --package basic-example -- --usecase toolkit
```
