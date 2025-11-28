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

For CPU usage

```sh
cargo run --package basic-example -- --usecase edge --device cpu
```

For CUDA usage - Make use cuDNN and cuda drivers are installed properly

```sh
cargo run --package basic-example -- --usecase edge --device cuda
```



### ToolKit

```sh
export OPENAI_API_KEY=your_openai_api_key_here
export BRAVE_API_KEY=your_brave_api_key_here
cargo run --package basic-example -- --usecase toolkit
```

### Token Tracking

Track token usage across LLM calls with a BasicAgent:

```sh
export OPENAI_API_KEY=your_openai_api_key_here
cargo run --package basic-example -- --usecase token-tracking
```

This example demonstrates:
- Real-time token tracking for each LLM call
- Cumulative token statistics across multiple tasks
- Detailed breakdown of prompt, completion, and total tokens
- Usage history for all LLM interactions

### ReAct Token Tracking

Track token usage in a ReAct agent with tool calls:

```sh
export OPENAI_API_KEY=your_openai_api_key_here
cargo run --package basic-example -- --usecase react-token-tracking
```

This example demonstrates:
- Token tracking across multiple reasoning turns
- Token usage with tool calls (calculator)
- Per-turn and cumulative statistics
- Cost efficiency metrics

### Token Tracking with Ollama (Local LLM)

Track token usage with a local Ollama model - no API key required:

```sh
# First, install Ollama: https://ollama.com/download
# Pull the model: ollama pull llama3.2:3b
# Start Ollama server: ollama serve (if not already running)

cargo run --package basic-example -- --usecase token-tracking-ollama
```

This example demonstrates:
- Token tracking with local LLM (Ollama)
- Zero cost inference ($0.00 vs OpenAI costs)
- Same token tracking capabilities without API keys
- Privacy-first approach with on-device inference

### ReAct Token Tracking with Ollama

Track token usage in a ReAct agent with tool calls using local Ollama:

```sh
# Requires Ollama setup (see above)
cargo run --package basic-example -- --usecase react-token-tracking-ollama
```

This example demonstrates:
- Multi-turn reasoning with local LLM
- Token tracking across tool calls (calculator) without API costs
- Cost comparison showing savings vs cloud APIs
- Complete privacy with local inference
