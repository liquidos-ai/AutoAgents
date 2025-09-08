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
