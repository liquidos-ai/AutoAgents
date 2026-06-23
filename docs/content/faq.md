# FAQ

## General

**What is AutoAgents?**
AutoAgents is a production-grade, multi-agent framework written in Rust. It provides a modular architecture for building intelligent systems with type-safe agent models, structured tool calling, configurable memory, and pluggable LLM backends — designed for performance, safety, and composability across server and edge environments.

**How does AutoAgents differ from other agent frameworks?**
AutoAgents is Rust-first, offering memory safety, zero-cost abstractions, and high performance. It provides a unified interface for cloud and local LLM providers, built-in guardrails, optimization passes (cache/retry), and a WASM sandbox for tool execution — all in a single framework.

**Is there a Python version?**
Yes. AutoAgents provides Python bindings via `autoagents-py` on PyPI, enabling Python developers to leverage the Rust core with a familiar API.

## Setup & Configuration

**How do I install AutoAgents?**
Install via Cargo: `cargo add autoagents`, or via PyPI for Python: `pip install autoagents-py`. See the [documentation](https://liquidos-ai.github.io/AutoAgents/) for detailed setup guides.

**Which LLM providers are supported?**
AutoAgents supports OpenAI, OpenRouter, Anthropic, DeepSeek, xAI, and local models via a unified interface. Configure your API keys in the environment or configuration file.

**Can I use local models?**
Yes. AutoAgents supports local LLM backends through its unified provider interface, enabling fully offline agent operation.

## Agent Development

**What is the ReAct executor?**
The ReAct (Reasoning + Acting) executor is AutoAgents' primary agent execution model. It alternates between reasoning steps and tool calls, enabling agents to plan, execute, and observe results in a loop until the task is complete.

**How does the tool system work?**
Tools are defined using derive macros (`#[derive(Tool)]`) for type-safe input/output. AutoAgents also provides a sandboxed WASM runtime for executing untrusted tools securely.

**What memory backends are available?**
AutoAgents uses a sliding window memory model by default, with extensible backends for custom memory strategies — enabling fine-grained control over context management.

## Multi-Agent Orchestration

**How do agents communicate?**
AutoAgents provides typed pub/sub communication between agents, enabling structured message passing with compile-time type safety. Agents can publish events and subscribe to topics in a decoupled architecture.

**What is the environment system?**
The environment system manages shared state and resources across multiple agents. It provides a controlled space where agents can interact, share observations, and coordinate actions. Register runtimes with `register_runtime`, start them with `run()`, await completion with `wait().await?`, or stop them with `shutdown().await?`. See [Actor Agents — Environment lifecycle](./core-concepts/actor_agents.md#environment-lifecycle) for patterns.

**Can I restart the same runtime after shutdown?**
`SingleThreadedRuntime` is single-cycle: its event loop cannot re-enter after it finishes. To run again, register a new runtime instance (or create a fresh `Environment`).

**Is `wait()` safe inside `tokio::select!`?**
Yes. If another branch wins and `wait()` is cancelled, the join handle is restored on the environment so `shutdown()` can still stop runtimes and join the run task.

## Troubleshooting

**Build fails with Rust version errors. What should I do?**
AutoAgents requires Rust 1.75+. Run `rustup update` to get the latest stable version. Check the [documentation](https://liquidos-ai.github.io/AutoAgents/) for minimum version requirements.

**Where can I get help?**
- Documentation: https://liquidos-ai.github.io/AutoAgents/
- Examples: `examples/` directory in the repository
- DeepWiki: https://deepwiki.com/liquidos-ai/AutoAgents
- GitHub Issues: https://github.com/liquidos-ai/AutoAgents/issues
