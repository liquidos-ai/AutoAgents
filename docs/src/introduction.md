# Introduction

AutoAgents is a modern multi‑agent framework in Rust for building intelligent, autonomous agents powered by Large Language Models (LLMs/SLMs) and [Ractor](https://github.com/slawlor/ractor).

Designed for performance, safety, and scalability, AutoAgents provides a robust foundation for AI systems that can reason, act, remember, and collaborate. You can build cloud‑native agents, edge‑native agents, and hybrid deployments — including WASM for the browser.

## What Is AutoAgents?

AutoAgents helps you create agents that can:

- Reason: Use execution strategies like ReAct and Basic for problem solving
- Act: Call tools and interact with external systems safely
- Remember: Maintain context with configurable memory providers
- Collaborate: Coordinate through an actor runtime and pub/sub topics
---

## High‑Level Architecture
```mermaid
graph TD

    Executor["Executor Layer"]
    Memory["Memory Layer"]
    Agent["Agent Definition"]
    DirectAgent["Direct Agent"]
    ActorAgent["Actor Based Agent"]
    Tools["Tools"]
    MCP["MCP"]
    Runtime["Runtime Engine"]
    Providers["LLM Providers"]
    CloudLLM["Cloud LLM Providers"]
    LocalLLM["Local LLM Providers"]
    Accelerators["Accelerators"]

    Executor --> Agent
    Memory --> Agent
    Tools --> Agent
    MCP --> Agent
    Agent --> ActorAgent
    Agent --> DirectAgent
    ActorAgent --> Runtime
    DirectAgent --> Runtime
    Runtime --> Providers
    Providers --> LocalLLM
    Providers --> CloudLLM
    LocalLLM --> Accelerators
```


## Community and Support

AutoAgents is developed by the [Liquidos AI](https://liquidos.ai) team and maintained by a growing community.

- 📖 Documentation: Guides and reference
- 💬 Discord: [discord.gg/Ghau8xYn](https://discord.gg/Ghau8xYn)
- 🐛 Issues: [GitHub](https://github.com/liquidos-ai/AutoAgents)
- 🤝 Contributing: PRs welcome
