# Architecture Overview

AutoAgents is built with a modular, extensible architecture that prioritizes performance, safety, and developer experience. This document provides a comprehensive overview of the framework's design and core components.

## High-Level Architecture
Key layers:

- Agent Definition: your agent’s metadata, tools, and output
- Executors: Basic (single‑turn) and ReAct (multi‑turn with tools, streaming)
- Memory: context storage (e.g., sliding window)
- Tools/MCP: capabilities the agent can call
- Runtime: optional actor system for multi‑agent workflows
- Providers: pluggable LLM backends (cloud/local)
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
