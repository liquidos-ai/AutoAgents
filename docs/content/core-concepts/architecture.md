---
title: Architecture Overview
slug: /architecture
description: Understand the main layers in AutoAgents, from executors and tools to runtimes and providers.
---

# Architecture Overview

AutoAgents is built with a modular, extensible architecture that prioritizes performance, safety, and developer experience. This document provides a comprehensive overview of the framework's design and core components.

## High-Level Architecture
Key layers:

- Agent Definition: your agent’s metadata, tools, and output
- Executors: Basic (single‑turn), ReAct (multi‑turn with direct tool calls), and CodeAct (multi‑turn with sandboxed TypeScript tool composition)
- Memory: context storage (e.g., sliding window)
- Tools/MCP: capabilities the agent can call
- Runtime: optional actor system for multi‑agent workflows
- Providers: pluggable LLM backends (cloud/local)

In practice, execution starts with an agent definition. That definition is paired
with an executor, which decides how the task is handled, whether the run stays
single-turn or enters a multi-step reasoning loop with tool calls. During a run,
the executor can read from memory, invoke tools or MCP-backed capabilities, and
dispatch prompts to an LLM provider.

Direct agents execute this flow inline and return results to the caller. Actor
agents add the runtime layer, which lets multiple agents communicate through
topics, coordinate background work, and participate in larger workflows. The
runtime sits above the provider layer, so the same agent logic can target cloud
or local models without changing the core execution model.

This separation is what keeps AutoAgents modular: agent behavior, execution
strategy, memory, tools, runtime orchestration, and model providers can evolve
independently while still composing into one consistent system.
