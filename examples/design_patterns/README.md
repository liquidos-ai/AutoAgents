# Agent Design Patterns

This example demonstrates common multi-agent design patterns using the AutoAgents framework. Each pattern showcases
different approaches to organizing and coordinating multiple AI agents to solve complex problems.

These examples are taken from "_Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems, Part 1, By
Antonio
Gulli_". Thanks to Antonio for an amazing comprehensive summary on these design patterns.

## Overview

Multi-agent systems benefit from well-established design patterns that help structure agent interactions, manage
workflows, and handle different types of coordination challenges. This collection implements four fundamental patterns:

- **Chaining**: Sequential processing pipeline
- **Parallel**: Concurrent execution with result synthesis
- **Routing**: Intelligent request classification and delegation
- **Planning**: Strategic planning with adaptive execution
- **Reflection**: Iterative improvement through self-critique

Each pattern addresses different use cases and demonstrates various aspects of agent coordination, communication, and
workflow management.

## Available Patterns

### Chaining Pattern

Sequential agent processing where each agent's output becomes the next agent's input.

**Use Cases**: Data transformation pipelines, document processing workflows, ETL operations

**Flow**: Agent1 → Agent2 → Final Result

```shell
export OPENAI_API_KEY=your_openai_api_key_here 
cargo run --package design_patterns -- --design-pattern chaining
```

### Parallel Pattern

Multiple agents work simultaneously on different aspects of the same problem, with results synthesized by a coordinator
agent.

**Use Cases**: Multi-aspect document analysis, distributed processing, complex analysis requiring multiple perspectives

**Flow**: Input → [Agent1 | Agent2 ] → Synthesis → Final Result

```shell
export OPENAI_API_KEY=your_openai_api_key_here 
cargo run --package design_patterns -- --design-pattern parallel
```

### Routing Pattern

An intelligent dispatcher agent classifies requests and routes them to appropriate specialized handlers.

**Use Cases**: Customer service routing, task delegation systems, multi-domain assistants

**Flow**: Request → Router Agent → [Handler1 | Handler2 | Handler3] → Response

```shell
export OPENAI_API_KEY=your_openai_api_key_here 
cargo run --package design_patterns -- --design-pattern routing
```

### Planning Pattern

Strategic planning agent creates comprehensive multi-step plans, then an executor agent carries out steps with feedback
loops and adaptive replanning.

**Use Cases**: Complex project management, research workflows, multi-stage problem solving

**Flow**: Task → Strategic Planner → Plan → Step-by-step Executor → Final Result

```shell
export OPENAI_API_KEY=your_openai_api_key_here 
cargo run --package design_patterns -- --design-pattern planning
```

### Reflection Pattern

Agents iteratively improve their outputs through self-critique and refinement cycles.

**Use Cases**: Code generation with review, content creation with quality assurance, iterative optimization

**Flow**: Generate → Critique → Refine → Critique → ... → Final Result

```shell
export OPENAI_API_KEY=your_openai_api_key_here 
cargo run --package design_patterns -- --design-pattern reflection
```

## Running Examples

### Prerequisites

1. Set up your OpenAI API key:
   ```shell
   export OPENAI_API_KEY=your_openai_api_key_here
   ```

2. Build the project:
   ```shell
   cargo build --package design_patterns --all-features
   ```

### Run All Patterns

You can run all patterns sequentially to see the differences:

```shell
# Run each pattern
cargo run --package design_patterns -- --design-pattern chaining
cargo run --package design_patterns -- --design-pattern parallel  
cargo run --package design_patterns -- --design-pattern routing
cargo run --package design_patterns -- --design-pattern planning
cargo run --package design_patterns -- --design-pattern reflection
```

These examples demonstrate how to structure multi-agent systems for different coordination patterns, providing a
foundation for building more complex agent workflows.