<div align="center">
  <img src="assets/logo.png" alt="AutoAgents Logo" width="200" height="200">

# AutoAgents

**面向生产的 Rust 多智能体框架**

[![Crates.io](https://img.shields.io/crates/v/autoagents.svg)](https://crates.io/crates/autoagents)
[![Documentation](https://docs.rs/autoagents/badge.svg)](https://liquidos-ai.github.io/AutoAgents)
[![License](https://img.shields.io/crates/l/autoagents.svg)](https://github.com/liquidos-ai/AutoAgents#license)
[![Build Status](https://github.com/liquidos-ai/AutoAgents/workflows/coverage/badge.svg)](https://github.com/liquidos-ai/AutoAgents/actions)
[![codecov](https://codecov.io/gh/liquidos-ai/AutoAgents/graph/badge.svg)](https://codecov.io/gh/liquidos-ai/AutoAgents)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/liquidos-ai/AutoAgents)

[English](README.md) | [中文](README.zh-CN.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [한국어](README.ko.md) | [Português (Brasil)](README.pt-BR.md)
<br />
<sub>该译文由社区维护，可能滞后；如有差异，以英文版为准。</sub>

[文档](https://liquidos-ai.github.io/AutoAgents/) | [示例](examples/) | [贡献指南](CONTRIBUTING.md)

<br />
<strong>喜欢这个项目？</strong> <a href="https://github.com/liquidos-ai/AutoAgents">在 GitHub 给我们点星</a>
</div>

---

## 概览

AutoAgents 是一个用于在 Rust 中构建智能系统的模块化多智能体框架。它结合了类型安全的智能体模型、结构化工具调用、可配置内存，以及可插拔的 LLM 后端。该架构面向性能、安全与可组合性，覆盖服务器与边缘环境。

---

## 关键特性

- **智能体执行**：ReAct 与基础执行器、流式响应、结构化输出
- **工具化**：工具与输出的派生宏，以及用于工具执行的沙盒化 WASM 运行时
- **记忆**：滑动窗口记忆与可扩展后端
- **LLM 提供方**：统一接口下的云端与本地后端
- **多智能体编排**：类型化的发布/订阅通信与环境管理
- **语音处理**：本地 TTS 与 STT 支持
- **可观测性**：OpenTelemetry 追踪与指标，支持可插拔导出器

---

## 支持的 LLM 提供方

### 云端提供方

| 提供方          | 状态 |
| --------------- | ---- |
| **OpenAI**      | ✅   |
| **OpenRouter**  | ✅   |
| **Anthropic**   | ✅   |
| **DeepSeek**    | ✅   |
| **xAI**         | ✅   |
| **Phind**       | ✅   |
| **Groq**        | ✅   |
| **Google**      | ✅   |
| **Azure OpenAI** | ✅   |
| **MiniMax**     | ✅   |

### 本地提供方

| 提供方       | 状态 |
| ------------ | ---- |
| **Ollama**   | ✅   |
| **Mistral-rs** | ✅ |
| **Llama-Cpp** | ✅ |

### 实验性提供方

详见 https://github.com/liquidos-ai/AutoAgents-Experimental-Backends

| 提供方 | 状态            |
| ------ | --------------- |
| **Burn** | ⚠️ 实验性     |
| **Onnx** | ⚠️ 实验性     |

提供方支持会根据社区需求持续扩展。

---

## 基准测试

![Benchmark](./assets/Benchmark.png)

更多信息见 [GitHub](https://github.com/liquidos-ai/autoagents-bench)

---

## 安装

### 先决条件

- **Rust**（推荐最新稳定版）
- **Cargo** 包管理器
- **LeftHook** 用于 Git hooks 管理

### 安装 LeftHook

macOS（Homebrew）：

```bash
brew install lefthook
```

Linux/Windows（npm）：

```bash
npm install -g lefthook
```

### 克隆并构建

```bash
git clone https://github.com/liquidos-ai/AutoAgents.git
cd AutoAgents
lefthook install
cargo build --workspace --all-features
```

### 运行测试

```bash
cargo test --workspace --features default --exclude autoagents-burn --exclude autoagents-mistral-rs --exclude wasm_agent
```

---

## 快速开始

```rust
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::{ReActAgent, ReActAgentOutput};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentDeriveT, AgentOutputT, DirectAgent};
use autoagents::core::error::Error;
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents::llm::LLMProvider;
use autoagents::llm::backends::openai::OpenAI;
use autoagents::llm::builder::LLMBuilder;
use autoagents_derive::{agent, tool, AgentHooks, AgentOutput, ToolInput};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct AdditionArgs {
    #[input(description = "Left Operand for addition")]
    left: i64,
    #[input(description = "Right Operand for addition")]
    right: i64,
}

#[tool(
    name = "Addition",
    description = "Use this tool to Add two numbers",
    input = AdditionArgs,
)]
struct Addition {}

#[async_trait]
impl ToolRuntime for Addition {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        println!("execute tool: {:?}", args);
        let typed_args: AdditionArgs = serde_json::from_value(args)?;
        let result = typed_args.left + typed_args.right;
        Ok(result.into())
    }
}

#[derive(Debug, Serialize, Deserialize, AgentOutput)]
pub struct MathAgentOutput {
    #[output(description = "The addition result")]
    value: i64,
    #[output(description = "Explanation of the logic")]
    explanation: String,
    #[output(description = "If user asks other than math questions, use this to answer them.")]
    generic: Option<String>,
}

#[agent(
    name = "math_agent",
    description = "You are a Math agent",
    tools = [Addition],
    output = MathAgentOutput,
)]
#[derive(Default, Clone, AgentHooks)]
pub struct MathAgent {}

impl From<ReActAgentOutput> for MathAgentOutput {
    fn from(output: ReActAgentOutput) -> Self {
        let resp = output.response;
        if output.done && !resp.trim().is_empty() {
            if let Ok(value) = serde_json::from_str::<MathAgentOutput>(&resp) {
                return value;
            }
        }
        MathAgentOutput {
            value: 0,
            explanation: resp,
            generic: None,
        }
    }
}

pub async fn simple_agent(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent_handle = AgentBuilder::<_, DirectAgent>::new(ReActAgent::new(MathAgent {}))
        .llm(llm)
        .memory(sliding_window_memory)
        .build()
        .await?;

    let result = agent_handle.agent.run(Task::new("What is 1 + 1?")).await?;
    println!("Result: {:?}", result);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or("".into());

    let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(api_key)
        .model("gpt-4o")
        .max_tokens(512)
        .temperature(0.2)
        .build()
        .expect("Failed to build LLM");

    let _ = simple_agent(llm).await?;
    Ok(())
}
```

### AutoAgents CLI

AutoAgents CLI 用于根据 YAML 配置运行智能体工作流，并通过 HTTP 提供服务。可在 https://github.com/liquidos-ai/AutoAgents-CLI 查看。

---

## 示例

浏览示例快速上手：

### [基础](examples/basic/)

演示多种示例，如带工具的简单智能体、非常基础的智能体、边缘智能体、链式调用、Actor 模型、流式响应以及添加 Agent Hooks。

### [MCP 集成](examples/mcp/)

演示如何将 AutoAgents 与 Model Context Protocol (MCP) 集成。

### [本地模型](examples/mistral_rs)

演示如何将 AutoAgents 与 Mistral-rs 集成以使用本地模型。

### [设计模式](examples/design_patterns/)

演示链式、规划、路由、并行与反思等设计模式。

### [提供方](examples/providers/)

包含如何使用不同 LLM 提供方的示例。

### [WASM 工具执行](examples/wasm_runner/)

一个可在 WASM 运行时中执行工具的简单智能体。

### [Coding Agent](examples/coding_agent/)

一个基于 ReAct 的复杂编码智能体，具备文件操作能力。

### [语音](examples/speech/)

运行 AutoAgents 语音示例，支持实时 TTS 与 STT。

### [Android 本地智能体](https://github.com/liquidos-ai/AutoAgents-Android-Example)

在 Android 上使用 AutoAgents-llamacpp 后端运行本地模型的示例应用。

---

## 组件

AutoAgents 采用模块化架构：

```
AutoAgents/
├── crates/
│   ├── autoagents/                # Main library entry point
│   ├── autoagents-core/           # Core agent framework
│   ├── autoagents-protocol/       # Shared protocol/event types
│   ├── autoagents-llm/            # LLM provider implementations
│   ├── autoagents-telemetry/      # OpenTelemetry integration
│   ├── autoagents-toolkit/        # Collection of ready-to-use tools
│   ├── autoagents-mistral-rs/     # LLM provider implementations using Mistral-rs
│   ├── autoagents-llamacpp/       # LLM provider implementation using LlamaCpp
│   ├── autoagents-speech/         # Speech model support for TTS and STT
│   ├── autoagents-qdrant/         # Qdrant vector store
│   └── autoagents-derive/         # Procedural macros
├── examples/                      # Example implementations
```

### 核心组件

- **智能体**：智能能力的基本单元
- **环境**：管理智能体生命周期与通信
- **记忆**：可配置的记忆系统
- **工具**：外部能力集成
- **执行器**：不同推理模式（ReAct、Chain-of-Thought）

---

## 开发

### 运行测试

```bash
cargo test --workspace --features default --exclude autoagents-burn --exclude autoagents-mistral-rs --exclude wasm_agent

# Coverage (requires cargo-tarpaulin)
cargo install cargo-tarpaulin
cargo tarpaulin --all-features --out html
```

### 运行基准测试

```bash
cargo bench -p autoagents-core --bench agent_runtime
```

### Git Hooks

本项目使用 LeftHook 进行 Git hooks 管理。Hooks 将自动：

- 使用 `cargo fmt --check` 格式化代码
- 使用 `cargo clippy -- -D warnings` 运行静态检查
- 使用 `cargo test --all-features --workspace --exclude autoagents-burn` 执行测试

### 贡献

欢迎贡献。详情请参阅 [贡献指南](CONTRIBUTING.md) 和 [行为准则](CODE_OF_CONDUCT.md)。

---

## 文档

- **[API 文档](https://liquidos-ai.github.io/AutoAgents)**：完整框架文档
- **[示例](examples/)**：实用实现示例

---

## 社区

- **GitHub Issues**：Bug 报告与功能请求
- **Discussions**：社区问答与想法交流
- **Discord**：加入我们的 Discord 社区 https://discord.gg/zfAF9MkEtK

---

## 性能

AutoAgents 面向高性能设计：

- **内存高效**：通过可配置后端优化内存使用
- **并发**：完整的 tokio async/await 支持
- **可扩展**：多智能体协作的水平扩展能力
- **类型安全**：Rust 类型系统的编译期保障

---

## 许可证

AutoAgents 采用双许可证：

- **MIT License**（[MIT_LICENSE](MIT_LICENSE)）
- **Apache License 2.0**（[APACHE_LICENSE](APACHE_LICENSE)）

你可以根据使用场景选择其一。

---

## 致谢

由 [Liquidos AI](https://liquidos.ai) 团队与优秀社区研究者和工程师构建。

<a href="https://github.com/liquidos-ai/AutoAgents/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=liquidos-ai/AutoAgents" />
</a>

特别感谢：

- Rust 社区提供的优秀生态
- LLM 提供方带来的高质量模型 API
- 所有帮助改进 AutoAgents 的贡献者

---

## Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=liquidos-ai/AutoAgents&type=Date)](https://www.star-history.com/#liquidos-ai/AutoAgents&Date)
