<div align="center">
  <img src="assets/logo.png" alt="AutoAgents Logo" width="200" height="200">

# AutoAgents

**Rustによる本番運用向けマルチエージェントフレームワーク**

[![Crates.io](https://img.shields.io/crates/v/autoagents.svg)](https://crates.io/crates/autoagents)
[![Documentation](https://docs.rs/autoagents/badge.svg)](https://liquidos-ai.github.io/AutoAgents)
[![License](https://img.shields.io/crates/l/autoagents.svg)](https://github.com/liquidos-ai/AutoAgents#license)
[![Build Status](https://github.com/liquidos-ai/AutoAgents/workflows/coverage/badge.svg)](https://github.com/liquidos-ai/AutoAgents/actions)
[![codecov](https://codecov.io/gh/liquidos-ai/AutoAgents/graph/badge.svg)](https://codecov.io/gh/liquidos-ai/AutoAgents)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/liquidos-ai/AutoAgents)

[English](README.md) | [中文](README.zh-CN.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [한국어](README.ko.md) | [Português (Brasil)](README.pt-BR.md)
<br />
<sub>この翻訳はコミュニティにより保守され、英語版より遅れる可能性があります。差異がある場合は英語版が正です。</sub>

[ドキュメント](https://liquidos-ai.github.io/AutoAgents/) | [例](examples/) | [コントリビュート](CONTRIBUTING.md)

<br />
<strong>このプロジェクトが気に入りましたか？</strong> <a href="https://github.com/liquidos-ai/AutoAgents">GitHubでスターを付けてください</a>
</div>

---

## 概要

AutoAgents は、Rust で知的システムを構築するためのモジュール型マルチエージェントフレームワークです。型安全なエージェントモデル、構造化されたツール呼び出し、構成可能なメモリ、差し替え可能な LLM バックエンドを統合しています。アーキテクチャは性能・安全性・合成可能性を重視し、サーバーとエッジの両方を対象に設計されています。

---

## 主な機能

- **エージェント実行**：ReAct と基本エグゼキュータ、ストリーミング応答、構造化出力
- **ツール**：ツールと出力の派生マクロ、ツール実行のためのサンドボックス化 WASM ランタイム
- **メモリ**：スライディングウィンドウメモリと拡張可能なバックエンド
- **LLM プロバイダー**：統一インターフェースによるクラウドとローカルのバックエンド
- **マルチエージェント編成**：型付き pub/sub 通信と環境管理
- **音声処理**：ローカル TTS と STT サポート
- **可観測性**：OpenTelemetry のトレースとメトリクス、プラガブルなエクスポーター

---

## 対応 LLM プロバイダー

### クラウドプロバイダー

| プロバイダー       | 状態 |
| ------------------ | ---- |
| **OpenAI**         | ✅   |
| **OpenRouter**     | ✅   |
| **Anthropic**      | ✅   |
| **DeepSeek**       | ✅   |
| **xAI**            | ✅   |
| **Phind**          | ✅   |
| **Groq**           | ✅   |
| **Google**         | ✅   |
| **Azure OpenAI**   | ✅   |
| **MiniMax**        | ✅   |

### ローカルプロバイダー

| プロバイダー   | 状態 |
| -------------- | ---- |
| **Ollama**     | ✅   |
| **Mistral-rs** | ✅   |
| **Llama-Cpp**  | ✅   |

### 実験的プロバイダー

詳細: https://github.com/liquidos-ai/AutoAgents-Experimental-Backends

| プロバイダー | 状態            |
| ------------ | --------------- |
| **Burn**     | ⚠️ 実験的       |
| **Onnx**     | ⚠️ 実験的       |

プロバイダーの対応はコミュニティの要望に応じて拡張中です。

---

## ベンチマーク

![Benchmark](./assets/Benchmark.png)

詳細は [GitHub](https://github.com/liquidos-ai/autoagents-bench) を参照してください。

---

## インストール

### 前提条件

- **Rust**（最新安定版推奨）
- **Cargo** パッケージマネージャ
- **LeftHook**（Git hooks 管理）

### LeftHook のインストール

macOS（Homebrew）：

```bash
brew install lefthook
```

Linux/Windows（npm）：

```bash
npm install -g lefthook
```

### クローンとビルド

```bash
git clone https://github.com/liquidos-ai/AutoAgents.git
cd AutoAgents
lefthook install
cargo build --workspace --all-features
```

### テストの実行

```bash
cargo test --workspace --features default --exclude autoagents-burn --exclude autoagents-mistral-rs --exclude wasm_agent
```

---

## クイックスタート

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

AutoAgents CLI は YAML 設定からエージェントワークフローを実行し、HTTP で提供します。詳細は https://github.com/liquidos-ai/AutoAgents-CLI を参照してください。

---

## 例

すぐに始めるための例：

### [基本](examples/basic/)

ツール付きのシンプルなエージェント、非常に基本的なエージェント、エッジエージェント、チェーン、Actor ベースモデル、ストリーミング、Agent Hooks の追加などを示します。

### [MCP 連携](examples/mcp/)

AutoAgents を Model Context Protocol (MCP) と統合する方法を示します。

### [ローカルモデル](examples/mistral_rs)

ローカルモデル向けに Mistral-rs と AutoAgents を統合する方法を示します。

### [デザインパターン](examples/design_patterns/)

チェーン、プランニング、ルーティング、並列、リフレクションの各パターンを示します。

### [プロバイダー](examples/providers/)

さまざまな LLM プロバイダーの利用例です。

### [WASM ツール実行](examples/wasm_runner/)

WASM ランタイムでツールを実行できるシンプルなエージェントです。

### [コーディングエージェント](examples/coding_agent/)

ファイル操作機能を備えた ReAct ベースの高度なコーディングエージェントです。

### [音声](examples/speech/)

リアルタイムの TTS と STT を用いた AutoAgents 音声例です。

### [Android ローカルエージェント](https://github.com/liquidos-ai/AutoAgents-Android-Example)

AutoAgents-llamacpp バックエンドを使って Android 上でローカルモデルを動かす例です。

---

## コンポーネント

AutoAgents はモジュール化アーキテクチャで構築されています：

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

### コアコンポーネント

- **エージェント**：知能の基本単位
- **環境**：エージェントのライフサイクルと通信を管理
- **メモリ**：構成可能なメモリシステム
- **ツール**：外部機能の統合
- **エグゼキュータ**：推論パターン（ReAct、Chain-of-Thought）

---

## 開発

### テストの実行

```bash
cargo test --workspace --features default --exclude autoagents-burn --exclude autoagents-mistral-rs --exclude wasm_agent

# Coverage (requires cargo-tarpaulin)
cargo install cargo-tarpaulin
cargo tarpaulin --all-features --out html
```

### ベンチマークの実行

```bash
cargo bench -p autoagents-core --bench agent_runtime
```

### Git Hooks

本プロジェクトは LeftHook による Git hooks 管理を利用しています。フックは自動的に次を実行します：

- `cargo fmt --check` でコードを整形
- `cargo clippy -- -D warnings` で静的解析
- `cargo test --all-features --workspace --exclude autoagents-burn` でテスト実行

### コントリビュート

コントリビュートを歓迎します。詳細は [コントリビュートガイド](CONTRIBUTING.md) と [行動規範](CODE_OF_CONDUCT.md) をご覧ください。

---

## ドキュメント

- **[API ドキュメント](https://liquidos-ai.github.io/AutoAgents)**：フレームワークの完全なドキュメント
- **[例](examples/)**：実用的な実装例

---

## コミュニティ

- **GitHub Issues**：バグ報告と機能要望
- **Discussions**：コミュニティ Q&A とアイデア
- **Discord**：Discord コミュニティに参加 https://discord.gg/zfAF9MkEtK

---

## パフォーマンス

AutoAgents は高性能を目指して設計されています：

- **メモリ効率**：構成可能なバックエンドによる最適化
- **並行性**：tokio による async/await 完全対応
- **スケーラブル**：マルチエージェント協調による水平スケール
- **型安全**：Rust の型システムによるコンパイル時保証

---

## ライセンス

AutoAgents はデュアルライセンスです：

- **MIT License**（[MIT_LICENSE](MIT_LICENSE)）
- **Apache License 2.0**（[APACHE_LICENSE](APACHE_LICENSE)）

利用用途に合わせていずれかを選択できます。

---

## 謝辞

[Liquidos AI](https://liquidos.ai) チームと、素晴らしい研究者・エンジニアコミュニティによって構築されています。

<a href="https://github.com/liquidos-ai/AutoAgents/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=liquidos-ai/AutoAgents" />
</a>

特別な感謝：

- 素晴らしいエコシステムを提供する Rust コミュニティ
- 高品質なモデル API を提供する LLM プロバイダー
- AutoAgents を改善してくれる全ての貢献者

---

## スター履歴

[![Star History Chart](https://api.star-history.com/svg?repos=liquidos-ai/AutoAgents&type=Date)](https://www.star-history.com/#liquidos-ai/AutoAgents&Date)
