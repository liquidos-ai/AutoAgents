<div align="center">
  <img src="assets/logo.png" alt="AutoAgents Logo" width="200" height="200">

# AutoAgents

**Ein produktionsreifes Multi-Agenten-Framework in Rust**

[![Crates.io](https://img.shields.io/crates/v/autoagents.svg)](https://crates.io/crates/autoagents)
[![Documentation](https://docs.rs/autoagents/badge.svg)](https://liquidos-ai.github.io/AutoAgents)
[![License](https://img.shields.io/crates/l/autoagents.svg)](https://github.com/liquidos-ai/AutoAgents#license)
[![Build Status](https://github.com/liquidos-ai/AutoAgents/workflows/coverage/badge.svg)](https://github.com/liquidos-ai/AutoAgents/actions)
[![codecov](https://codecov.io/gh/liquidos-ai/AutoAgents/graph/badge.svg)](https://codecov.io/gh/liquidos-ai/AutoAgents)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/liquidos-ai/AutoAgents)

[English](README.md) | [中文](README.zh-CN.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [한국어](README.ko.md) | [Português (Brasil)](README.pt-BR.md)
<br />
<sub>Diese Übersetzung wird von der Community gepflegt und kann hinterherhinken; bei Abweichungen gilt die englische Version.</sub>

[Dokumentation](https://liquidos-ai.github.io/AutoAgents/) | [Beispiele](examples/) | [Mitwirken](CONTRIBUTING.md)

<br />
<strong>Gefällt dir dieses Projekt?</strong> <a href="https://github.com/liquidos-ai/AutoAgents">Gib uns einen Stern auf GitHub</a>
</div>

---

## Überblick

AutoAgents ist ein modulares Multi-Agenten-Framework zum Aufbau intelligenter Systeme in Rust. Es kombiniert ein typsicheres Agentenmodell, strukturierte Tool-Aufrufe, konfigurierbaren Speicher und austauschbare LLM-Backends. Die Architektur ist auf Performance, Sicherheit und Komponierbarkeit ausgelegt, sowohl für Server als auch für Edge.

---

## Wichtige Funktionen

- **Agentenausführung**: ReAct und grundlegende Ausführungsmodelle, Streaming-Antworten und strukturierte Ausgaben
- **Tooling**: Derive-Makros für Tools und Ausgaben sowie eine sandboxed WASM-Runtime für die Tool-Ausführung
- **Speicher**: Sliding-Window-Speicher mit erweiterbaren Backends
- **LLM-Anbieter**: Cloud- und lokale Backends hinter einer einheitlichen Schnittstelle
- **Multi-Agenten-Orchestrierung**: Typisierte Pub/Sub-Kommunikation und Umgebungsverwaltung
- **Sprachverarbeitung**: Lokale TTS- und STT-Unterstützung
- **Observability**: OpenTelemetry-Tracing und Metriken mit austauschbaren Exportern

---

## Unterstützte LLM-Anbieter

### Cloud-Anbieter

| Anbieter          | Status |
| ----------------- | ------ |
| **OpenAI**        | ✅     |
| **OpenRouter**    | ✅     |
| **Anthropic**     | ✅     |
| **DeepSeek**      | ✅     |
| **xAI**           | ✅     |
| **Phind**         | ✅     |
| **Groq**          | ✅     |
| **Google**        | ✅     |
| **Azure OpenAI**  | ✅     |
| **MiniMax**       | ✅     |

### Lokale Anbieter

| Anbieter      | Status |
| ------------ | ------ |
| **Ollama**   | ✅     |
| **Mistral-rs** | ✅   |
| **Llama-Cpp** | ✅    |

### Experimentelle Anbieter

Siehe https://github.com/liquidos-ai/AutoAgents-Experimental-Backends

| Anbieter | Status            |
| ------- | ----------------- |
| **Burn** | ⚠️ Experimentell |
| **Onnx** | ⚠️ Experimentell |

Die Anbieter-Unterstützung wird aktiv anhand der Community-Bedürfnisse erweitert.

---

## Benchmarks

![Benchmark](./assets/Benchmark.png)

Weitere Infos auf [GitHub](https://github.com/liquidos-ai/autoagents-bench)

---

## Installation

### Voraussetzungen

- **Rust** (neueste stabile Version empfohlen)
- **Cargo** als Paketmanager
- **LeftHook** zur Verwaltung von Git Hooks

### Prerequisite

```bash
sudo apt update
sudo apt install build-essential libasound2-dev alsa-utils pkg-config libssl-dev -y
```

### LeftHook installieren

macOS (Homebrew):

```bash
brew install lefthook
```

Linux/Windows (npm):

```bash
npm install -g lefthook
```

### Klonen und bauen

```bash
git clone https://github.com/liquidos-ai/AutoAgents.git
cd AutoAgents
lefthook install
cargo build --workspace --all-features
```

### Tests ausführen

```bash
cargo test --workspace --features default --exclude autoagents-burn --exclude autoagents-mistral-rs --exclude wasm_agent
```

---

## Schnellstart

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

AutoAgents CLI hilft beim Ausführen von Agenten-Workflows aus YAML-Konfigurationen und stellt sie über HTTP bereit. Siehe https://github.com/liquidos-ai/AutoAgents-CLI.

---

## Beispiele

Entdecke die Beispiele für einen schnellen Einstieg:

### [Basic](examples/basic/)

Zeigt verschiedene Beispiele wie einen einfachen Agenten mit Tools, einen sehr einfachen Agenten, Edge-Agenten, Chaining, ein Actor-basiertes Modell, Streaming und das Hinzufügen von Agent Hooks.

### [MCP-Integration](examples/mcp/)

Zeigt, wie AutoAgents mit dem Model Context Protocol (MCP) integriert wird.

### [Lokale Modelle](examples/mistral_rs)

Zeigt die Integration von AutoAgents mit Mistral-rs für lokale Modelle.

### [Entwurfsmuster](examples/design_patterns/)

Zeigt Muster wie Chaining, Planung, Routing, Parallelisierung und Reflexion.

### [Provider](examples/providers/)

Enthält Beispiele zur Nutzung verschiedener LLM-Provider mit AutoAgents.

### [WASM-Tool-Ausführung](examples/wasm_runner/)

Ein einfacher Agent, der Tools in einer WASM-Runtime ausführen kann.

### [Coding Agent](examples/coding_agent/)

Ein anspruchsvoller ReAct-basierter Coding-Agent mit Dateimanipulationsfunktionen.

### [Sprache](examples/speech/)

Führt das AutoAgents-Sprachbeispiel mit Echtzeit-TTS und STT aus.

### [Android Local Agent](https://github.com/liquidos-ai/AutoAgents-Android-Example)

Beispiel-App, die AutoAgents mit lokalen Modellen auf Android über das autoagents-llamacpp-Backend ausführt.

---

## Komponenten

AutoAgents ist modular aufgebaut:

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

### Kernkomponenten

- **Agent**: grundlegende Einheit der Intelligenz
- **Umgebung**: verwaltet Lebenszyklus und Kommunikation von Agenten
- **Speicher**: konfigurierbare Speichersysteme
- **Tools**: Integration externer Fähigkeiten
- **Executors**: verschiedene Denkmodelle (ReAct, Chain-of-Thought)

---

## Entwicklung

### Tests ausführen

```bash
cargo test --workspace --features default --exclude autoagents-burn --exclude autoagents-mistral-rs --exclude wasm_agent

# Coverage (requires cargo-tarpaulin)
cargo install cargo-tarpaulin
cargo tarpaulin --all-features --out html
```

### Benchmarks ausführen

```bash
cargo bench -p autoagents-core --bench agent_runtime
```

### Git Hooks

Dieses Projekt nutzt LeftHook zur Verwaltung von Git Hooks. Die Hooks führen automatisch aus:

- Formatierung mit `cargo fmt --check`
- Linting mit `cargo clippy -- -D warnings`
- Tests mit `cargo test --all-features --workspace --exclude autoagents-burn`

### Mitwirken

Beiträge sind willkommen. Siehe [Contribution Guidelines](CONTRIBUTING.md) und [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Dokumentation

- **[API-Dokumentation](https://liquidos-ai.github.io/AutoAgents)**: vollständige Framework-Dokumentation
- **[Beispiele](examples/)**: praxisnahe Implementierungen

---

## Community

- **GitHub Issues**: Fehlerberichte und Feature-Wünsche
- **Discussions**: Community Q&A und Ideen
- **Discord**: Tritt unserer Discord-Community bei https://discord.gg/zfAF9MkEtK

---

## Performance

AutoAgents ist auf hohe Performance ausgelegt:

- **Speichereffizient**: optimierte Speichernutzung mit konfigurierbaren Backends
- **Nebenläufig**: vollständige async/await-Unterstützung mit tokio
- **Skalierbar**: horizontale Skalierung durch Multi-Agenten-Koordination
- **Typsicher**: Compile-Time-Garantien durch Rusts Typsystem

---

## Lizenz

AutoAgents ist dual-lizenziert:

- **MIT License** ([MIT_LICENSE](MIT_LICENSE))
- **Apache License 2.0** ([APACHE_LICENSE](APACHE_LICENSE))

Du kannst die passende Lizenz für deinen Anwendungsfall wählen.

---

## Danksagung

Erstellt vom [Liquidos AI](https://liquidos.ai) Team und einer großartigen Community von Forschenden und Ingenieuren.

<a href="https://github.com/liquidos-ai/AutoAgents/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=liquidos-ai/AutoAgents" />
</a>

Besonderer Dank an:

- Die Rust-Community für das hervorragende Ökosystem
- LLM-Provider für hochwertige Modell-APIs
- Alle Mitwirkenden, die AutoAgents verbessern

---

## Star-Verlauf

[![Star History Chart](https://api.star-history.com/svg?repos=liquidos-ai/AutoAgents&type=Date)](https://www.star-history.com/#liquidos-ai/AutoAgents&Date)
