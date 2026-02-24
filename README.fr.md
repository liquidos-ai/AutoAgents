<div align="center">
  <img src="assets/logo.png" alt="AutoAgents Logo" width="200" height="200">

# AutoAgents

**Un framework multi-agents de niveau production en Rust**

[![Crates.io](https://img.shields.io/crates/v/autoagents.svg)](https://crates.io/crates/autoagents)
[![Documentation](https://docs.rs/autoagents/badge.svg)](https://liquidos-ai.github.io/AutoAgents)
[![License](https://img.shields.io/crates/l/autoagents.svg)](https://github.com/liquidos-ai/AutoAgents#license)
[![Build Status](https://github.com/liquidos-ai/AutoAgents/workflows/coverage/badge.svg)](https://github.com/liquidos-ai/AutoAgents/actions)
[![codecov](https://codecov.io/gh/liquidos-ai/AutoAgents/graph/badge.svg)](https://codecov.io/gh/liquidos-ai/AutoAgents)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/liquidos-ai/AutoAgents)

[English](README.md) | [中文](README.zh-CN.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [한국어](README.ko.md) | [Português (Brasil)](README.pt-BR.md)
<br />
<sub>Cette traduction est maintenue par la communauté et peut être en retard ; en cas de divergence, la version anglaise fait foi.</sub>

[Documentation](https://liquidos-ai.github.io/AutoAgents/) | [Exemples](examples/) | [Contribuer](CONTRIBUTING.md)

<br />
<strong>Vous aimez ce projet ?</strong> <a href="https://github.com/liquidos-ai/AutoAgents">Mettez une étoile sur GitHub</a>
</div>

---

## Vue d'ensemble

AutoAgents est un framework multi-agents modulaire pour construire des systèmes intelligents en Rust. Il combine un modèle d'agents typé de manière sûre, des appels d'outils structurés, une mémoire configurable et des backends LLM interchangeables. L'architecture est conçue pour la performance, la sécurité et la composabilité, sur serveur comme sur edge.

---

## Fonctionnalités clés

- **Exécution d'agents** : ReAct et exécuteurs de base, réponses en streaming et sorties structurées
- **Outils** : macros dérivées pour outils et sorties, plus un runtime WASM sandboxé pour l'exécution d'outils
- **Mémoire** : mémoire à fenêtre glissante avec backends extensibles
- **Fournisseurs LLM** : backends cloud et locaux derrière une interface unifiée
- **Orchestration multi-agents** : communication pub/sub typée et gestion d'environnement
- **Traitement vocal** : support TTS et STT en local
- **Observabilité** : traces et métriques OpenTelemetry avec exporteurs plug-and-play

---

## Fournisseurs LLM pris en charge

### Fournisseurs cloud

| Fournisseur      | Statut |
| ---------------- | ------ |
| **OpenAI**       | ✅     |
| **OpenRouter**   | ✅     |
| **Anthropic**    | ✅     |
| **DeepSeek**     | ✅     |
| **xAI**          | ✅     |
| **Phind**        | ✅     |
| **Groq**         | ✅     |
| **Google**       | ✅     |
| **Azure OpenAI** | ✅     |
| **MiniMax**      | ✅     |

### Fournisseurs locaux

| Fournisseur   | Statut |
| ------------ | ------ |
| **Ollama**   | ✅     |
| **Mistral-rs** | ✅   |
| **Llama-Cpp** | ✅    |

### Fournisseurs expérimentaux

Voir https://github.com/liquidos-ai/AutoAgents-Experimental-Backends

| Fournisseur | Statut           |
| ---------- | ---------------- |
| **Burn**   | ⚠️ Expérimental |
| **Onnx**   | ⚠️ Expérimental |

La prise en charge des fournisseurs s'étend activement selon les besoins de la communauté.

---

## Benchmarks

![Benchmark](./assets/Benchmark.png)

Plus d'infos sur [GitHub](https://github.com/liquidos-ai/autoagents-bench)

---

## Installation

### Prérequis

- **Rust** (dernière version stable recommandée)
- **Cargo** comme gestionnaire de paquets
- **LeftHook** pour la gestion des Git hooks

### Installer LeftHook

macOS (Homebrew) :

```bash
brew install lefthook
```

Linux/Windows (npm) :

```bash
npm install -g lefthook
```

### Cloner et compiler

```bash
git clone https://github.com/liquidos-ai/AutoAgents.git
cd AutoAgents
lefthook install
cargo build --workspace --all-features
```

### Exécuter les tests

```bash
cargo test --workspace --features default --exclude autoagents-burn --exclude autoagents-mistral-rs --exclude wasm_agent
```

---

## Démarrage rapide

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

AutoAgents CLI permet d'exécuter des workflows d'agents à partir de configurations YAML et de les exposer via HTTP. Voir https://github.com/liquidos-ai/AutoAgents-CLI.

---

## Exemples

Explorez les exemples pour démarrer rapidement :

### [Basique](examples/basic/)

Présente des exemples comme un agent simple avec outils, un agent très basique, un agent edge, le chaînage, un modèle basé sur les acteurs, le streaming et l'ajout d'Agent Hooks.

### [Intégration MCP](examples/mcp/)

Montre comment intégrer AutoAgents avec le Model Context Protocol (MCP).

### [Modèles locaux](examples/mistral_rs)

Montre comment intégrer AutoAgents avec Mistral-rs pour des modèles locaux.

### [Patrons de conception](examples/design_patterns/)

Montre les patrons comme le chaînage, la planification, le routage, le parallélisme et la réflexion.

### [Fournisseurs](examples/providers/)

Contient des exemples d'utilisation de différents fournisseurs LLM avec AutoAgents.

### [Exécution d'outils WASM](examples/wasm_runner/)

Un agent simple capable d'exécuter des outils dans un runtime WASM.

### [Agent de codage](examples/coding_agent/)

Un agent de codage sophistiqué basé sur ReAct avec des capacités de manipulation de fichiers.

### [Voix](examples/speech/)

Exécute l'exemple audio AutoAgents avec TTS et STT en temps réel.

### [Agent local Android](https://github.com/liquidos-ai/AutoAgents-Android-Example)

Application exemple qui exécute AutoAgents avec des modèles locaux sur Android via le backend autoagents-llamacpp.

---

## Composants

AutoAgents est construit avec une architecture modulaire :

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

### Composants principaux

- **Agent** : l'unité fondamentale d'intelligence
- **Environnement** : gère le cycle de vie et la communication des agents
- **Mémoire** : systèmes de mémoire configurables
- **Outils** : intégration de capacités externes
- **Exécuteurs** : différents schémas de raisonnement (ReAct, Chain-of-Thought)

---

## Développement

### Exécuter les tests

```bash
cargo test --workspace --features default --exclude autoagents-burn --exclude autoagents-mistral-rs --exclude wasm_agent

# Coverage (requires cargo-tarpaulin)
cargo install cargo-tarpaulin
cargo tarpaulin --all-features --out html
```

### Exécuter les benchmarks

```bash
cargo bench -p autoagents-core --bench agent_runtime
```

### Git Hooks

Ce projet utilise LeftHook pour la gestion des Git hooks. Les hooks exécutent automatiquement :

- Formatage avec `cargo fmt --check`
- Linting avec `cargo clippy -- -D warnings`
- Tests avec `cargo test --all-features --workspace --exclude autoagents-burn`

### Contribuer

Nous accueillons les contributions. Voir le [Guide de contribution](CONTRIBUTING.md) et le [Code de conduite](CODE_OF_CONDUCT.md).

---

## Documentation

- **[Documentation API](https://liquidos-ai.github.io/AutoAgents)** : documentation complète du framework
- **[Exemples](examples/)** : implémentations pratiques

---

## Communauté

- **GitHub Issues** : rapports de bugs et demandes de fonctionnalités
- **Discussions** : Q&A et idées de la communauté
- **Discord** : rejoignez notre communauté https://discord.gg/zfAF9MkEtK

---

## Performance

AutoAgents est conçu pour de hautes performances :

- **Efficacité mémoire** : usage optimisé avec des backends configurables
- **Concurrence** : prise en charge complète d'async/await avec tokio
- **Scalabilité** : montée en charge horizontale via la coordination multi-agents
- **Sûreté de types** : garanties à la compilation grâce au système de types de Rust

---

## Licence

AutoAgents est sous double licence :

- **MIT License** ([MIT_LICENSE](MIT_LICENSE))
- **Apache License 2.0** ([APACHE_LICENSE](APACHE_LICENSE))

Vous pouvez choisir celle qui convient à votre usage.

---

## Remerciements

Construit par l'équipe [Liquidos AI](https://liquidos.ai) et une formidable communauté de chercheurs et d'ingénieurs.

<a href="https://github.com/liquidos-ai/AutoAgents/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=liquidos-ai/AutoAgents" />
</a>

Remerciements particuliers :

- La communauté Rust pour son excellent écosystème
- Les fournisseurs LLM pour les APIs de modèles de haute qualité
- Tous les contributeurs qui améliorent AutoAgents

---

## Historique des étoiles

[![Star History Chart](https://api.star-history.com/svg?repos=liquidos-ai/AutoAgents&type=Date)](https://www.star-history.com/#liquidos-ai/AutoAgents&Date)
