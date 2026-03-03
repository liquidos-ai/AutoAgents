<div align="center">
  <img src="assets/logo.png" alt="AutoAgents Logo" width="200" height="200">

# AutoAgents

**Um framework multiagente de nível produção em Rust**

[![Crates.io](https://img.shields.io/crates/v/autoagents.svg)](https://crates.io/crates/autoagents)
[![Documentation](https://docs.rs/autoagents/badge.svg)](https://liquidos-ai.github.io/AutoAgents)
[![License](https://img.shields.io/crates/l/autoagents.svg)](https://github.com/liquidos-ai/AutoAgents#license)
[![Build Status](https://github.com/liquidos-ai/AutoAgents/workflows/coverage/badge.svg)](https://github.com/liquidos-ai/AutoAgents/actions)
[![codecov](https://codecov.io/gh/liquidos-ai/AutoAgents/graph/badge.svg)](https://codecov.io/gh/liquidos-ai/AutoAgents)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/liquidos-ai/AutoAgents)

[English](README.md) | [中文](README.zh-CN.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [한국어](README.ko.md) | [Português (Brasil)](README.pt-BR.md)
<br />
<sub>Esta tradução é mantida pela comunidade e pode ficar desatualizada; em caso de divergência, a versão em inglês é a referência.</sub>

[Documentação](https://liquidos-ai.github.io/AutoAgents/) | [Exemplos](examples/) | [Contribuir](CONTRIBUTING.md)

<br />
<strong>Gostou do projeto?</strong> <a href="https://github.com/liquidos-ai/AutoAgents">Dê uma estrela no GitHub</a>
</div>

---

## Visão geral

AutoAgents é um framework modular multiagente para construir sistemas inteligentes em Rust. Ele combina um modelo de agentes com segurança de tipos, chamadas de ferramentas estruturadas, memória configurável e backends de LLM plugáveis. A arquitetura é projetada para desempenho, segurança e composabilidade em servidores e edge.

---

## Principais recursos

- **Execução de agentes**: ReAct e executores básicos, respostas em streaming e saídas estruturadas
- **Ferramentas**: macros derivadas para ferramentas e saídas, além de um runtime WASM sandboxed para execução de ferramentas
- **Memória**: memória de janela deslizante com backends extensíveis
- **Provedores LLM**: backends em nuvem e locais sob uma interface unificada
- **Orquestração multiagente**: comunicação pub/sub tipada e gerenciamento de ambiente
- **Processamento de fala**: suporte local a TTS e STT
- **Observabilidade**: tracing e métricas OpenTelemetry com exportadores plugáveis

---

## Provedores de LLM suportados

### Provedores em nuvem

| Provedor         | Status |
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

### Provedores locais

| Provedor      | Status |
| ------------- | ------ |
| **Ollama**    | ✅     |
| **Mistral-rs**| ✅     |
| **Llama-Cpp** | ✅     |

### Provedores experimentais

Veja https://github.com/liquidos-ai/AutoAgents-Experimental-Backends

| Provedor | Status          |
| -------- | --------------- |
| **Burn** | ⚠️ Experimental |
| **Onnx** | ⚠️ Experimental |

O suporte a provedores continua se expandindo de acordo com as necessidades da comunidade.

---

## Benchmarks

![Benchmark](./assets/Benchmark.png)

Mais informações em [GitHub](https://github.com/liquidos-ai/autoagents-bench)

---

## Instalação

### Pré-requisitos

- **Rust** (última versão estável recomendada)
- **Cargo** como gerenciador de pacotes
- **LeftHook** para gerenciamento de Git hooks

### Prerequisite

```bash
sudo apt update
sudo apt install build-essential libasound2-dev alsa-utils pkg-config libssl-dev -y
```

### Instalar LeftHook

macOS (Homebrew):

```bash
brew install lefthook
```

Linux/Windows (npm):

```bash
npm install -g lefthook
```

### Clonar e compilar

```bash
git clone https://github.com/liquidos-ai/AutoAgents.git
cd AutoAgents
lefthook install
cargo build --workspace --all-features
```

### Executar testes

```bash
cargo test --workspace --features default --exclude autoagents-burn --exclude autoagents-mistral-rs --exclude wasm_agent
```

---

## Início rápido

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

AutoAgents CLI ajuda a executar fluxos de trabalho de agentes a partir de configurações YAML e expô-los via HTTP. Veja em https://github.com/liquidos-ai/AutoAgents-CLI.

---

## Exemplos

Explore os exemplos para começar rapidamente:

### [Básico](examples/basic/)

Demonstra diversos exemplos como agente simples com ferramentas, agente muito básico, agente edge, encadeamento, modelo baseado em atores, streaming e adição de Agent Hooks.

### [Integração MCP](examples/mcp/)

Demonstra como integrar AutoAgents com o Model Context Protocol (MCP).

### [Modelos locais](examples/mistral_rs)

Demonstra como integrar AutoAgents com Mistral-rs para modelos locais.

### [Padrões de design](examples/design_patterns/)

Demonstra padrões como encadeamento, planejamento, roteamento, paralelismo e reflexão.

### [Provedores](examples/providers/)

Contém exemplos de como usar diferentes provedores LLM com AutoAgents.

### [Execução de ferramentas WASM](examples/wasm_runner/)

Um agente simples que pode executar ferramentas em um runtime WASM.

### [Agente de código](examples/coding_agent/)

Um agente de programação sofisticado baseado em ReAct com capacidade de manipulação de arquivos.

### [Voz](examples/speech/)

Execute o exemplo de voz do AutoAgents com TTS e STT em tempo real.

### [Agente local Android](https://github.com/liquidos-ai/AutoAgents-Android-Example)

Aplicativo de exemplo que executa AutoAgents com modelos locais no Android usando o backend autoagents-llamacpp.

---

## Componentes

AutoAgents é construído com uma arquitetura modular:

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

### Componentes principais

- **Agente**: unidade fundamental de inteligência
- **Ambiente**: gerencia o ciclo de vida e a comunicação dos agentes
- **Memória**: sistemas de memória configuráveis
- **Ferramentas**: integração de capacidades externas
- **Executores**: diferentes padrões de raciocínio (ReAct, Chain-of-Thought)

---

## Desenvolvimento

### Executar testes

```bash
cargo test --workspace --features default --exclude autoagents-burn --exclude autoagents-mistral-rs --exclude wasm_agent

# Coverage (requires cargo-tarpaulin)
cargo install cargo-tarpaulin
cargo tarpaulin --all-features --out html
```

### Executar benchmarks

```bash
cargo bench -p autoagents-core --bench agent_runtime
```

### Git Hooks

Este projeto usa LeftHook para gerenciamento de Git hooks. Os hooks executam automaticamente:

- Formatação com `cargo fmt --check`
- Linting com `cargo clippy -- -D warnings`
- Testes com `cargo test --all-features --workspace --exclude autoagents-burn`

### Contribuir

Contribuições são bem-vindas. Consulte o [Guia de Contribuição](CONTRIBUTING.md) e o [Código de Conduta](CODE_OF_CONDUCT.md).

---

## Documentação

- **[Documentação da API](https://liquidos-ai.github.io/AutoAgents)**: documentação completa do framework
- **[Exemplos](examples/)**: implementações práticas

---

## Comunidade

- **GitHub Issues**: relatos de bugs e solicitações de recursos
- **Discussions**: perguntas e respostas da comunidade
- **Discord**: participe da nossa comunidade https://discord.gg/zfAF9MkEtK

---

## Desempenho

AutoAgents é projetado para alto desempenho:

- **Eficiência de memória**: uso otimizado de memória com backends configuráveis
- **Concorrência**: suporte completo a async/await com tokio
- **Escalabilidade**: escalonamento horizontal com coordenação multiagente
- **Segurança de tipos**: garantias em tempo de compilação com o sistema de tipos do Rust

---

## Licença

AutoAgents é dual-licenciado:

- **MIT License** ([MIT_LICENSE](MIT_LICENSE))
- **Apache License 2.0** ([APACHE_LICENSE](APACHE_LICENSE))

Você pode escolher a licença apropriada para seu caso de uso.

---

## Agradecimentos

Construído pela equipe da [Liquidos AI](https://liquidos.ai) e por uma comunidade incrível de pesquisadores e engenheiros.

<a href="https://github.com/liquidos-ai/AutoAgents/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=liquidos-ai/AutoAgents" />
</a>

Agradecimentos especiais:

- A comunidade Rust pelo excelente ecossistema
- Provedores de LLM por habilitar APIs de modelos de alta qualidade
- Todos os contribuidores que ajudam a melhorar o AutoAgents

---

## Histórico de estrelas

[![Star History Chart](https://api.star-history.com/svg?repos=liquidos-ai/AutoAgents&type=Date)](https://www.star-history.com/#liquidos-ai/AutoAgents&Date)
