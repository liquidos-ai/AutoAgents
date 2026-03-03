<div align="center">
  <img src="assets/logo.png" alt="AutoAgents Logo" width="200" height="200">

# AutoAgents

**Un framework multiagente de grado producción en Rust**

[![Crates.io](https://img.shields.io/crates/v/autoagents.svg)](https://crates.io/crates/autoagents)
[![Documentation](https://docs.rs/autoagents/badge.svg)](https://liquidos-ai.github.io/AutoAgents)
[![License](https://img.shields.io/crates/l/autoagents.svg)](https://github.com/liquidos-ai/AutoAgents#license)
[![Build Status](https://github.com/liquidos-ai/AutoAgents/workflows/coverage/badge.svg)](https://github.com/liquidos-ai/AutoAgents/actions)
[![codecov](https://codecov.io/gh/liquidos-ai/AutoAgents/graph/badge.svg)](https://codecov.io/gh/liquidos-ai/AutoAgents)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/liquidos-ai/AutoAgents)

[English](README.md) | [中文](README.zh-CN.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [한국어](README.ko.md) | [Português (Brasil)](README.pt-BR.md)
<br />
<sub>Esta traducción es mantenida por la comunidad y puede quedar desactualizada; si hay diferencias, la versión en inglés es la referencia.</sub>

[Documentación](https://liquidos-ai.github.io/AutoAgents/) | [Ejemplos](examples/) | [Contribuir](CONTRIBUTING.md)

<br />
<strong>¿Te gusta este proyecto?</strong> <a href="https://github.com/liquidos-ai/AutoAgents">Danos una estrella en GitHub</a>
</div>

---

## Descripción general

AutoAgents es un framework modular multiagente para construir sistemas inteligentes en Rust. Combina un modelo de agentes con seguridad de tipos, llamadas a herramientas estructuradas, memoria configurable y backends de LLM intercambiables. La arquitectura está diseñada para rendimiento, seguridad y composabilidad en servidores y edge.

---

## Características clave

- **Ejecución de agentes**: ReAct y ejecutores básicos, respuestas en streaming y salidas estructuradas
- **Herramientas**: macros derivadas para herramientas y salidas, además de un runtime WASM aislado para ejecutar herramientas
- **Memoria**: memoria de ventana deslizante con backends extensibles
- **Proveedores LLM**: backends en la nube y locales detrás de una interfaz unificada
- **Orquestación multiagente**: comunicación pub/sub tipada y gestión de entornos
- **Procesamiento de voz**: soporte local de TTS y STT
- **Observabilidad**: trazas y métricas OpenTelemetry con exportadores conectables

---

## Proveedores de LLM compatibles

### Proveedores en la nube

| Proveedor        | Estado |
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

### Proveedores locales

| Proveedor      | Estado |
| -------------- | ------ |
| **Ollama**     | ✅     |
| **Mistral-rs** | ✅     |
| **Llama-Cpp**  | ✅     |

### Proveedores experimentales

Ver https://github.com/liquidos-ai/AutoAgents-Experimental-Backends

| Proveedor | Estado           |
| --------- | ---------------- |
| **Burn**  | ⚠️ Experimental |
| **Onnx**  | ⚠️ Experimental |

El soporte de proveedores se expande activamente según las necesidades de la comunidad.

---

## Benchmarks

![Benchmark](./assets/Benchmark.png)

Más información en [GitHub](https://github.com/liquidos-ai/autoagents-bench)

---

## Instalación

### Requisitos previos

- **Rust** (recomendado el último estable)
- **Cargo** como gestor de paquetes
- **LeftHook** para gestionar Git hooks

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

### Clonar y compilar

```bash
git clone https://github.com/liquidos-ai/AutoAgents.git
cd AutoAgents
lefthook install
cargo build --workspace --all-features
```

### Ejecutar pruebas

```bash
cargo test --workspace --features default --exclude autoagents-burn --exclude autoagents-mistral-rs --exclude wasm_agent
```

---

## Inicio rápido

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

AutoAgents CLI ayuda a ejecutar flujos de trabajo de agentes desde configuraciones YAML y a exponerlos por HTTP. Puedes verlo en https://github.com/liquidos-ai/AutoAgents-CLI.

---

## Ejemplos

Explora los ejemplos para comenzar rápidamente:

### [Básico](examples/basic/)

Demuestra varios ejemplos como agente simple con herramientas, agente muy básico, agente en edge, encadenamiento, modelo basado en actores, streaming y agregar Agent Hooks.

### [Integración MCP](examples/mcp/)

Demuestra cómo integrar AutoAgents con Model Context Protocol (MCP).

### [Modelos locales](examples/mistral_rs)

Demuestra cómo integrar AutoAgents con Mistral-rs para modelos locales.

### [Patrones de diseño](examples/design_patterns/)

Demuestra varios patrones como encadenamiento, planificación, enrutamiento, paralelismo y reflexión.

### [Proveedores](examples/providers/)

Contiene ejemplos de cómo usar distintos proveedores LLM con AutoAgents.

### [Ejecución de herramientas WASM](examples/wasm_runner/)

Un agente simple que puede ejecutar herramientas en un runtime WASM.

### [Agente de código](examples/coding_agent/)

Un agente de programación sofisticado basado en ReAct con capacidades de manipulación de archivos.

### [Voz](examples/speech/)

Ejecuta el ejemplo de voz de AutoAgents con TTS y STT en tiempo real.

### [Agente local Android](https://github.com/liquidos-ai/AutoAgents-Android-Example)

Aplicación de ejemplo que ejecuta AutoAgents con modelos locales en Android usando el backend autoagents-llamacpp.

---

## Componentes

AutoAgents está construido con una arquitectura modular:

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

### Componentes principales

- **Agente**: la unidad fundamental de inteligencia
- **Entorno**: gestiona el ciclo de vida y la comunicación entre agentes
- **Memoria**: sistemas de memoria configurables
- **Herramientas**: integración de capacidades externas
- **Ejecutores**: distintos patrones de razonamiento (ReAct, Chain-of-Thought)

---

## Desarrollo

### Ejecutar pruebas

```bash
cargo test --workspace --features default --exclude autoagents-burn --exclude autoagents-mistral-rs --exclude wasm_agent

# Coverage (requires cargo-tarpaulin)
cargo install cargo-tarpaulin
cargo tarpaulin --all-features --out html
```

### Ejecutar benchmarks

```bash
cargo bench -p autoagents-core --bench agent_runtime
```

### Git Hooks

Este proyecto usa LeftHook para gestionar los Git hooks. Los hooks ejecutan automáticamente:

- Formato con `cargo fmt --check`
- Linting con `cargo clippy -- -D warnings`
- Pruebas con `cargo test --all-features --workspace --exclude autoagents-burn`

### Contribuir

Damos la bienvenida a las contribuciones. Consulta [Guía de contribución](CONTRIBUTING.md) y [Código de conducta](CODE_OF_CONDUCT.md).

---

## Documentación

- **[Documentación API](https://liquidos-ai.github.io/AutoAgents)**: documentación completa del framework
- **[Ejemplos](examples/)**: implementaciones prácticas

---

## Comunidad

- **GitHub Issues**: reportes de bugs y solicitudes de funciones
- **Discussions**: preguntas y respuestas de la comunidad
- **Discord**: únete a nuestra comunidad en https://discord.gg/zfAF9MkEtK

---

## Rendimiento

AutoAgents está diseñado para alto rendimiento:

- **Eficiencia de memoria**: uso de memoria optimizado con backends configurables
- **Concurrencia**: soporte completo de async/await con tokio
- **Escalabilidad**: escalado horizontal con coordinación multiagente
- **Seguridad de tipos**: garantías en tiempo de compilación con el sistema de tipos de Rust

---

## Licencia

AutoAgents tiene doble licencia:

- **MIT License** ([MIT_LICENSE](MIT_LICENSE))
- **Apache License 2.0** ([APACHE_LICENSE](APACHE_LICENSE))

Puedes elegir cualquiera según tu caso de uso.

---

## Agradecimientos

Construido por el equipo de [Liquidos AI](https://liquidos.ai) y una gran comunidad de investigadores e ingenieros.

<a href="https://github.com/liquidos-ai/AutoAgents/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=liquidos-ai/AutoAgents" />
</a>

Agradecimientos especiales:

- La comunidad Rust por su excelente ecosistema
- Los proveedores de LLM por habilitar APIs de modelos de alta calidad
- Todos los contribuyentes que ayudan a mejorar AutoAgents

---

## Historial de estrellas

[![Star History Chart](https://api.star-history.com/svg?repos=liquidos-ai/AutoAgents&type=Date)](https://www.star-history.com/#liquidos-ai/AutoAgents&Date)
