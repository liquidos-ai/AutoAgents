<div align="center">
  <img src="assets/logo.png" alt="AutoAgents Logo" width="200" height="200">

# AutoAgents

**Rust 기반의 프로덕션급 멀티 에이전트 프레임워크**

[![Crates.io](https://img.shields.io/crates/v/autoagents.svg)](https://crates.io/crates/autoagents)
[![Documentation](https://docs.rs/autoagents/badge.svg)](https://liquidos-ai.github.io/AutoAgents)
[![License](https://img.shields.io/crates/l/autoagents.svg)](https://github.com/liquidos-ai/AutoAgents#license)
[![Build Status](https://github.com/liquidos-ai/AutoAgents/workflows/coverage/badge.svg)](https://github.com/liquidos-ai/AutoAgents/actions)
[![codecov](https://codecov.io/gh/liquidos-ai/AutoAgents/graph/badge.svg)](https://codecov.io/gh/liquidos-ai/AutoAgents)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/liquidos-ai/AutoAgents)

[English](README.md) | [中文](README.zh-CN.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [한국어](README.ko.md) | [Português (Brasil)](README.pt-BR.md)
<br />
<sub>이 번역은 커뮤니티에서 유지되며 영어 버전보다 늦을 수 있습니다. 차이가 있을 경우 영어 버전이 기준입니다.</sub>

[문서](https://liquidos-ai.github.io/AutoAgents/) | [예제](examples/) | [기여하기](CONTRIBUTING.md)

<br />
<strong>이 프로젝트가 마음에 드시나요?</strong> <a href="https://github.com/liquidos-ai/AutoAgents">GitHub에서 스타를 눌러주세요</a>
</div>

---

## 개요

AutoAgents는 Rust에서 지능형 시스템을 구축하기 위한 모듈식 멀티 에이전트 프레임워크입니다. 타입 안전한 에이전트 모델, 구조화된 도구 호출, 구성 가능한 메모리, 플러그형 LLM 백엔드를 결합합니다. 아키텍처는 서버와 엣지 환경 모두에서 성능, 안전성, 합성 가능성을 목표로 설계되었습니다.

---

## 핵심 기능

- **에이전트 실행**: ReAct 및 기본 실행기, 스트리밍 응답, 구조화된 출력
- **도구**: 도구 및 출력용 파생 매크로, 도구 실행을 위한 샌드박스 WASM 런타임
- **메모리**: 확장 가능한 백엔드와 함께하는 슬라이딩 윈도 메모리
- **LLM 제공자**: 통합 인터페이스 뒤의 클라우드 및 로컬 백엔드
- **멀티 에이전트 오케스트레이션**: 타입 기반 pub/sub 통신과 환경 관리
- **음성 처리**: 로컬 TTS 및 STT 지원
- **관측 가능성**: OpenTelemetry 트레이싱 및 메트릭, 플러그형 익스포터

---

## 지원 LLM 제공자

### 클라우드 제공자

| 제공자          | 상태 |
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

### 로컬 제공자

| 제공자       | 상태 |
| ------------ | ---- |
| **Ollama**   | ✅   |
| **Mistral-rs** | ✅ |
| **Llama-Cpp** | ✅ |

### 실험적 제공자

자세히 보기: https://github.com/liquidos-ai/AutoAgents-Experimental-Backends

| 제공자 | 상태           |
| ------ | -------------- |
| **Burn** | ⚠️ 실험적    |
| **Onnx** | ⚠️ 실험적    |

제공자 지원은 커뮤니티 요구에 따라 계속 확장됩니다.

---

## 벤치마크

![Benchmark](./assets/Benchmark.png)

자세한 정보는 [GitHub](https://github.com/liquidos-ai/autoagents-bench)에서 확인하세요.

---

## 설치

### 사전 요구 사항

- **Rust** (최신 안정 버전 권장)
- **Cargo** 패키지 관리자
- **LeftHook** Git hooks 관리용

### Prerequisite

```bash
sudo apt update
sudo apt install build-essential libasound2-dev alsa-utils pkg-config libssl-dev -y
```

### LeftHook 설치

macOS (Homebrew):

```bash
brew install lefthook
```

Linux/Windows (npm):

```bash
npm install -g lefthook
```

### 클론 및 빌드

```bash
git clone https://github.com/liquidos-ai/AutoAgents.git
cd AutoAgents
lefthook install
cargo build --workspace --all-features
```

### 테스트 실행

```bash
cargo test --workspace --features default --exclude autoagents-burn --exclude autoagents-mistral-rs --exclude wasm_agent
```

---

## 빠른 시작

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

AutoAgents CLI는 YAML 구성으로 에이전트 워크플로를 실행하고 HTTP로 제공하도록 돕습니다. 자세한 내용은 https://github.com/liquidos-ai/AutoAgents-CLI 를 참고하세요.

---

## 예제

예제를 통해 빠르게 시작하세요:

### [기본](examples/basic/)

도구가 있는 간단한 에이전트, 매우 기본적인 에이전트, 엣지 에이전트, 체이닝, 액터 기반 모델, 스트리밍, Agent Hooks 추가 등 다양한 예제를 보여줍니다.

### [MCP 통합](examples/mcp/)

AutoAgents를 Model Context Protocol (MCP)과 통합하는 방법을 보여줍니다.

### [로컬 모델](examples/mistral_rs)

로컬 모델을 위해 AutoAgents와 Mistral-rs를 통합하는 방법을 보여줍니다.

### [디자인 패턴](examples/design_patterns/)

체이닝, 계획, 라우팅, 병렬, 리플렉션 등의 패턴을 보여줍니다.

### [제공자](examples/providers/)

AutoAgents에서 다양한 LLM 제공자를 사용하는 예제를 포함합니다.

### [WASM 도구 실행](examples/wasm_runner/)

WASM 런타임에서 도구를 실행할 수 있는 간단한 에이전트입니다.

### [코딩 에이전트](examples/coding_agent/)

파일 조작 기능을 갖춘 ReAct 기반의 고급 코딩 에이전트입니다.

### [음성](examples/speech/)

실시간 TTS 및 STT를 사용하는 AutoAgents 음성 예제를 실행합니다.

### [Android 로컬 에이전트](https://github.com/liquidos-ai/AutoAgents-Android-Example)

autoagents-llamacpp 백엔드를 사용하여 Android에서 로컬 모델로 AutoAgents를 실행하는 예제 앱입니다.

---

## 구성 요소

AutoAgents는 모듈형 아키텍처로 구성됩니다:

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

### 핵심 구성 요소

- **에이전트**: 지능의 기본 단위
- **환경**: 에이전트 라이프사이클과 통신 관리
- **메모리**: 구성 가능한 메모리 시스템
- **도구**: 외부 기능 통합
- **실행기**: 다양한 추론 패턴 (ReAct, Chain-of-Thought)

---

## 개발

### 테스트 실행

```bash
cargo test --workspace --features default --exclude autoagents-burn --exclude autoagents-mistral-rs --exclude wasm_agent

# Coverage (requires cargo-tarpaulin)
cargo install cargo-tarpaulin
cargo tarpaulin --all-features --out html
```

### 벤치마크 실행

```bash
cargo bench -p autoagents-core --bench agent_runtime
```

### Git Hooks

이 프로젝트는 LeftHook을 사용해 Git hooks를 관리합니다. hooks는 자동으로 다음을 수행합니다:

- `cargo fmt --check` 로 코드 포맷
- `cargo clippy -- -D warnings` 로 린트
- `cargo test --all-features --workspace --exclude autoagents-burn` 로 테스트 실행

### 기여하기

기여를 환영합니다. 자세한 내용은 [기여 가이드](CONTRIBUTING.md)와 [행동 강령](CODE_OF_CONDUCT.md)을 참고하세요.

---

## 문서

- **[API 문서](https://liquidos-ai.github.io/AutoAgents)**: 전체 프레임워크 문서
- **[예제](examples/)**: 실용적인 구현 예시

---

## 커뮤니티

- **GitHub Issues**: 버그 보고 및 기능 요청
- **Discussions**: 커뮤니티 Q&A 및 아이디어
- **Discord**: Discord 커뮤니티 참여 https://discord.gg/zfAF9MkEtK

---

## 성능

AutoAgents는 고성능을 목표로 설계되었습니다:

- **메모리 효율**: 구성 가능한 백엔드로 메모리 사용 최적화
- **동시성**: tokio 기반의 async/await 완전 지원
- **확장성**: 멀티 에이전트 조정으로 수평 확장
- **타입 안전성**: Rust 타입 시스템의 컴파일 타임 보장

---

## 라이선스

AutoAgents는 이중 라이선스를 사용합니다:

- **MIT License** ([MIT_LICENSE](MIT_LICENSE))
- **Apache License 2.0** ([APACHE_LICENSE](APACHE_LICENSE))

사용 사례에 따라 원하는 라이선스를 선택할 수 있습니다.

---

## 감사의 말

[Liquidos AI](https://liquidos.ai) 팀과 훌륭한 연구자 및 엔지니어 커뮤니티가 함께 만들었습니다.

<a href="https://github.com/liquidos-ai/AutoAgents/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=liquidos-ai/AutoAgents" />
</a>

특별 감사:

- 뛰어난 생태계를 제공하는 Rust 커뮤니티
- 고품질 모델 API를 제공하는 LLM 제공자
- AutoAgents 개선에 기여하는 모든 분들

---

## 스타 기록

[![Star History Chart](https://api.star-history.com/svg?repos=liquidos-ai/AutoAgents&type=Date)](https://www.star-history.com/#liquidos-ai/AutoAgents&Date)
