# LLM Guardrails

`autoagents-guardrails` provides policy-driven request/response safety controls for any `Arc<dyn LLMProvider>`.

It supports:

- Input guards (prompt checks)
- Output guards (response checks)
- Configurable enforcement policy (`Block`, `Sanitize`, `Audit`)
- Direct wrapper mode and `LLMLayer` mode for `PipelineBuilder`

## Basic Usage

```rust,ignore
use std::sync::Arc;

use autoagents::guardrails::{
    EnforcementPolicy, Guardrails,
    guards::{PromptInjectionGuard, ToxicityGuard},
};
use autoagents::llm::LLMProvider;
use autoagents::llm::pipeline::PipelineBuilder;

let base: Arc<dyn LLMProvider> = build_provider();

let guardrails = Guardrails::builder()
    .input_guard(PromptInjectionGuard::default())
    .output_guard(ToxicityGuard::default())
    .enforcement_policy(EnforcementPolicy::Block)
    .build();

let llm: Arc<dyn LLMProvider> = PipelineBuilder::new(base)
    .add_layer(guardrails.layer())
    .build();
```

## Policy Semantics

- `Block`: fail fast on violations.
- `Sanitize`: rewrite payload to a redacted safe form and continue.
- `Audit`: log violation and continue.

Global policy is set with `enforcement_policy(...)`. You can override policy for
specific guards with:

- `input_guard_with_policy(...)`
- `output_guard_with_policy(...)`

## Custom Sanitizers

You can override sanitize behavior with custom functions:

```rust,ignore
use autoagents::guardrails::{EnforcementPolicy, Guardrails, GuardedOutput};

let guardrails = Guardrails::builder()
    .enforcement_policy(EnforcementPolicy::Sanitize)
    .output_sanitizer(|output, _violation, _ctx| {
        if let GuardedOutput::Completion(c) = output {
            c.text = "[custom sanitized]".to_string();
        }
    })
    .build();
```

Built-in sanitizer helpers are available in `autoagents::guardrails::sanitizers`:

- `default_input_sanitizer()`
- `default_output_sanitizer()`
- `redact_input_payload(...)`
- `redact_output_payload(...)`
- `redact_output_text_only_payload(...)`
- `noop_input_sanitizer()`
- `noop_output_sanitizer()`

## Built-in Input Redaction Guard

`RegexPiiRedactionGuard` redacts common PII in input payloads (email, phone,
SSN, and card-like numbers) before provider calls:

```rust,ignore
use autoagents::guardrails::{Guardrails, guards::RegexPiiRedactionGuard};

let guardrails = Guardrails::builder()
    .input_guard(RegexPiiRedactionGuard::default())
    .build();
```

## Streaming Behavior

Streaming calls use pre-flight and post-aggregate checks:

- Input guards run before stream starts.
- Output guards run once after the stream fully completes.
- For blocked outputs, a final stream error is emitted.

## Extending With New Guards

Implement `InputGuard` or `OutputGuard`:

```rust,ignore
use async_trait::async_trait;
use autoagents::guardrails::{
    GuardContext, GuardDecision, GuardError, GuardedInput, InputGuard,
};

struct MyInputGuard;

#[async_trait]
impl InputGuard for MyInputGuard {
    fn name(&self) -> &'static str { "my-input-guard" }

    async fn inspect(
        &self,
        input: &mut GuardedInput,
        _ctx: &GuardContext,
    ) -> Result<GuardDecision, GuardError> {
        let _ = input;
        Ok(GuardDecision::Pass)
    }
}
```

## Cloud Moderation / External Providers

For cloud moderation services, implement `InputGuard` and/or `OutputGuard`
that call your provider SDK or HTTP API and return `GuardDecision`.
