//! Guardrails framework for `autoagents-llm` providers.
//!
//! This crate provides a policy-driven guardrails engine that can wrap any
//! `Arc<dyn LLMProvider>` directly or be inserted as an `LLMLayer` in the
//! `PipelineBuilder` chain.

mod engine;
mod guard;
mod layer;
mod policy;
mod provider;
pub mod sanitizers;
mod stream;

pub mod guards;

pub use engine::{Guardrails, GuardrailsBuilder};
pub use guard::{
    ChatGuardInput, ChatGuardOutput, CompletionGuardInput, CompletionGuardOutput, GuardContext,
    GuardDecision, GuardError, GuardOperation, GuardViolation, GuardedInput, GuardedOutput,
    InputGuard, OutputGuard, WebSearchGuardInput,
};
pub use layer::GuardrailsLayer;
pub use policy::{EnforcementPolicy, GuardCategory, GuardSeverity};
pub use sanitizers::{
    InputSanitizer, OutputSanitizer, SharedInputSanitizer, SharedOutputSanitizer,
    default_input_sanitizer, default_output_sanitizer, noop_input_payload, noop_input_sanitizer,
    noop_output_payload, noop_output_sanitizer, redact_input_payload, redact_output_payload,
    redact_output_text_only_payload, redact_output_text_only_sanitizer,
};
