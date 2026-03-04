use std::fmt;

use autoagents_llm::error::{GuardrailPhase as LlmGuardrailPhase, LLMError};

use crate::guard::GuardViolation;

/// Enforcement policy applied when a guard reports a violation.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Default)]
pub enum EnforcementPolicy {
    /// Immediately fail the request.
    #[default]
    Block,
    /// Sanitize payload and continue.
    Sanitize,
    /// Log violations and continue without forcing changes.
    Audit,
}

/// Broad category for a guardrail rule.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum GuardCategory {
    PromptInjection,
    Toxicity,
    Custom(String),
}

impl fmt::Display for GuardCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GuardCategory::PromptInjection => f.write_str("prompt_injection"),
            GuardCategory::Toxicity => f.write_str("toxicity"),
            GuardCategory::Custom(value) => f.write_str(value),
        }
    }
}

/// Severity associated with a violation.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum GuardSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for GuardSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GuardSeverity::Low => f.write_str("low"),
            GuardSeverity::Medium => f.write_str("medium"),
            GuardSeverity::High => f.write_str("high"),
            GuardSeverity::Critical => f.write_str("critical"),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum GuardPhase {
    Input,
    Output,
}

impl fmt::Display for GuardPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GuardPhase::Input => f.write_str("input"),
            GuardPhase::Output => f.write_str("output"),
        }
    }
}

pub(crate) fn violation_to_llm_error(
    phase: GuardPhase,
    guard_name: &str,
    violation: &GuardViolation,
) -> LLMError {
    LLMError::GuardrailBlocked {
        phase: match phase {
            GuardPhase::Input => LlmGuardrailPhase::Input,
            GuardPhase::Output => LlmGuardrailPhase::Output,
        },
        guard: guard_name.to_string().into(),
        rule_id: violation.rule_id.clone().into(),
        category: violation.category.to_string().into(),
        severity: violation.severity.to_string().into(),
        message: violation.message.clone().into(),
    }
}

pub(crate) fn guard_failure_to_llm_error(guard_name: &str, message: &str) -> LLMError {
    LLMError::GuardrailExecutionFailed {
        guard: guard_name.to_string(),
        message: message.to_string(),
    }
}
