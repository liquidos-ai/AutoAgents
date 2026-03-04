use std::fmt;

use autoagents_llm::error::LLMError;

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
    let msg = format!(
        "guardrail blocked {phase}: guard={guard_name}, rule={}, category={}, severity={}, message={}",
        violation.rule_id, violation.category, violation.severity, violation.message
    );

    match phase {
        GuardPhase::Input => LLMError::InvalidRequest(msg),
        GuardPhase::Output => LLMError::ProviderError(msg),
    }
}

pub(crate) fn guard_failure_to_llm_error(guard_name: &str, message: &str) -> LLMError {
    LLMError::ProviderError(format!(
        "guardrail execution failed: guard={guard_name}, error={message}"
    ))
}
