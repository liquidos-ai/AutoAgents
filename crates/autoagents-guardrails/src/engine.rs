use std::sync::Arc;

use autoagents_llm::{LLMProvider, error::LLMError};

use crate::{
    guard::{GuardContext, GuardDecision, GuardedInput, GuardedOutput, InputGuard, OutputGuard},
    layer::GuardrailsLayer,
    policy::{EnforcementPolicy, GuardPhase, guard_failure_to_llm_error, violation_to_llm_error},
    provider::GuardedProvider,
    sanitizers::{
        SharedInputSanitizer, SharedOutputSanitizer, default_input_sanitizer,
        default_output_sanitizer,
    },
};

struct InputGuardEntry {
    guard: Arc<dyn InputGuard>,
    policy_override: Option<EnforcementPolicy>,
}

struct OutputGuardEntry {
    guard: Arc<dyn OutputGuard>,
    policy_override: Option<EnforcementPolicy>,
}

/// User-facing guardrails handle.
#[derive(Clone)]
pub struct Guardrails {
    pub(crate) engine: Arc<GuardrailsEngine>,
}

impl Guardrails {
    /// Create guardrails directly from input/output guard lists with default
    /// [`EnforcementPolicy::Block`] behavior.
    pub fn new(
        input_guards: Vec<Arc<dyn InputGuard>>,
        output_guards: Vec<Arc<dyn OutputGuard>>,
    ) -> Self {
        let input_guards = input_guards
            .into_iter()
            .map(|guard| InputGuardEntry {
                guard,
                policy_override: None,
            })
            .collect();
        let output_guards = output_guards
            .into_iter()
            .map(|guard| OutputGuardEntry {
                guard,
                policy_override: None,
            })
            .collect();

        Self {
            engine: Arc::new(GuardrailsEngine {
                input_guards,
                output_guards,
                policy: EnforcementPolicy::Block,
                input_sanitizer: default_input_sanitizer(),
                output_sanitizer: default_output_sanitizer(),
            }),
        }
    }

    /// Start building a guardrails configuration.
    pub fn builder() -> GuardrailsBuilder {
        GuardrailsBuilder::default()
    }

    /// Create an `LLMLayer` for use with `PipelineBuilder`.
    pub fn layer(&self) -> GuardrailsLayer {
        GuardrailsLayer::new(self.engine.clone())
    }

    /// Wrap a provider directly without a pipeline.
    pub fn wrap(&self, inner: Arc<dyn LLMProvider>) -> Arc<dyn LLMProvider> {
        Arc::new(GuardedProvider::new(inner, self.engine.clone()))
    }
}

/// Builder for [`Guardrails`].
pub struct GuardrailsBuilder {
    input_guards: Vec<InputGuardEntry>,
    output_guards: Vec<OutputGuardEntry>,
    policy: EnforcementPolicy,
    input_sanitizer: SharedInputSanitizer,
    output_sanitizer: SharedOutputSanitizer,
}

impl Default for GuardrailsBuilder {
    fn default() -> Self {
        Self {
            input_guards: Vec::new(),
            output_guards: Vec::new(),
            policy: EnforcementPolicy::default(),
            input_sanitizer: default_input_sanitizer(),
            output_sanitizer: default_output_sanitizer(),
        }
    }
}

impl GuardrailsBuilder {
    pub fn input_guard<G: InputGuard>(mut self, guard: G) -> Self {
        self.input_guards.push(InputGuardEntry {
            guard: Arc::new(guard),
            policy_override: None,
        });
        self
    }

    pub fn output_guard<G: OutputGuard>(mut self, guard: G) -> Self {
        self.output_guards.push(OutputGuardEntry {
            guard: Arc::new(guard),
            policy_override: None,
        });
        self
    }

    pub fn input_guard_arc(mut self, guard: Arc<dyn InputGuard>) -> Self {
        self.input_guards.push(InputGuardEntry {
            guard,
            policy_override: None,
        });
        self
    }

    pub fn output_guard_arc(mut self, guard: Arc<dyn OutputGuard>) -> Self {
        self.output_guards.push(OutputGuardEntry {
            guard,
            policy_override: None,
        });
        self
    }

    /// Add an input guard with a per-guard policy override.
    pub fn input_guard_with_policy<G: InputGuard>(
        mut self,
        guard: G,
        policy: EnforcementPolicy,
    ) -> Self {
        self.input_guards.push(InputGuardEntry {
            guard: Arc::new(guard),
            policy_override: Some(policy),
        });
        self
    }

    /// Add an output guard with a per-guard policy override.
    pub fn output_guard_with_policy<G: OutputGuard>(
        mut self,
        guard: G,
        policy: EnforcementPolicy,
    ) -> Self {
        self.output_guards.push(OutputGuardEntry {
            guard: Arc::new(guard),
            policy_override: Some(policy),
        });
        self
    }

    /// Add an input guard Arc with a per-guard policy override.
    pub fn input_guard_arc_with_policy(
        mut self,
        guard: Arc<dyn InputGuard>,
        policy: EnforcementPolicy,
    ) -> Self {
        self.input_guards.push(InputGuardEntry {
            guard,
            policy_override: Some(policy),
        });
        self
    }

    /// Add an output guard Arc with a per-guard policy override.
    pub fn output_guard_arc_with_policy(
        mut self,
        guard: Arc<dyn OutputGuard>,
        policy: EnforcementPolicy,
    ) -> Self {
        self.output_guards.push(OutputGuardEntry {
            guard,
            policy_override: Some(policy),
        });
        self
    }

    pub fn enforcement_policy(mut self, policy: EnforcementPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Set custom input sanitizer logic used when policy is
    /// [`EnforcementPolicy::Sanitize`].
    pub fn input_sanitizer<F>(mut self, sanitizer: F) -> Self
    where
        F: Fn(&mut GuardedInput, &crate::guard::GuardViolation, &GuardContext)
            + Send
            + Sync
            + 'static,
    {
        self.input_sanitizer = Arc::new(sanitizer);
        self
    }

    /// Set custom output sanitizer logic used when policy is
    /// [`EnforcementPolicy::Sanitize`].
    pub fn output_sanitizer<F>(mut self, sanitizer: F) -> Self
    where
        F: Fn(&mut GuardedOutput, &crate::guard::GuardViolation, &GuardContext)
            + Send
            + Sync
            + 'static,
    {
        self.output_sanitizer = Arc::new(sanitizer);
        self
    }

    /// Set input sanitizer using a pre-built shared sanitizer handle.
    pub fn input_sanitizer_arc(mut self, sanitizer: SharedInputSanitizer) -> Self {
        self.input_sanitizer = sanitizer;
        self
    }

    /// Set output sanitizer using a pre-built shared sanitizer handle.
    pub fn output_sanitizer_arc(mut self, sanitizer: SharedOutputSanitizer) -> Self {
        self.output_sanitizer = sanitizer;
        self
    }

    pub fn build(self) -> Guardrails {
        Guardrails {
            engine: Arc::new(GuardrailsEngine {
                input_guards: self.input_guards,
                output_guards: self.output_guards,
                policy: self.policy,
                input_sanitizer: self.input_sanitizer,
                output_sanitizer: self.output_sanitizer,
            }),
        }
    }
}

pub(crate) struct GuardrailsEngine {
    input_guards: Vec<InputGuardEntry>,
    output_guards: Vec<OutputGuardEntry>,
    policy: EnforcementPolicy,
    input_sanitizer: SharedInputSanitizer,
    output_sanitizer: SharedOutputSanitizer,
}

impl GuardrailsEngine {
    pub(crate) fn has_input_guards(&self) -> bool {
        !self.input_guards.is_empty()
    }

    pub(crate) fn has_output_guards(&self) -> bool {
        !self.output_guards.is_empty()
    }

    pub(crate) async fn evaluate_input(
        &self,
        input: &mut GuardedInput,
        context: &GuardContext,
    ) -> Result<(), LLMError> {
        for entry in &self.input_guards {
            let decision = entry
                .guard
                .inspect(input, context)
                .await
                .map_err(|err| guard_failure_to_llm_error(entry.guard.name(), &err.message))?;
            self.apply_input_decision(
                decision,
                input,
                entry.guard.name(),
                entry.policy_override.unwrap_or(self.policy),
                context,
            )?;
        }
        Ok(())
    }

    pub(crate) async fn evaluate_output(
        &self,
        output: &mut GuardedOutput,
        context: &GuardContext,
    ) -> Result<(), LLMError> {
        for entry in &self.output_guards {
            let decision = entry
                .guard
                .inspect(output, context)
                .await
                .map_err(|err| guard_failure_to_llm_error(entry.guard.name(), &err.message))?;
            self.apply_output_decision(
                decision,
                output,
                entry.guard.name(),
                entry.policy_override.unwrap_or(self.policy),
                context,
            )?;
        }
        Ok(())
    }

    fn apply_input_decision(
        &self,
        decision: GuardDecision,
        input: &mut GuardedInput,
        guard_name: &str,
        policy: EnforcementPolicy,
        context: &GuardContext,
    ) -> Result<(), LLMError> {
        match decision {
            GuardDecision::Pass => Ok(()),
            GuardDecision::Modify { violation } => {
                if let Some(violation) = violation {
                    self.handle_input_violation(input, guard_name, policy, context, &violation)
                } else {
                    Ok(())
                }
            }
            GuardDecision::Reject(violation) => {
                self.handle_input_violation(input, guard_name, policy, context, &violation)
            }
        }
    }

    fn apply_output_decision(
        &self,
        decision: GuardDecision,
        output: &mut GuardedOutput,
        guard_name: &str,
        policy: EnforcementPolicy,
        context: &GuardContext,
    ) -> Result<(), LLMError> {
        match decision {
            GuardDecision::Pass => Ok(()),
            GuardDecision::Modify { violation } => {
                if let Some(violation) = violation {
                    self.handle_output_violation(output, guard_name, policy, context, &violation)
                } else {
                    Ok(())
                }
            }
            GuardDecision::Reject(violation) => {
                self.handle_output_violation(output, guard_name, policy, context, &violation)
            }
        }
    }

    fn handle_input_violation(
        &self,
        input: &mut GuardedInput,
        guard_name: &str,
        policy: EnforcementPolicy,
        context: &GuardContext,
        violation: &crate::guard::GuardViolation,
    ) -> Result<(), LLMError> {
        match policy {
            EnforcementPolicy::Block => Err(violation_to_llm_error(
                GuardPhase::Input,
                guard_name,
                violation,
            )),
            EnforcementPolicy::Sanitize => {
                (self.input_sanitizer)(input, violation, context);
                log::warn!(
                    "guardrail input violation sanitized: request_id={}, op={}, guard={}, rule={}, category={}, severity={}, message={}",
                    context.request_id,
                    context.operation,
                    guard_name,
                    violation.rule_id,
                    violation.category,
                    violation.severity,
                    violation.message,
                );
                Ok(())
            }
            EnforcementPolicy::Audit => {
                log::warn!(
                    "guardrail input violation audited: request_id={}, op={}, guard={}, rule={}, category={}, severity={}, message={}",
                    context.request_id,
                    context.operation,
                    guard_name,
                    violation.rule_id,
                    violation.category,
                    violation.severity,
                    violation.message,
                );
                Ok(())
            }
        }
    }

    fn handle_output_violation(
        &self,
        output: &mut GuardedOutput,
        guard_name: &str,
        policy: EnforcementPolicy,
        context: &GuardContext,
        violation: &crate::guard::GuardViolation,
    ) -> Result<(), LLMError> {
        match policy {
            EnforcementPolicy::Block => Err(violation_to_llm_error(
                GuardPhase::Output,
                guard_name,
                violation,
            )),
            EnforcementPolicy::Sanitize => {
                (self.output_sanitizer)(output, violation, context);
                log::warn!(
                    "guardrail output violation sanitized: request_id={}, op={}, guard={}, rule={}, category={}, severity={}, message={}",
                    context.request_id,
                    context.operation,
                    guard_name,
                    violation.rule_id,
                    violation.category,
                    violation.severity,
                    violation.message,
                );
                Ok(())
            }
            EnforcementPolicy::Audit => {
                log::warn!(
                    "guardrail output violation audited: request_id={}, op={}, guard={}, rule={}, category={}, severity={}, message={}",
                    context.request_id,
                    context.operation,
                    guard_name,
                    violation.rule_id,
                    violation.category,
                    violation.severity,
                    violation.message,
                );
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;

    use crate::{
        guard::{
            DEFAULT_REDACTED_TEXT, GuardContext, GuardDecision, GuardError, GuardOperation,
            GuardViolation, GuardedInput, GuardedOutput, InputGuard, OutputGuard,
        },
        policy::{EnforcementPolicy, GuardCategory, GuardSeverity},
    };

    use super::Guardrails;

    struct RejectInputGuard;

    #[async_trait]
    impl InputGuard for RejectInputGuard {
        fn name(&self) -> &'static str {
            "reject-input"
        }

        async fn inspect(
            &self,
            _input: &mut GuardedInput,
            _context: &GuardContext,
        ) -> Result<GuardDecision, GuardError> {
            Ok(GuardDecision::Reject(GuardViolation::new(
                "reject",
                GuardCategory::Custom("test".to_string()),
                GuardSeverity::High,
                "blocked",
            )))
        }
    }

    struct RejectOutputGuard;

    #[async_trait]
    impl OutputGuard for RejectOutputGuard {
        fn name(&self) -> &'static str {
            "reject-output"
        }

        async fn inspect(
            &self,
            _output: &mut GuardedOutput,
            _context: &GuardContext,
        ) -> Result<GuardDecision, GuardError> {
            Ok(GuardDecision::Reject(GuardViolation::new(
                "reject",
                GuardCategory::Custom("test".to_string()),
                GuardSeverity::High,
                "blocked",
            )))
        }
    }

    #[tokio::test]
    async fn block_policy_fails_input_violations() {
        let guardrails = Guardrails::builder()
            .input_guard(RejectInputGuard)
            .enforcement_policy(EnforcementPolicy::Block)
            .build();

        let mut input = GuardedInput::WebSearch(crate::guard::WebSearchGuardInput {
            input: "hello".to_string(),
        });
        let context = GuardContext::new(GuardOperation::ChatWithWebSearch);

        let result = guardrails.engine.evaluate_input(&mut input, &context).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn sanitize_policy_rewrites_output() {
        let guardrails = Guardrails::builder()
            .output_guard(RejectOutputGuard)
            .enforcement_policy(EnforcementPolicy::Sanitize)
            .build();

        let mut output = GuardedOutput::Completion(crate::guard::CompletionGuardOutput {
            text: "unsafe".to_string(),
        });
        let context = GuardContext::new(GuardOperation::Complete);

        guardrails
            .engine
            .evaluate_output(&mut output, &context)
            .await
            .unwrap();

        match output {
            GuardedOutput::Completion(value) => {
                assert_eq!(value.text, DEFAULT_REDACTED_TEXT);
            }
            _ => panic!("unexpected output variant"),
        }
    }

    #[tokio::test]
    async fn audit_policy_allows_violations() {
        let guardrails = Guardrails::builder()
            .input_guard(RejectInputGuard)
            .enforcement_policy(EnforcementPolicy::Audit)
            .build();

        let mut input = GuardedInput::WebSearch(crate::guard::WebSearchGuardInput {
            input: "hello".to_string(),
        });
        let context = GuardContext::new(GuardOperation::ChatWithWebSearch);

        guardrails
            .engine
            .evaluate_input(&mut input, &context)
            .await
            .unwrap();

        match input {
            GuardedInput::WebSearch(value) => assert_eq!(value.input, "hello"),
            _ => panic!("unexpected input variant"),
        }
    }

    #[tokio::test]
    async fn custom_sanitizers_are_applied() {
        let guardrails = Guardrails::builder()
            .input_guard(RejectInputGuard)
            .output_guard(RejectOutputGuard)
            .enforcement_policy(EnforcementPolicy::Sanitize)
            .input_sanitizer(|input, _violation, _context| {
                if let GuardedInput::WebSearch(web) = input {
                    web.input = "custom-input".to_string();
                }
            })
            .output_sanitizer(|output, _violation, _context| {
                if let GuardedOutput::Completion(completion) = output {
                    completion.text = "custom-output".to_string();
                }
            })
            .build();

        let mut input = GuardedInput::WebSearch(crate::guard::WebSearchGuardInput {
            input: "hello".to_string(),
        });
        let input_context = GuardContext::new(GuardOperation::ChatWithWebSearch);
        guardrails
            .engine
            .evaluate_input(&mut input, &input_context)
            .await
            .unwrap();

        match input {
            GuardedInput::WebSearch(web) => assert_eq!(web.input, "custom-input"),
            _ => panic!("unexpected input variant"),
        }

        let mut output = GuardedOutput::Completion(crate::guard::CompletionGuardOutput {
            text: "unsafe".to_string(),
        });
        let output_context = GuardContext::new(GuardOperation::Complete);
        guardrails
            .engine
            .evaluate_output(&mut output, &output_context)
            .await
            .unwrap();

        match output {
            GuardedOutput::Completion(completion) => assert_eq!(completion.text, "custom-output"),
            _ => panic!("unexpected output variant"),
        }
    }

    #[tokio::test]
    async fn per_guard_policy_override_can_block_when_global_is_audit() {
        let guardrails = Guardrails::builder()
            .enforcement_policy(EnforcementPolicy::Audit)
            .output_guard_with_policy(RejectOutputGuard, EnforcementPolicy::Block)
            .build();

        let mut output = GuardedOutput::Completion(crate::guard::CompletionGuardOutput {
            text: "unsafe".to_string(),
        });
        let context = GuardContext::new(GuardOperation::Complete);

        let result = guardrails
            .engine
            .evaluate_output(&mut output, &context)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn per_guard_policy_override_can_sanitize_when_global_is_block() {
        let guardrails = Guardrails::builder()
            .enforcement_policy(EnforcementPolicy::Block)
            .input_guard_with_policy(RejectInputGuard, EnforcementPolicy::Sanitize)
            .build();

        let mut input = GuardedInput::WebSearch(crate::guard::WebSearchGuardInput {
            input: "hello".to_string(),
        });
        let context = GuardContext::new(GuardOperation::ChatWithWebSearch);

        guardrails
            .engine
            .evaluate_input(&mut input, &context)
            .await
            .unwrap();

        match input {
            GuardedInput::WebSearch(web) => assert_eq!(web.input, DEFAULT_REDACTED_TEXT),
            _ => panic!("unexpected input variant"),
        }
    }
}
