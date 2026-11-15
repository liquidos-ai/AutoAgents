//! Cost budget guard backed by the `llm-budget` crate (feature `cost-budget`).
//!
//! [`BudgetGuard`] turns an [`llm_budget::AgentBudget`] hard USD cap into a
//! guardrail: before a call is sent, the guard estimates its cost from the
//! prompt size and [`reserve_output_tokens`](BudgetGuard::reserve_output_tokens),
//! and reserves that amount against the budget — a request that does not fit
//! is rejected before the provider ever sees it. When the response comes back,
//! the reservation is settled at the real usage the provider reported, priced
//! from `llm-budget`'s built-in table of 232 models.
//!
//! When the settled spend crosses the cap, the guard kills the budget: the
//! answer that was just paid for still reaches the caller, and every later
//! request is rejected on the input side with a `budget-exhausted` violation.
//!
//! ```no_run
//! use std::sync::Arc;
//! use autoagents_guardrails::{BudgetGuard, Guardrails};
//! use llm_budget::AgentBudget;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let budget = Arc::new(AgentBudget::builder("agent", 0.25).build()?);
//! let guard = BudgetGuard::new(budget, "claude-haiku-4-5")?;
//!
//! let guardrails = Guardrails::builder().input_guard(guard).build();
//! # let _ = guardrails;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use autoagents_llm::chat::Usage;
use llm_budget::{AgentBudget, PriceTable, Reservation};

use crate::{
    guard::{
        GuardContext, GuardDecision, GuardError, GuardViolation, GuardedInput, GuardedOutput,
        InputGuard, OutputGuard,
    },
    policy::{GuardCategory, GuardSeverity},
};

/// Rough chars-per-token estimate used when pricing a prompt before it is sent.
const DEFAULT_CHARS_PER_TOKEN: usize = 4;

/// An input+output guard that enforces a hard USD spend cap via `llm-budget`.
///
/// Register the same instance as both an input and an output guard (the
/// builder methods take `Arc` clones, so `input_guard(guard.clone())` and
/// `output_guard(guard)` share the budget):
///
/// ```no_run
/// # use std::sync::Arc;
/// # use autoagents_guardrails::{BudgetGuard, Guardrails};
/// # use llm_budget::AgentBudget;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let guard = BudgetGuard::new(Arc::new(AgentBudget::builder("a", 1.0).build()?), "gpt-4o")?;
/// let guardrails = Guardrails::builder()
///     .input_guard(guard.clone())
///     .output_guard(guard)
///     .build();
/// # let _ = guardrails;
/// # Ok(())
/// # }
/// ```
pub struct BudgetGuard {
    budget: Arc<AgentBudget>,
    model: String,
    /// Output tokens reserved on top of the prompt estimate per call.
    reserve_output_tokens: u64,
    chars_per_token: usize,
    /// In-flight reservations keyed by [`GuardContext::request_id`], settled
    /// when the matching output evaluation runs. Shared across clones so a
    /// guard can be split between the input and output sides of one engine.
    open: Arc<Mutex<HashMap<u64, Reservation>>>,
}

impl std::fmt::Debug for BudgetGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BudgetGuard")
            .field("model", &self.model)
            .field("reserve_output_tokens", &self.reserve_output_tokens)
            .field("limit_usd", &self.budget.limit_usd())
            .field("spent_usd", &self.budget.spent_usd())
            .finish_non_exhaustive()
    }
}

impl Clone for BudgetGuard {
    fn clone(&self) -> Self {
        Self {
            budget: Arc::clone(&self.budget),
            model: self.model.clone(),
            reserve_output_tokens: self.reserve_output_tokens,
            chars_per_token: self.chars_per_token,
            open: Arc::clone(&self.open),
        }
    }
}

impl BudgetGuard {
    /// A guard pricing every call as `model` against `budget`'s USD cap.
    ///
    /// # Errors
    /// [`GuardError`] when `model` is not in the built-in price table.
    pub fn new(budget: Arc<AgentBudget>, model: impl Into<String>) -> Result<Self, GuardError> {
        let model = model.into();
        if PriceTable::builtin().get(&model).is_none() {
            return Err(GuardError::new(format!(
                "budget guard: unknown model for pricing: {model}"
            )));
        }
        Ok(Self {
            budget,
            model,
            reserve_output_tokens: 0,
            chars_per_token: DEFAULT_CHARS_PER_TOKEN,
            open: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Reserve room for up to `n` completion tokens on top of the prompt
    /// estimate. Without this, a call whose estimate fits but whose answer
    /// does not is only caught *after* the money is spent.
    #[must_use]
    pub fn reserve_output_tokens(mut self, n: u64) -> Self {
        self.reserve_output_tokens = n;
        self
    }

    /// Chars-per-token estimate for prompt pricing (default 4).
    #[must_use]
    pub fn chars_per_token(mut self, n: usize) -> Self {
        self.chars_per_token = n.max(1);
        self
    }

    /// The shared budget, for the audit log, kill switch and spend figures.
    pub fn budget(&self) -> &Arc<AgentBudget> {
        &self.budget
    }

    fn exhausted_violation(&self) -> GuardViolation {
        GuardViolation::new(
            "budget-exhausted",
            GuardCategory::Custom("cost".to_string()),
            GuardSeverity::High,
            format!(
                "USD budget exhausted for model {}: ${:.4} of ${:.4} spent",
                self.model,
                self.budget.spent_usd(),
                self.budget.limit_usd()
            ),
        )
    }

    fn estimate_cost(&self, input: &GuardedInput) -> f64 {
        let chars: usize = match input {
            GuardedInput::Chat(chat) => chat.messages.iter().map(|m| m.content.len()).sum(),
            GuardedInput::Completion(completion) => completion.request.prompt.len(),
            GuardedInput::WebSearch(web) => web.input.len(),
        };
        let input_tokens = (chars / self.chars_per_token) as u64;
        PriceTable::builtin()
            .get(&self.model)
            .map_or(0.0, |p| p.cost(input_tokens, self.reserve_output_tokens))
    }

    fn settle(&self, request_id: u64, usage: Option<&Usage>) {
        let reservation = self
            .open
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&request_id);
        let Some(usage) = usage else {
            // No usage reported (streams, some backends): release the hold;
            // the estimate was never spent.
            drop(reservation);
            return;
        };
        let actual = PriceTable::builtin().get(&self.model).map_or(0.0, |p| {
            p.cost(
                u64::from(usage.prompt_tokens),
                u64::from(usage.completion_tokens),
            )
        });
        match reservation {
            Some(r) => {
                let outcome = r.settle(actual);
                if outcome.over_cap {
                    self.budget.kill("settled spend crossed the cap");
                }
            }
            None => {
                let outcome = self.budget.charge_usd_force(
                    actual,
                    llm_budget::ChargeContext::new().model(self.model.clone()),
                );
                if outcome.over_cap {
                    self.budget.kill("settled spend crossed the cap");
                }
            }
        }
    }
}

#[async_trait]
impl InputGuard for BudgetGuard {
    fn name(&self) -> &'static str {
        "cost-budget"
    }

    async fn inspect(
        &self,
        input: &mut GuardedInput,
        context: &GuardContext,
    ) -> Result<GuardDecision, GuardError> {
        if self.budget.is_killed() || self.budget.is_exhausted() {
            return Ok(GuardDecision::reject(self.exhausted_violation()));
        }
        let estimate = self.estimate_cost(input);
        match self.budget.reserve(estimate) {
            Ok(reservation) => {
                self.open
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .insert(context.request_id, reservation);
                Ok(GuardDecision::pass())
            }
            Err(_) => Ok(GuardDecision::reject(self.exhausted_violation())),
        }
    }
}

#[async_trait]
impl OutputGuard for BudgetGuard {
    fn name(&self) -> &'static str {
        "cost-budget"
    }

    async fn inspect(
        &self,
        output: &mut GuardedOutput,
        context: &GuardContext,
    ) -> Result<GuardDecision, GuardError> {
        match output {
            GuardedOutput::Chat(chat) => self.settle(context.request_id, chat.usage.as_ref()),
            GuardedOutput::Completion(_) => self.settle(context.request_id, None),
        }
        Ok(GuardDecision::pass())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoagents_llm::chat::{ChatMessage, Usage};

    fn budget(limit: f64) -> Arc<AgentBudget> {
        Arc::new(AgentBudget::builder("test", limit).build().unwrap())
    }

    fn chat_input(text: &str) -> GuardedInput {
        GuardedInput::Chat(crate::guard::ChatGuardInput {
            messages: vec![ChatMessage::user().content(text).build()],
            tools: None,
            json_schema: None,
        })
    }

    fn chat_output(prompt: u32, completion: u32) -> GuardedOutput {
        GuardedOutput::Chat(crate::guard::ChatGuardOutput {
            text: Some("ok".to_string()),
            tool_calls: None,
            thinking: None,
            usage: Some(Usage {
                prompt_tokens: prompt,
                completion_tokens: completion,
                total_tokens: prompt + completion,
                completion_tokens_details: None,
                prompt_tokens_details: None,
            }),
        })
    }

    #[tokio::test]
    async fn passes_and_books_real_usage() {
        let b = budget(1.0);
        let guard = BudgetGuard::new(b, "claude-haiku-4-5").unwrap();
        let mut input = chat_input("hello world");
        let ctx = GuardContext::new(crate::guard::GuardOperation::Chat);

        let decision = InputGuard::inspect(&guard, &mut input, &ctx).await.unwrap();
        assert!(matches!(decision, GuardDecision::Pass));
        assert!(guard.budget.reserved_usd() > 0.0);

        let mut output = chat_output(1_000, 500);
        let decision = OutputGuard::inspect(&guard, &mut output, &ctx)
            .await
            .unwrap();
        assert!(matches!(decision, GuardDecision::Pass));
        // 1k input at $1/M + 500 output at $5/M = $0.001 + $0.0025
        assert!((guard.budget.spent_usd() - 0.0035).abs() < 1e-9);
        assert_eq!(guard.budget.reserved_usd(), 0.0);
    }

    #[tokio::test]
    async fn rejects_when_cap_cannot_cover_estimate() {
        // $0.000001 cap; any haiku estimate exceeds it.
        let b = budget(0.000001);
        let guard = BudgetGuard::new(b, "claude-haiku-4-5")
            .unwrap()
            .reserve_output_tokens(1_000);
        let mut input = chat_input("a much longer prompt that costs more than the cap allows");
        let ctx = GuardContext::new(crate::guard::GuardOperation::Chat);
        let decision = InputGuard::inspect(&guard, &mut input, &ctx).await.unwrap();
        assert!(matches!(decision, GuardDecision::Reject(_)));
    }

    #[tokio::test]
    async fn kills_budget_when_settled_spend_crosses_cap() {
        let b = budget(0.001);
        let guard = BudgetGuard::new(b, "claude-haiku-4-5").unwrap();
        // No reservation (request id never went through the input side).
        let ctx = GuardContext::new(crate::guard::GuardOperation::Chat);
        let mut output = chat_output(10_000, 1_000); // $0.01 + $0.005 > $0.001
        OutputGuard::inspect(&guard, &mut output, &ctx)
            .await
            .unwrap();
        assert!(guard.budget.is_killed());

        let mut input = chat_input("hi");
        let ctx2 = GuardContext::new(crate::guard::GuardOperation::Chat);
        let decision = InputGuard::inspect(&guard, &mut input, &ctx2)
            .await
            .unwrap();
        assert!(matches!(decision, GuardDecision::Reject(_)));
    }

    #[tokio::test]
    async fn releases_reservation_when_no_usage_reported() {
        let b = budget(1.0);
        let guard = BudgetGuard::new(b, "claude-haiku-4-5").unwrap();
        let mut input = chat_input("hello");
        let ctx = GuardContext::new(crate::guard::GuardOperation::ChatStream);
        InputGuard::inspect(&guard, &mut input, &ctx).await.unwrap();
        assert!(guard.budget.reserved_usd() > 0.0);

        let mut output = GuardedOutput::Chat(crate::guard::ChatGuardOutput {
            text: Some("partial".to_string()),
            tool_calls: None,
            thinking: None,
            usage: None,
        });
        OutputGuard::inspect(&guard, &mut output, &ctx)
            .await
            .unwrap();
        assert_eq!(guard.budget.reserved_usd(), 0.0);
        assert_eq!(guard.budget.spent_usd(), 0.0);
    }

    #[test]
    fn unknown_model_is_rejected() {
        let b = budget(1.0);
        assert!(BudgetGuard::new(b, "not-a-model").is_err());
    }
}
