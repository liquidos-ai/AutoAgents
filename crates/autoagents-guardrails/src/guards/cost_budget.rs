//! Cost budget guard backed by the `llm-budget` crate (feature `cost-budget`).
//!
//! [`BudgetGuard`] turns an [`llm_budget::AgentBudget`] hard USD cap into a
//! guardrail: before a call is sent, the guard estimates its cost from the
//! prompt size and [`reserve_output_tokens`](BudgetGuard::reserve_output_tokens),
//! and reserves that amount against the budget — a request that does not fit
//! is rejected before the provider ever sees it. When the response comes back,
//! the reservation is settled at the real usage the provider reported, priced
//! from `llm-budget`'s built-in table of 232 models. Streams settle when they
//! finish, from the final usage chunk; when a backend reports no usage, the
//! call is charged from the prompt estimate plus the length of the answer.
//!
//! Every request ends in exactly one of these ways:
//!
//! - **Never sent** (another input guard rejected it) or **provider error**:
//!   the reservation is released; nothing was spent.
//! - **Answered**: charged at the reported usage, or estimated from the prompt
//!   and the answer. This also covers a reply another output guard blocks,
//!   and a stream dropped or failing part way: the prompt plus every chunk
//!   delivered before it stopped is charged, so cancelling a stream early
//!   does not free the budget for output that was already generated.
//! - **Cancelled while waiting on the provider** (a dropped future, for
//!   example a timeout): charged at the reserved estimate, since the
//!   provider may still bill for it.
//!
//! When the settled spend crosses the cap, the guard kills the budget: the
//! answer that was just paid for still reaches the caller, and every later
//! request is rejected on the input side with a `budget-exhausted` violation.
//!
//! # Sub-agents
//!
//! An agent that delegates to several sub-agents at once can give each one
//! its own guard with [`BudgetGuard::sub_agent`]. The sub-agent's slice
//! ([`Limit::Usd`] or [`Limit::Fraction`] of what is left) is reserved from
//! the parent budget before the sub-agent starts, so parallel sub-agents can
//! never be admitted for more than the parent has left. Every call a
//! sub-agent settles is booked on the parent at the same moment, and when the
//! last clone of the sub-agent's guard is dropped, the unused part of its
//! slice goes back to the parent. A sub-agent that hits its own cap is
//! stopped without killing the parent.
//!
//! ```no_run
//! use std::sync::Arc;
//! use autoagents_guardrails::BudgetGuard;
//! use llm_budget::{AgentBudget, Limit};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let orchestrator = BudgetGuard::new(Arc::new(AgentBudget::builder("orchestrator", 1.00).build()?), "gpt-4o")?;
//! let researcher = orchestrator.sub_agent("researcher", Limit::Usd(0.40))?;
//! let writer = orchestrator.sub_agent("writer", Limit::Fraction(0.5))?; // half of the $0.60 left
//! // Give each sub-agent's LLM its own guard: Guardrails::builder()
//! //     .input_guard(researcher.clone()).output_guard(researcher) ...
//! # let _ = (researcher, writer);
//! # Ok(())
//! # }
//! ```
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
//! let guardrails = Guardrails::builder()
//!     .input_guard(guard.clone())
//!     .output_guard(guard)
//!     .build();
//! # let _ = guardrails;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use autoagents_llm::chat::Usage;
use llm_budget::{AgentBudget, Limit, PriceTable, Reservation, ScopeGuard};

use crate::{
    guard::{
        GuardContext, GuardDecision, GuardError, GuardViolation, GuardedInput, GuardedOutput,
        InputGuard, OutputGuard, RequestOutcome, RequestStage,
    },
    policy::{GuardCategory, GuardSeverity},
};

/// Rough chars-per-token estimate used when pricing a prompt before it is sent.
const DEFAULT_CHARS_PER_TOKEN: usize = 4;

/// An input+output guard that enforces a hard USD spend cap via `llm-budget`.
///
/// Register the same instance as both an input and an output guard (clones
/// share the budget and the in-flight reservations, so
/// `input_guard(guard.clone())` and `output_guard(guard)` work together).
/// The input side reserves and is told how every request ended, so calls are
/// charged even when another output guard blocks the reply; the output side
/// settles a call as soon as its reply is available:
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
    /// when the matching output evaluation runs or when the request ends.
    /// Shared across clones so a guard can be split between the input and
    /// output sides of one engine.
    open: Arc<Mutex<HashMap<u64, OpenCall>>>,
    /// For a [`sub_agent`](BudgetGuard::sub_agent) guard: the slice reserved
    /// from the parent budget, refunded when the last clone is dropped.
    scope: Option<Arc<ScopeGuard>>,
}

/// A call that passed the input side and has not been settled yet.
struct OpenCall {
    reservation: Reservation,
    /// Prompt tokens estimated on the input side, used to price the call
    /// when the provider reports no usage.
    input_tokens: u64,
    /// USD reserved for the call; charged when it is cancelled while
    /// waiting on the provider.
    estimate_usd: f64,
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
            scope: self.scope.clone(),
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
            scope: None,
        })
    }

    /// A guard for a sub-agent, on a slice of this guard's budget: `limit`
    /// is reserved from it now, before the sub-agent makes any call, and the
    /// new guard caps the sub-agent at that slice. Same model and settings.
    ///
    /// Everything the sub-agent spends is booked on this budget at the same
    /// moment. When the last clone of the returned guard is dropped, the
    /// unused part of the slice goes back. Sub-agents can have sub-agents.
    ///
    /// # Errors
    /// [`GuardError`] when the slice does not fit in what this budget has
    /// left, when this budget was killed, or for an invalid `limit`.
    pub fn sub_agent(&self, name: impl Into<String>, limit: Limit) -> Result<Self, GuardError> {
        let scope = self
            .budget
            .child(name, limit)
            .map_err(|e| GuardError::new(format!("budget guard: sub-agent slice refused: {e}")))?;
        Ok(Self {
            budget: Arc::clone(scope.budget()),
            model: self.model.clone(),
            reserve_output_tokens: self.reserve_output_tokens,
            chars_per_token: self.chars_per_token,
            open: Arc::new(Mutex::new(HashMap::new())),
            scope: Some(Arc::new(scope)),
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

    fn estimate_tokens(&self, chars: usize) -> u64 {
        (chars / self.chars_per_token) as u64
    }

    fn price(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        PriceTable::builtin()
            .get(&self.model)
            .map_or(0.0, |p| p.cost(input_tokens, output_tokens))
    }

    fn input_tokens(&self, input: &GuardedInput) -> u64 {
        let chars: usize = match input {
            GuardedInput::Chat(chat) => chat.messages.iter().map(|m| m.content.len()).sum(),
            GuardedInput::Completion(completion) => completion.request.prompt.len(),
            GuardedInput::WebSearch(web) => web.input.len(),
        };
        self.estimate_tokens(chars)
    }

    fn take_open(&self, request_id: u64) -> Option<OpenCall> {
        self.open
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&request_id)
    }

    /// Price a call from the provider's reported usage, or, when there is
    /// none (plain text streams, some backends, a stream dropped before its
    /// usage chunk), from the prompt estimate plus `output_chars` of answer.
    fn actual_cost(&self, input_tokens: u64, usage: Option<&Usage>, output_chars: usize) -> f64 {
        match usage {
            Some(usage) => self.price(
                u64::from(usage.prompt_tokens),
                u64::from(usage.completion_tokens),
            ),
            None => self.price(input_tokens, self.estimate_tokens(output_chars)),
        }
    }

    /// Book `actual` USD for a call, settling its reservation when it has
    /// one, and kill the budget when the spend crosses the cap.
    fn book(&self, open: Option<OpenCall>, actual: f64) {
        let over_cap = match open {
            Some(call) => call.reservation.settle(actual).over_cap,
            None => {
                self.budget
                    .charge_usd_force(
                        actual,
                        llm_budget::ChargeContext::new().model(self.model.clone()),
                    )
                    .over_cap
            }
        };
        if over_cap {
            self.budget.kill("settled spend crossed the cap");
        }
    }

    /// Book the call identified by `request_id` from its reply.
    fn settle(&self, request_id: u64, usage: Option<&Usage>, output_chars: usize) {
        let open = self.take_open(request_id);
        let input_tokens = open.as_ref().map_or(0, |call| call.input_tokens);
        let actual = self.actual_cost(input_tokens, usage, output_chars);
        self.book(open, actual);
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
        let input_tokens = self.input_tokens(input);
        let estimate = self.price(input_tokens, self.reserve_output_tokens);
        match self.budget.reserve(estimate) {
            Ok(reservation) => {
                self.open.lock().unwrap_or_else(|p| p.into_inner()).insert(
                    context.request_id,
                    OpenCall {
                        reservation,
                        input_tokens,
                        estimate_usd: estimate,
                    },
                );
                Ok(GuardDecision::pass())
            }
            Err(_) => Ok(GuardDecision::reject(self.exhausted_violation())),
        }
    }

    fn finish(&self, context: &GuardContext, outcome: &RequestOutcome) {
        // Still open: the request ended before the output side settled it.
        let Some(call) = self.take_open(context.request_id) else {
            return;
        };
        match outcome.stage {
            // Nothing reached the provider, or it refused the call: no spend.
            RequestStage::NotSent | RequestStage::Failed => call.reservation.release(),
            // Cancelled while waiting on the provider, which may still bill
            // for it: charge what was reserved.
            RequestStage::InFlight => {
                let estimate = call.estimate_usd;
                self.book(Some(call), estimate);
            }
            // The provider answered (another output guard blocked the reply,
            // or a stream was dropped or failed part way): charge the prompt
            // and the output delivered before the request ended.
            RequestStage::Responded => {
                let actual = self.actual_cost(
                    call.input_tokens,
                    outcome.usage.as_ref(),
                    outcome.output_chars,
                );
                self.book(Some(call), actual);
            }
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
            GuardedOutput::Chat(chat) => {
                let output_chars = chat.text.as_ref().map_or(0, String::len)
                    + chat.thinking.as_ref().map_or(0, String::len)
                    + chat.tool_calls.as_ref().map_or(0, |calls| {
                        calls
                            .iter()
                            .map(|call| call.function.name.len() + call.function.arguments.len())
                            .sum()
                    });
                self.settle(context.request_id, chat.usage.as_ref(), output_chars);
            }
            GuardedOutput::Completion(completion) => self.settle(
                context.request_id,
                completion.usage.as_ref(),
                completion.text.len(),
            ),
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
    async fn charges_estimate_when_no_usage_reported() {
        let b = budget(1.0);
        let guard = BudgetGuard::new(b, "claude-haiku-4-5").unwrap();
        let mut input = chat_input(&"x".repeat(4_000));
        let ctx = GuardContext::new(crate::guard::GuardOperation::ChatStream);
        InputGuard::inspect(&guard, &mut input, &ctx).await.unwrap();
        assert!(guard.budget.reserved_usd() > 0.0);

        let mut output = GuardedOutput::Chat(crate::guard::ChatGuardOutput {
            text: Some("y".repeat(2_000)),
            tool_calls: None,
            thinking: None,
            usage: None,
        });
        OutputGuard::inspect(&guard, &mut output, &ctx)
            .await
            .unwrap();
        assert_eq!(guard.budget.reserved_usd(), 0.0);
        // 1k estimated input at $1/M + 500 estimated output at $5/M
        assert!((guard.budget.spent_usd() - 0.0035).abs() < 1e-9);
    }

    fn outcome(stage: RequestStage, output_chars: usize, usage: Option<Usage>) -> RequestOutcome {
        RequestOutcome {
            stage,
            output_chars,
            usage,
        }
    }

    #[tokio::test]
    async fn finish_releases_when_nothing_was_spent() {
        for stage in [RequestStage::NotSent, RequestStage::Failed] {
            let guard = BudgetGuard::new(budget(1.0), "claude-haiku-4-5").unwrap();
            let mut input = chat_input("hello");
            let ctx = GuardContext::new(crate::guard::GuardOperation::Chat);
            InputGuard::inspect(&guard, &mut input, &ctx).await.unwrap();
            assert!(guard.budget.reserved_usd() > 0.0);

            InputGuard::finish(&guard, &ctx, &outcome(stage, 0, None));
            assert_eq!(guard.budget.reserved_usd(), 0.0);
            assert_eq!(guard.budget.spent_usd(), 0.0);
            assert!(guard.open.lock().unwrap().is_empty());
        }
    }

    #[tokio::test]
    async fn finish_charges_reservation_when_cancelled_in_flight() {
        let guard = BudgetGuard::new(budget(1.0), "claude-haiku-4-5")
            .unwrap()
            .reserve_output_tokens(1_000);
        let mut input = chat_input(&"x".repeat(4_000));
        let ctx = GuardContext::new(crate::guard::GuardOperation::Chat);
        InputGuard::inspect(&guard, &mut input, &ctx).await.unwrap();

        InputGuard::finish(&guard, &ctx, &outcome(RequestStage::InFlight, 0, None));
        assert_eq!(guard.budget.reserved_usd(), 0.0);
        // 1k estimated input at $1/M + 1k reserved output at $5/M
        assert!((guard.budget.spent_usd() - 0.006).abs() < 1e-9);
    }

    #[tokio::test]
    async fn finish_charges_output_received_before_the_request_ended() {
        let guard = BudgetGuard::new(budget(1.0), "claude-haiku-4-5")
            .unwrap()
            .reserve_output_tokens(10_000);
        let mut input = chat_input(&"x".repeat(4_000));
        let ctx = GuardContext::new(crate::guard::GuardOperation::ChatStream);
        InputGuard::inspect(&guard, &mut input, &ctx).await.unwrap();

        InputGuard::finish(&guard, &ctx, &outcome(RequestStage::Responded, 2_000, None));
        assert_eq!(guard.budget.reserved_usd(), 0.0);
        // 1k estimated input at $1/M + 500 estimated output at $5/M, not the
        // 10k output tokens that were reserved.
        assert!((guard.budget.spent_usd() - 0.0035).abs() < 1e-9);

        // Reported usage wins over the estimate.
        let ctx = GuardContext::new(crate::guard::GuardOperation::ChatStream);
        InputGuard::inspect(&guard, &mut input, &ctx).await.unwrap();
        let usage = Usage {
            prompt_tokens: 2_000,
            completion_tokens: 1_000,
            total_tokens: 3_000,
            completion_tokens_details: None,
            prompt_tokens_details: None,
        };
        InputGuard::finish(
            &guard,
            &ctx,
            &outcome(RequestStage::Responded, 10, Some(usage)),
        );
        assert!((guard.budget.spent_usd() - (0.0035 + 0.007)).abs() < 1e-9);
    }

    #[tokio::test]
    async fn finish_after_settlement_is_a_no_op() {
        let guard = BudgetGuard::new(budget(1.0), "claude-haiku-4-5").unwrap();
        let mut input = chat_input("hello world");
        let ctx = GuardContext::new(crate::guard::GuardOperation::Chat);
        InputGuard::inspect(&guard, &mut input, &ctx).await.unwrap();
        let mut output = chat_output(1_000, 500);
        OutputGuard::inspect(&guard, &mut output, &ctx)
            .await
            .unwrap();

        InputGuard::finish(
            &guard,
            &ctx,
            &outcome(RequestStage::Responded, 99_999, None),
        );
        assert!((guard.budget.spent_usd() - 0.0035).abs() < 1e-9);
    }

    mod provider {
        use std::pin::Pin;

        use async_trait::async_trait;
        use autoagents_llm::{
            HasConfig, LLMProvider, NoConfig, ToolCall,
            chat::{
                ChatMessage, ChatProvider, ChatResponse, StreamChoice, StreamChunk, StreamDelta,
                StreamResponse, StructuredOutputFormat, Tool, Usage,
            },
            completion::{CompletionProvider, CompletionRequest, CompletionResponse},
            embedding::EmbeddingProvider,
            error::LLMError,
            models::ModelsProvider,
        };
        use futures::{FutureExt, Stream, StreamExt, stream};

        use super::*;
        use crate::Guardrails;

        /// Every answer is 2,000 chars (500 estimated tokens) and, where the
        /// API carries it, reports 1k prompt + 500 completion tokens.
        const ANSWER_CHARS: usize = 2_000;

        fn usage(prompt: u32, completion: u32) -> Usage {
            Usage {
                prompt_tokens: prompt,
                completion_tokens: completion,
                total_tokens: prompt + completion,
                completion_tokens_details: None,
                prompt_tokens_details: None,
            }
        }

        fn answer() -> String {
            "y".repeat(ANSWER_CHARS)
        }

        /// How the mock answers plain (non-streaming) chat calls.
        #[derive(Clone, Copy)]
        enum ChatMode {
            Fail,
            Answer,
            Hang,
        }

        #[derive(Debug)]
        struct Reply;

        impl std::fmt::Display for Reply {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("reply")
            }
        }

        impl ChatResponse for Reply {
            fn text(&self) -> Option<String> {
                Some(answer())
            }

            fn tool_calls(&self) -> Option<Vec<ToolCall>> {
                None
            }

            fn usage(&self) -> Option<Usage> {
                Some(usage(1_000, 500))
            }
        }

        struct MockProvider {
            chat: ChatMode,
        }

        #[async_trait]
        impl ChatProvider for MockProvider {
            async fn chat_with_tools(
                &self,
                _messages: &[ChatMessage],
                _tools: Option<&[Tool]>,
                _json_schema: Option<StructuredOutputFormat>,
            ) -> Result<Box<dyn ChatResponse>, LLMError> {
                match self.chat {
                    ChatMode::Fail => {
                        Err(LLMError::ProviderError("upstream unavailable".to_string()))
                    }
                    ChatMode::Answer => Ok(Box::new(Reply)),
                    ChatMode::Hang => futures::future::pending().await,
                }
            }

            /// Two text chunks of one answer each, no usage reported.
            async fn chat_stream(
                &self,
                _messages: &[ChatMessage],
                _json_schema: Option<StructuredOutputFormat>,
            ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
            {
                Ok(Box::pin(stream::iter(vec![Ok(answer()), Ok(answer())])))
            }

            async fn chat_stream_struct(
                &self,
                _messages: &[ChatMessage],
                _tools: Option<&[Tool]>,
                _json_schema: Option<StructuredOutputFormat>,
            ) -> Result<
                Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>,
                LLMError,
            > {
                let text = StreamResponse {
                    choices: vec![StreamChoice {
                        delta: StreamDelta {
                            content: Some(answer()),
                            reasoning_content: None,
                            tool_calls: None,
                        },
                    }],
                    usage: None,
                };
                let last = StreamResponse {
                    choices: vec![],
                    usage: Some(usage(1_000, 500)),
                };
                Ok(Box::pin(stream::iter(vec![Ok(text), Ok(last)])))
            }

            async fn chat_stream_with_tools(
                &self,
                _messages: &[ChatMessage],
                _tools: Option<&[Tool]>,
                _json_schema: Option<StructuredOutputFormat>,
            ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError>
            {
                Ok(Box::pin(stream::iter(vec![
                    Ok(StreamChunk::Text(answer())),
                    Ok(StreamChunk::Usage(usage(1_000, 500))),
                ])))
            }
        }

        #[async_trait]
        impl CompletionProvider for MockProvider {
            async fn complete(
                &self,
                _req: &CompletionRequest,
                _json_schema: Option<StructuredOutputFormat>,
            ) -> Result<CompletionResponse, LLMError> {
                Err(LLMError::ProviderError("upstream unavailable".to_string()))
            }
        }

        #[async_trait]
        impl EmbeddingProvider for MockProvider {
            async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
                Ok(vec![])
            }
        }

        #[async_trait]
        impl ModelsProvider for MockProvider {}

        impl LLMProvider for MockProvider {}

        impl HasConfig for MockProvider {
            type Config = NoConfig;
        }

        /// An output guard that blocks every reply.
        struct BlockEverything;

        #[async_trait]
        impl OutputGuard for BlockEverything {
            fn name(&self) -> &'static str {
                "block-everything"
            }

            async fn inspect(
                &self,
                _output: &mut GuardedOutput,
                _context: &GuardContext,
            ) -> Result<GuardDecision, GuardError> {
                Ok(GuardDecision::reject(GuardViolation::new(
                    "blocked",
                    GuardCategory::Custom("test".to_string()),
                    GuardSeverity::High,
                    "blocked by test guard",
                )))
            }
        }

        fn mock(chat: ChatMode) -> Arc<dyn LLMProvider> {
            Arc::new(MockProvider { chat })
        }

        fn wrapped(guard: &BudgetGuard, chat: ChatMode) -> Arc<dyn LLMProvider> {
            Guardrails::builder()
                .input_guard(guard.clone())
                .output_guard(guard.clone())
                .build()
                .wrap(mock(chat))
        }

        /// A 4,000-char prompt: 1k estimated tokens, $0.001 on haiku.
        fn messages() -> Vec<ChatMessage> {
            vec![ChatMessage::user().content("x".repeat(4_000)).build()]
        }

        fn guard(reserve_output_tokens: u64) -> BudgetGuard {
            BudgetGuard::new(budget(1.0), "claude-haiku-4-5")
                .unwrap()
                .reserve_output_tokens(reserve_output_tokens)
        }

        fn assert_spent(guard: &BudgetGuard, expected: f64) {
            let spent = guard.budget.spent_usd();
            assert!(
                (spent - expected).abs() < 1e-9,
                "spent {spent}, expected {expected}"
            );
            assert_eq!(guard.budget.reserved_usd(), 0.0);
            assert!(guard.open.lock().unwrap().is_empty());
        }

        #[tokio::test]
        async fn provider_error_releases_reservation() {
            let guard = guard(1_000);
            let provider = wrapped(&guard, ChatMode::Fail);

            assert!(provider.chat(&messages(), None).await.is_err());
            assert_spent(&guard, 0.0);
        }

        #[tokio::test]
        async fn cancelled_in_flight_call_charges_reservation() {
            let guard = guard(1_000);
            let provider = wrapped(&guard, ChatMode::Hang);

            // Poll once (the request is sent and waits on the provider), then
            // drop the future, as a timeout would.
            assert!(provider.chat(&messages(), None).now_or_never().is_none());
            // 1k prompt at $1/M + 1k reserved output at $5/M
            assert_spent(&guard, 0.006);
        }

        #[tokio::test]
        async fn reply_blocked_by_another_output_guard_is_still_charged() {
            let guard = guard(1_000);
            // The blocking guard runs first, so the budget's own output
            // side never sees the reply.
            let provider = Guardrails::builder()
                .input_guard(guard.clone())
                .output_guard(BlockEverything)
                .output_guard(guard.clone())
                .build()
                .wrap(mock(ChatMode::Answer));

            assert!(provider.chat(&messages(), None).await.is_err());
            // Reported usage: 1k input at $1/M + 500 output at $5/M
            assert_spent(&guard, 0.0035);
        }

        #[tokio::test]
        async fn input_only_registration_still_charges() {
            let guard = guard(1_000);
            let provider = Guardrails::builder()
                .input_guard(guard.clone())
                .build()
                .wrap(mock(ChatMode::Answer));

            provider.chat(&messages(), None).await.unwrap();
            assert_spent(&guard, 0.0035);

            let mut stream = provider
                .chat_stream_with_tools(&messages(), None, None)
                .await
                .unwrap();
            while let Some(chunk) = stream.next().await {
                chunk.unwrap();
            }
            drop(stream);
            assert_spent(&guard, 0.007);

            // Plain text stream, no usage: 1k prompt + 1k answer tokens.
            let mut stream = provider.chat_stream(&messages(), None).await.unwrap();
            let mut chunks = 0;
            while let Some(chunk) = stream.next().await {
                assert_eq!(chunk.unwrap().len(), ANSWER_CHARS);
                chunks += 1;
            }
            assert_eq!(chunks, 2);
            drop(stream);
            assert_spent(&guard, 0.013);
        }

        #[tokio::test]
        async fn struct_stream_settles_final_usage() {
            let guard = guard(0);
            let provider = wrapped(&guard, ChatMode::Answer);

            let mut stream = provider
                .chat_stream_struct(&messages(), None, None)
                .await
                .unwrap();
            while let Some(chunk) = stream.next().await {
                chunk.unwrap();
            }
            // 1k input at $1/M + 500 output at $5/M
            assert_spent(&guard, 0.0035);
        }

        #[tokio::test]
        async fn tool_stream_settles_final_usage() {
            let guard = guard(0);
            let provider = wrapped(&guard, ChatMode::Answer);

            let mut stream = provider
                .chat_stream_with_tools(&messages(), None, None)
                .await
                .unwrap();
            while let Some(chunk) = stream.next().await {
                chunk.unwrap();
            }
            assert_spent(&guard, 0.0035);
        }

        #[tokio::test]
        async fn text_stream_without_usage_charges_estimate() {
            let guard = guard(0);
            let provider = wrapped(&guard, ChatMode::Answer);

            let mut stream = provider.chat_stream(&messages(), None).await.unwrap();
            while let Some(chunk) = stream.next().await {
                chunk.unwrap();
            }
            // 1k estimated input at $1/M + 1k estimated output at $5/M
            assert_spent(&guard, 0.006);
        }

        // A stream dropped after one 2,000-char chunk is charged for the
        // prompt plus that chunk (1k input + 500 output tokens = $0.0035),
        // not released, and not the 10k output tokens that were reserved.

        #[tokio::test]
        async fn dropped_tool_stream_charges_delivered_output() {
            let guard = guard(10_000);
            let provider = wrapped(&guard, ChatMode::Answer);

            let mut stream = provider
                .chat_stream_with_tools(&messages(), None, None)
                .await
                .unwrap();
            stream.next().await.unwrap().unwrap();
            assert!(guard.budget.reserved_usd() > 0.0);
            drop(stream);

            assert_spent(&guard, 0.0035);
        }

        #[tokio::test]
        async fn dropped_struct_stream_charges_delivered_output() {
            let guard = guard(10_000);
            let provider = wrapped(&guard, ChatMode::Answer);

            let mut stream = provider
                .chat_stream_struct(&messages(), None, None)
                .await
                .unwrap();
            stream.next().await.unwrap().unwrap();
            drop(stream);

            assert_spent(&guard, 0.0035);
        }

        #[tokio::test]
        async fn dropped_text_stream_charges_delivered_output() {
            let guard = guard(10_000);
            let provider = wrapped(&guard, ChatMode::Answer);

            let mut stream = provider.chat_stream(&messages(), None).await.unwrap();
            stream.next().await.unwrap().unwrap();
            drop(stream);

            assert_spent(&guard, 0.0035);
        }

        #[tokio::test]
        async fn stream_dropped_before_first_chunk_charges_prompt() {
            let guard = guard(10_000);
            let provider = wrapped(&guard, ChatMode::Answer);

            let stream = provider
                .chat_stream_with_tools(&messages(), None, None)
                .await
                .unwrap();
            drop(stream);

            // The provider accepted the prompt: 1k input at $1/M.
            assert_spent(&guard, 0.001);
        }

        #[tokio::test]
        async fn repeated_early_drops_exhaust_the_budget() {
            // $0.01 cap; each dropped stream costs $0.0035.
            let guard = BudgetGuard::new(budget(0.01), "claude-haiku-4-5").unwrap();
            let provider = wrapped(&guard, ChatMode::Answer);

            let mut rejected = false;
            for _ in 0..10 {
                match provider
                    .chat_stream_with_tools(&messages(), None, None)
                    .await
                {
                    Ok(mut stream) => {
                        stream.next().await.unwrap().unwrap();
                    }
                    Err(_) => {
                        rejected = true;
                        break;
                    }
                }
            }
            assert!(rejected, "early drops must not keep the budget open");
        }
    }

    #[test]
    fn unknown_model_is_rejected() {
        let b = budget(1.0);
        assert!(BudgetGuard::new(b, "not-a-model").is_err());
    }

    #[tokio::test]
    async fn sub_agent_spend_is_booked_on_the_parent_and_the_rest_refunded() {
        let parent = BudgetGuard::new(budget(1.0), "claude-haiku-4-5").unwrap();
        let sub = parent
            .sub_agent("researcher", llm_budget::Limit::Usd(0.25))
            .unwrap();
        assert!((parent.budget.reserved_usd() - 0.25).abs() < 1e-9);
        assert!((sub.budget.limit_usd() - 0.25).abs() < 1e-9);

        let mut input = chat_input("hello world");
        let ctx = GuardContext::new(crate::guard::GuardOperation::Chat);
        InputGuard::inspect(&sub, &mut input, &ctx).await.unwrap();
        let mut output = chat_output(1_000, 500);
        OutputGuard::inspect(&sub, &mut output, &ctx).await.unwrap();
        // Booked on the sub-agent and on the parent at the same moment.
        assert!((sub.budget.spent_usd() - 0.0035).abs() < 1e-9);
        assert!((parent.budget.spent_usd() - 0.0035).abs() < 1e-9);

        // Dropping the last clone gives the unused slice back.
        let clone = sub.clone();
        drop(sub);
        assert!((parent.budget.reserved_usd() - 0.2465).abs() < 1e-9);
        drop(clone);
        assert_eq!(parent.budget.reserved_usd(), 0.0);
        assert!((parent.budget.remaining_usd() - 0.9965).abs() < 1e-9);
    }

    #[tokio::test]
    async fn parallel_sub_agents_cannot_overcommit_the_parent() {
        let parent = BudgetGuard::new(budget(1.0), "claude-haiku-4-5").unwrap();
        let a = parent.sub_agent("a", llm_budget::Limit::Usd(0.6)).unwrap();
        // The second slice does not fit in the $0.40 left: refused up front.
        assert!(parent.sub_agent("b", llm_budget::Limit::Usd(0.6)).is_err());
        let b = parent
            .sub_agent("b", llm_budget::Limit::Fraction(1.0))
            .unwrap();
        assert!((b.budget.limit_usd() - 0.4).abs() < 1e-9);
        assert!(parent.budget.remaining_usd() < 1e-9);
        drop((a, b));
        assert!((parent.budget.remaining_usd() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn competing_sub_agent_calls_never_overcommit_the_parent() {
        // 16 threads ask for a $0.10 slice of a $1.00 budget at the same
        // moment, over and over: exactly 10 may be admitted each round.
        for _ in 0..200 {
            let parent = BudgetGuard::new(budget(1.0), "claude-haiku-4-5").unwrap();
            let start = Arc::new(std::sync::Barrier::new(16));
            let handles: Vec<_> = (0..16)
                .map(|i| {
                    let parent = parent.clone();
                    let start = Arc::clone(&start);
                    std::thread::spawn(move || {
                        start.wait();
                        parent
                            .sub_agent(format!("sub-{i}"), llm_budget::Limit::Usd(0.1))
                            .ok()
                    })
                })
                .collect();
            let admitted: Vec<BudgetGuard> = handles
                .into_iter()
                .filter_map(|h| h.join().unwrap())
                .collect();
            let reserved: f64 = admitted.iter().map(|g| g.budget.limit_usd()).sum();
            assert_eq!(admitted.len(), 10);
            assert!(reserved <= 1.0 + 1e-9);
            assert!(parent.budget.remaining_usd() < 1e-9);
            drop(admitted);
            assert!((parent.budget.remaining_usd() - 1.0).abs() < 1e-9);
        }
    }

    #[tokio::test]
    async fn a_sub_agent_over_its_slice_is_stopped_without_killing_the_parent() {
        let parent = BudgetGuard::new(budget(1.0), "claude-haiku-4-5").unwrap();
        let sub = parent
            .sub_agent("runaway", llm_budget::Limit::Usd(0.001))
            .unwrap();
        let ctx = GuardContext::new(crate::guard::GuardOperation::Chat);
        let mut output = chat_output(10_000, 1_000); // $0.015, over the $0.001 slice
        OutputGuard::inspect(&sub, &mut output, &ctx).await.unwrap();
        assert!(sub.budget.is_killed());
        assert!(!parent.budget.is_killed());
        // The spend happened, so the parent carries it.
        assert!((parent.budget.spent_usd() - 0.015).abs() < 1e-9);

        let mut input = chat_input("hi");
        let ctx2 = GuardContext::new(crate::guard::GuardOperation::Chat);
        let sub_decision = InputGuard::inspect(&sub, &mut input, &ctx2).await.unwrap();
        assert!(matches!(sub_decision, GuardDecision::Reject(_)));
        let parent_decision = InputGuard::inspect(&parent, &mut input, &ctx2)
            .await
            .unwrap();
        assert!(matches!(parent_decision, GuardDecision::Pass));
    }
}
