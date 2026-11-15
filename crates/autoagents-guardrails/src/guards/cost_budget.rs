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
//! A request that fails, is rejected by another guard, or is cancelled (a
//! dropped future or stream) releases its reservation instead of holding it.
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
/// `output_guard(guard)` share the budget). Register it first among the
/// output guards, so a call another output guard blocks is still charged:
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
    open: Arc<Mutex<HashMap<u64, OpenCall>>>,
}

/// A call that passed the input side and has not been settled yet.
struct OpenCall {
    reservation: Reservation,
    /// Prompt tokens estimated on the input side, used to price the call
    /// when the provider reports no usage.
    input_tokens: u64,
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

    /// Book the call identified by `request_id`. `output_chars` is the size of
    /// the answer, used to price it only when `usage` is `None`.
    fn settle(&self, request_id: u64, usage: Option<&Usage>, output_chars: usize) {
        let open = self.take_open(request_id);
        let actual = match usage {
            Some(usage) => self.price(
                u64::from(usage.prompt_tokens),
                u64::from(usage.completion_tokens),
            ),
            // No usage reported (plain text streams, some backends): charge
            // the prompt estimate plus the answer, so these calls still
            // count against the cap.
            None => self.price(
                open.as_ref().map_or(0, |call| call.input_tokens),
                self.estimate_tokens(output_chars),
            ),
        };
        match open {
            Some(call) => {
                let outcome = call.reservation.settle(actual);
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
        let input_tokens = self.input_tokens(input);
        let estimate = self.price(input_tokens, self.reserve_output_tokens);
        match self.budget.reserve(estimate) {
            Ok(reservation) => {
                self.open.lock().unwrap_or_else(|p| p.into_inner()).insert(
                    context.request_id,
                    OpenCall {
                        reservation,
                        input_tokens,
                    },
                );
                Ok(GuardDecision::pass())
            }
            Err(_) => Ok(GuardDecision::reject(self.exhausted_violation())),
        }
    }

    fn finish(&self, context: &GuardContext) {
        // The request ended without reaching settlement (provider error,
        // another guard rejected it, or it was cancelled): release the hold.
        if let Some(call) = self.take_open(context.request_id) {
            call.reservation.release();
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

    #[tokio::test]
    async fn finish_releases_unsettled_reservation() {
        let guard = BudgetGuard::new(budget(1.0), "claude-haiku-4-5").unwrap();
        let mut input = chat_input("hello");
        let ctx = GuardContext::new(crate::guard::GuardOperation::Chat);
        InputGuard::inspect(&guard, &mut input, &ctx).await.unwrap();
        assert!(guard.budget.reserved_usd() > 0.0);

        InputGuard::finish(&guard, &ctx);
        assert_eq!(guard.budget.reserved_usd(), 0.0);
        assert_eq!(guard.budget.spent_usd(), 0.0);
        assert!(guard.open.lock().unwrap().is_empty());
    }

    mod provider {
        use std::pin::Pin;

        use async_trait::async_trait;
        use autoagents_llm::{
            HasConfig, LLMProvider, NoConfig,
            chat::{
                ChatMessage, ChatProvider, ChatResponse, StreamChoice, StreamChunk, StreamDelta,
                StreamResponse, StructuredOutputFormat, Tool, Usage,
            },
            completion::{CompletionProvider, CompletionRequest, CompletionResponse},
            embedding::EmbeddingProvider,
            error::LLMError,
            models::ModelsProvider,
        };
        use futures::{Stream, StreamExt, stream};

        use super::*;
        use crate::Guardrails;

        fn usage(prompt: u32, completion: u32) -> Usage {
            Usage {
                prompt_tokens: prompt,
                completion_tokens: completion,
                total_tokens: prompt + completion,
                completion_tokens_details: None,
                prompt_tokens_details: None,
            }
        }

        /// Fails plain chat calls and streams fixed chunks with a final
        /// usage report of 1k prompt + 500 completion tokens.
        struct MockProvider;

        #[async_trait]
        impl ChatProvider for MockProvider {
            async fn chat_with_tools(
                &self,
                _messages: &[ChatMessage],
                _tools: Option<&[Tool]>,
                _json_schema: Option<StructuredOutputFormat>,
            ) -> Result<Box<dyn ChatResponse>, LLMError> {
                Err(LLMError::ProviderError("upstream unavailable".to_string()))
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
                            content: Some("hello".to_string()),
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
                    Ok(StreamChunk::Text("hello".to_string())),
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

        fn wrapped(guard: &BudgetGuard) -> Arc<dyn LLMProvider> {
            Guardrails::builder()
                .input_guard(guard.clone())
                .output_guard(guard.clone())
                .build()
                .wrap(Arc::new(MockProvider))
        }

        fn messages() -> Vec<ChatMessage> {
            vec![ChatMessage::user().content("hello").build()]
        }

        #[tokio::test]
        async fn provider_error_releases_reservation() {
            let guard = BudgetGuard::new(budget(1.0), "claude-haiku-4-5")
                .unwrap()
                .reserve_output_tokens(1_000);
            let provider = wrapped(&guard);

            assert!(provider.chat(&messages(), None).await.is_err());
            assert_eq!(guard.budget.reserved_usd(), 0.0);
            assert_eq!(guard.budget.spent_usd(), 0.0);
            assert!(guard.open.lock().unwrap().is_empty());
        }

        #[tokio::test]
        async fn struct_stream_settles_final_usage() {
            let guard = BudgetGuard::new(budget(1.0), "claude-haiku-4-5").unwrap();
            let provider = wrapped(&guard);

            let mut stream = provider
                .chat_stream_struct(&messages(), None, None)
                .await
                .unwrap();
            while let Some(chunk) = stream.next().await {
                chunk.unwrap();
            }
            // 1k input at $1/M + 500 output at $5/M
            assert!((guard.budget.spent_usd() - 0.0035).abs() < 1e-9);
            assert_eq!(guard.budget.reserved_usd(), 0.0);
        }

        #[tokio::test]
        async fn tool_stream_settles_final_usage() {
            let guard = BudgetGuard::new(budget(1.0), "claude-haiku-4-5").unwrap();
            let provider = wrapped(&guard);

            let mut stream = provider
                .chat_stream_with_tools(&messages(), None, None)
                .await
                .unwrap();
            while let Some(chunk) = stream.next().await {
                chunk.unwrap();
            }
            assert!((guard.budget.spent_usd() - 0.0035).abs() < 1e-9);
            assert_eq!(guard.budget.reserved_usd(), 0.0);
        }

        #[tokio::test]
        async fn dropped_stream_releases_reservation() {
            let guard = BudgetGuard::new(budget(1.0), "claude-haiku-4-5")
                .unwrap()
                .reserve_output_tokens(1_000);
            let provider = wrapped(&guard);

            let mut stream = provider
                .chat_stream_with_tools(&messages(), None, None)
                .await
                .unwrap();
            stream.next().await.unwrap().unwrap();
            assert!(guard.budget.reserved_usd() > 0.0);
            drop(stream);

            assert_eq!(guard.budget.reserved_usd(), 0.0);
            assert!(guard.open.lock().unwrap().is_empty());
        }
    }

    #[test]
    fn unknown_model_is_rejected() {
        let b = budget(1.0);
        assert!(BudgetGuard::new(b, "not-a-model").is_err());
    }
}
