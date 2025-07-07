//! Module for evaluating and comparing responses from multiple LLM providers.
//!
//! This module provides functionality to run the same prompt through multiple LLMs
//! and score their responses using custom evaluation functions.

mod parallel;

use crate::{chat::ChatMessage, error::LLMError, LLMProvider};

pub use parallel::{ParallelEvalResult, ParallelEvaluator};

/// Type alias for scoring functions that evaluate LLM responses
pub type ScoringFn = dyn Fn(&str) -> f32 + Send + Sync + 'static;

/// Evaluator for comparing responses from multiple LLM providers
pub struct LLMEvaluator {
    /// Collection of LLM providers to evaluate
    llms: Vec<Box<dyn LLMProvider>>,
    /// Optional scoring function to evaluate responses
    scorings_fns: Vec<Box<ScoringFn>>,
}

impl LLMEvaluator {
    /// Creates a new evaluator with the given LLM providers
    ///
    /// # Arguments
    /// * `llms` - Vector of LLM providers to evaluate
    pub fn new(llms: Vec<Box<dyn LLMProvider>>) -> Self {
        Self {
            llms,
            scorings_fns: Vec::new(),
        }
    }

    /// Adds a scoring function to evaluate LLM responses
    ///
    /// # Arguments
    /// * `f` - Function that takes a response string and returns a score
    pub fn scoring<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> f32 + Send + Sync + 'static,
    {
        self.scorings_fns.push(Box::new(f));
        self
    }

    /// Evaluates chat responses from all providers for the given messages
    ///
    /// # Arguments
    /// * `messages` - Chat messages to send to each provider
    ///
    /// # Returns
    /// Vector of evaluation results containing responses and scores
    pub async fn evaluate_chat(
        &self,
        messages: &[ChatMessage],
    ) -> Result<Vec<EvalResult>, LLMError> {
        let mut results = Vec::new();
        for llm in &self.llms {
            let response = llm.chat(messages).await?;
            let score = self.compute_score(&response.text().unwrap_or_default());
            results.push(EvalResult {
                text: response.text().unwrap_or_default(),
                score,
            });
        }
        Ok(results)
    }

    /// Computes the score for a given response
    ///
    /// # Arguments
    /// * `response` - The response to score
    ///
    /// # Returns
    /// The computed score
    fn compute_score(&self, response: &str) -> f32 {
        let mut total = 0.0;
        for sc in &self.scorings_fns {
            total += sc(response);
        }
        total
    }
}

/// Result of evaluating an LLM response
pub struct EvalResult {
    /// The text response from the LLM
    pub text: String,
    /// Score assigned by the scoring function, if any
    pub score: f32,
}
