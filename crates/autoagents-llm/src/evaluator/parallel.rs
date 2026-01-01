//! Module for parallel evaluation of multiple LLM providers.
//!
//! This module provides functionality to run the same prompt through multiple LLMs
//! in parallel and select the best response based on scoring functions.

use std::time::Instant;

use futures::future::join_all;

use crate::{
    chat::{ChatMessage, Tool},
    completion::CompletionRequest,
    error::LLMError,
    LLMProvider,
};

use super::ScoringFn;

/// Result of a parallel evaluation including response, score, and timing information
#[derive(Debug)]
pub struct ParallelEvalResult {
    /// The text response from the LLM
    pub text: String,
    /// Score assigned by the scoring function
    pub score: f32,
    /// Time taken to generate the response in milliseconds
    pub time_ms: u128,
    /// Identifier of the provider that generated this response
    pub provider_id: String,
}

/// Evaluator for running multiple LLM providers in parallel and selecting the best response
pub struct ParallelEvaluator {
    /// Collection of LLM providers to evaluate with their identifiers
    providers: Vec<(String, Box<dyn LLMProvider>)>,
    /// Scoring functions to evaluate responses
    scoring_fns: Vec<Box<ScoringFn>>,
    /// Whether to include timing information in results
    include_timing: bool,
}

impl ParallelEvaluator {
    /// Creates a new parallel evaluator
    ///
    /// # Arguments
    /// * `providers` - Vector of (id, provider) tuples to evaluate
    pub fn new(providers: Vec<(String, Box<dyn LLMProvider>)>) -> Self {
        Self {
            providers,
            scoring_fns: Vec::new(),
            include_timing: true,
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
        self.scoring_fns.push(Box::new(f));
        self
    }

    /// Sets whether to include timing information in results
    pub fn include_timing(mut self, include: bool) -> Self {
        self.include_timing = include;
        self
    }

    /// Evaluates chat responses from all providers in parallel for the given messages
    ///
    /// # Arguments
    /// * `messages` - Chat messages to send to each provider
    ///
    /// # Returns
    /// Vector of evaluation results containing responses, scores, and timing information
    pub async fn evaluate_chat_parallel(
        &self,
        messages: &[ChatMessage],
    ) -> Result<Vec<ParallelEvalResult>, LLMError> {
        let futures = self
            .providers
            .iter()
            .map(|(id, provider)| {
                let id = id.clone();
                let messages = messages.to_vec();
                async move {
                    let start = Instant::now();
                    let result = provider.chat(&messages, None).await;
                    let elapsed = start.elapsed().as_millis();
                    (id, result, elapsed)
                }
            })
            .collect::<Vec<_>>();

        let results = join_all(futures).await;

        let mut eval_results = Vec::new();
        for (id, result, elapsed) in results {
            match result {
                Ok(response) => {
                    let text = response.text().unwrap_or_default();
                    let score = self.compute_score(&text);
                    eval_results.push(ParallelEvalResult {
                        text,
                        score,
                        time_ms: elapsed,
                        provider_id: id,
                    });
                }
                Err(e) => {
                    // Log the error but continue with other results
                    eprintln!("Error from provider {id}: {e}");
                }
            }
        }

        Ok(eval_results)
    }

    /// Evaluates chat responses with tools from all providers in parallel
    ///
    /// # Arguments
    /// * `messages` - Chat messages to send to each provider
    /// * `tools` - Optional tools to use in the chat
    ///
    /// # Returns
    /// Vector of evaluation results
    pub async fn evaluate_chat_with_tools_parallel(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Vec<ParallelEvalResult>, LLMError> {
        let futures = self
            .providers
            .iter()
            .map(|(id, provider)| {
                let id = id.clone();
                let messages = messages.to_vec();
                async move {
                    let start = Instant::now();
                    let result = provider.chat_with_tools(&messages, tools, None).await;
                    let elapsed = start.elapsed().as_millis();
                    (id, result, elapsed)
                }
            })
            .collect::<Vec<_>>();

        let results = join_all(futures).await;

        let mut eval_results = Vec::new();
        for (id, result, elapsed) in results {
            match result {
                Ok(response) => {
                    let text = response.text().unwrap_or_default();
                    let score = self.compute_score(&text);
                    eval_results.push(ParallelEvalResult {
                        text,
                        score,
                        time_ms: elapsed,
                        provider_id: id,
                    });
                }
                Err(e) => {
                    // Log the error but continue with other results
                    eprintln!("Error from provider {id}: {e}");
                }
            }
        }

        Ok(eval_results)
    }

    /// Evaluates completion responses from all providers in parallel
    ///
    /// # Arguments
    /// * `request` - Completion request to send to each provider
    ///
    /// # Returns
    /// Vector of evaluation results
    pub async fn evaluate_completion_parallel(
        &self,
        request: &CompletionRequest,
    ) -> Result<Vec<ParallelEvalResult>, LLMError> {
        let futures = self
            .providers
            .iter()
            .map(|(id, provider)| {
                let id = id.clone();
                let request = request.clone();
                async move {
                    let start = Instant::now();
                    let result = provider.complete(&request, None).await;
                    let elapsed = start.elapsed().as_millis();
                    (id, result, elapsed)
                }
            })
            .collect::<Vec<_>>();

        let results = join_all(futures).await;

        let mut eval_results = Vec::new();
        for (id, result, elapsed) in results {
            match result {
                Ok(response) => {
                    let score = self.compute_score(&response.text);
                    eval_results.push(ParallelEvalResult {
                        text: response.text,
                        score,
                        time_ms: elapsed,
                        provider_id: id,
                    });
                }
                Err(e) => {
                    // Log the error but continue with other results
                    eprintln!("Error from provider {id}: {e}");
                }
            }
        }

        Ok(eval_results)
    }

    /// Returns the best response based on scoring
    ///
    /// # Arguments
    /// * `results` - Vector of evaluation results
    ///
    /// # Returns
    /// The best result or None if no results are available
    pub fn best_response<'a>(
        &self,
        results: &'a [ParallelEvalResult],
    ) -> Option<&'a ParallelEvalResult> {
        if results.is_empty() {
            return None;
        }

        results.iter().max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
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
        for sc in &self.scoring_fns {
            total += sc(response);
        }
        total
    }
}
