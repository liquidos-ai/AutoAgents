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
                let tools_clone = tools.map(|t| t.to_vec());
                async move {
                    let start = Instant::now();
                    let result = provider
                        .chat_with_tools(&messages, tools_clone.as_deref(), None)
                        .await;
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

        #[cfg(test)]
        mod tests {
            use super::*;
            use crate::chat::{
                ChatMessage, ChatProvider, ChatResponse, ChatRole, MessageType,
                StructuredOutputFormat,
            };
            use crate::completion::{CompletionProvider, CompletionRequest, CompletionResponse};
            use crate::embedding::EmbeddingProvider;
            use crate::error::LLMError;
            use crate::models::ModelsProvider;
            use crate::ToolCall;
            use async_trait::async_trait;

            // Mock LLM Provider for testing
            struct MockLLMProvider {
                response_text: String,
                should_fail: bool,
                delay_ms: u64,
            }

            impl MockLLMProvider {
                fn new(response_text: &str) -> Self {
                    Self {
                        response_text: response_text.to_string(),
                        should_fail: false,
                        delay_ms: 0,
                    }
                }

                fn with_failure(response_text: &str) -> Self {
                    Self {
                        response_text: response_text.to_string(),
                        should_fail: true,
                        delay_ms: 0,
                    }
                }

                fn with_delay(response_text: &str, delay_ms: u64) -> Self {
                    Self {
                        response_text: response_text.to_string(),
                        should_fail: false,
                        delay_ms,
                    }
                }
            }

            #[async_trait]
            impl ChatProvider for MockLLMProvider {
                async fn chat_with_tools(
                    &self,
                    _messages: &[ChatMessage],
                    _tools: Option<&[crate::chat::Tool]>,
                    _json_schema: Option<StructuredOutputFormat>,
                ) -> Result<Box<dyn ChatResponse>, LLMError> {
                    if self.delay_ms > 0 {
                        tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
                    }

                    if self.should_fail {
                        return Err(LLMError::ProviderError("Mock provider failed".to_string()));
                    }
                    Ok(Box::new(MockChatResponse {
                        text: Some(self.response_text.clone()),
                    }))
                }
            }

            #[async_trait]
            impl CompletionProvider for MockLLMProvider {
                async fn complete(
                    &self,
                    _req: &CompletionRequest,
                    _json_schema: Option<StructuredOutputFormat>,
                ) -> Result<CompletionResponse, LLMError> {
                    if self.delay_ms > 0 {
                        tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
                    }

                    if self.should_fail {
                        return Err(LLMError::ProviderError("Mock provider failed".to_string()));
                    }
                    Ok(CompletionResponse {
                        text: self.response_text.clone(),
                    })
                }
            }

            #[async_trait]
            impl EmbeddingProvider for MockLLMProvider {
                async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
                    if self.should_fail {
                        return Err(LLMError::ProviderError("Mock provider failed".to_string()));
                    }
                    Ok(vec![vec![0.1, 0.2, 0.3]])
                }
            }

            #[async_trait]
            impl ModelsProvider for MockLLMProvider {}

            impl LLMProvider for MockLLMProvider {}

            struct MockChatResponse {
                text: Option<String>,
            }

            impl ChatResponse for MockChatResponse {
                fn text(&self) -> Option<String> {
                    self.text.clone()
                }

                fn tool_calls(&self) -> Option<Vec<ToolCall>> {
                    None
                }
            }

            impl std::fmt::Debug for MockChatResponse {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "MockChatResponse")
                }
            }

            impl std::fmt::Display for MockChatResponse {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "{}", self.text.as_deref().unwrap_or(""))
                }
            }

            #[test]
            fn test_parallel_evaluator_new() {
                let providers = vec![
                    (
                        "provider1".to_string(),
                        Box::new(MockLLMProvider::new("response1")) as Box<dyn LLMProvider>,
                    ),
                    (
                        "provider2".to_string(),
                        Box::new(MockLLMProvider::new("response2")) as Box<dyn LLMProvider>,
                    ),
                ];
                let evaluator = ParallelEvaluator::new(providers);

                assert_eq!(evaluator.providers.len(), 2);
                assert_eq!(evaluator.scoring_fns.len(), 0);
                assert!(evaluator.include_timing);
            }

            #[test]
            fn test_parallel_evaluator_with_scoring() {
                let providers = vec![(
                    "provider1".to_string(),
                    Box::new(MockLLMProvider::new("response1")) as Box<dyn LLMProvider>,
                )];
                let evaluator = ParallelEvaluator::new(providers)
                    .scoring(|response| response.len() as f32)
                    .scoring(|response| if response.contains("good") { 10.0 } else { 0.0 });

                assert_eq!(evaluator.providers.len(), 1);
                assert_eq!(evaluator.scoring_fns.len(), 2);
            }

            #[test]
            fn test_parallel_evaluator_include_timing() {
                let providers = vec![(
                    "provider1".to_string(),
                    Box::new(MockLLMProvider::new("response1")) as Box<dyn LLMProvider>,
                )];
                let evaluator = ParallelEvaluator::new(providers).include_timing(false);

                assert!(!evaluator.include_timing);
            }

            #[test]
            fn test_parallel_evaluator_compute_score() {
                let providers = vec![(
                    "provider1".to_string(),
                    Box::new(MockLLMProvider::new("response1")) as Box<dyn LLMProvider>,
                )];
                let evaluator = ParallelEvaluator::new(providers)
                    .scoring(|response| response.len() as f32)
                    .scoring(|_| 5.0);

                let score = evaluator.compute_score("hello");
                assert_eq!(score, 10.0); // 5 (length) + 5 (fixed score)
            }

            #[test]
            fn test_parallel_evaluator_compute_score_no_functions() {
                let providers = vec![(
                    "provider1".to_string(),
                    Box::new(MockLLMProvider::new("response1")) as Box<dyn LLMProvider>,
                )];
                let evaluator = ParallelEvaluator::new(providers);

                let score = evaluator.compute_score("hello");
                assert_eq!(score, 0.0);
            }

            #[tokio::test]
            async fn test_parallel_evaluator_evaluate_chat_parallel_single_provider() {
                let providers = vec![(
                    "provider1".to_string(),
                    Box::new(MockLLMProvider::new("test response")) as Box<dyn LLMProvider>,
                )];
                let evaluator =
                    ParallelEvaluator::new(providers).scoring(|response| response.len() as f32);

                let messages = vec![ChatMessage {
                    role: ChatRole::User,
                    message_type: MessageType::Text,
                    content: "test message".to_string(),
                }];

                let results = evaluator.evaluate_chat_parallel(&messages).await.unwrap();
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].text, "test response");
                assert_eq!(results[0].score, 13.0); // "test response".len()
                assert_eq!(results[0].provider_id, "provider1");
                assert!(results[0].time_ms >= 0);
            }

            #[tokio::test]
            async fn test_parallel_evaluator_evaluate_chat_parallel_multiple_providers() {
                let providers = vec![
                    (
                        "fast".to_string(),
                        Box::new(MockLLMProvider::new("short")) as Box<dyn LLMProvider>,
                    ),
                    (
                        "slow".to_string(),
                        Box::new(MockLLMProvider::new("this is a longer response"))
                            as Box<dyn LLMProvider>,
                    ),
                ];
                let evaluator =
                    ParallelEvaluator::new(providers).scoring(|response| response.len() as f32);

                let messages = vec![ChatMessage {
                    role: ChatRole::User,
                    message_type: MessageType::Text,
                    content: "test message".to_string(),
                }];

                let results = evaluator.evaluate_chat_parallel(&messages).await.unwrap();
                assert_eq!(results.len(), 2);

                // Find results by provider_id
                let fast_result = results.iter().find(|r| r.provider_id == "fast").unwrap();
                let slow_result = results.iter().find(|r| r.provider_id == "slow").unwrap();

                assert_eq!(fast_result.text, "short");
                assert_eq!(fast_result.score, 5.0);
                assert_eq!(slow_result.text, "this is a longer response");
                assert_eq!(slow_result.score, 26.0);
            }

            #[tokio::test]
            async fn test_parallel_evaluator_evaluate_chat_parallel_with_failure() {
                let providers = vec![
                    (
                        "working".to_string(),
                        Box::new(MockLLMProvider::new("good response")) as Box<dyn LLMProvider>,
                    ),
                    (
                        "failing".to_string(),
                        Box::new(MockLLMProvider::with_failure("bad response"))
                            as Box<dyn LLMProvider>,
                    ),
                ];
                let evaluator =
                    ParallelEvaluator::new(providers).scoring(|response| response.len() as f32);

                let messages = vec![ChatMessage {
                    role: ChatRole::User,
                    message_type: MessageType::Text,
                    content: "test message".to_string(),
                }];

                let results = evaluator.evaluate_chat_parallel(&messages).await.unwrap();
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].text, "good response");
                assert_eq!(results[0].provider_id, "working");
            }

            #[tokio::test]
            async fn test_parallel_evaluator_evaluate_chat_with_tools_parallel() {
                let providers = vec![(
                    "provider1".to_string(),
                    Box::new(MockLLMProvider::new("tool response")) as Box<dyn LLMProvider>,
                )];
                let evaluator =
                    ParallelEvaluator::new(providers).scoring(|response| response.len() as f32);

                let messages = vec![ChatMessage {
                    role: ChatRole::User,
                    message_type: MessageType::Text,
                    content: "test message".to_string(),
                }];

                let results = evaluator
                    .evaluate_chat_with_tools_parallel(&messages, None)
                    .await
                    .unwrap();
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].text, "tool response");
                assert_eq!(results[0].score, 13.0);
            }

            #[tokio::test]
            async fn test_parallel_evaluator_evaluate_completion_parallel() {
                let providers = vec![(
                    "provider1".to_string(),
                    Box::new(MockLLMProvider::new("completion response")) as Box<dyn LLMProvider>,
                )];
                let evaluator =
                    ParallelEvaluator::new(providers).scoring(|response| response.len() as f32);

                let request = CompletionRequest::new("test prompt");

                let results = evaluator
                    .evaluate_completion_parallel(&request)
                    .await
                    .unwrap();
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].text, "completion response");
                assert_eq!(results[0].score, 19.0);
            }

            #[tokio::test]
            async fn test_parallel_evaluator_evaluate_completion_parallel_with_failure() {
                let providers = vec![
                    (
                        "working".to_string(),
                        Box::new(MockLLMProvider::new("good completion")) as Box<dyn LLMProvider>,
                    ),
                    (
                        "failing".to_string(),
                        Box::new(MockLLMProvider::with_failure("bad completion"))
                            as Box<dyn LLMProvider>,
                    ),
                ];
                let evaluator =
                    ParallelEvaluator::new(providers).scoring(|response| response.len() as f32);

                let request = CompletionRequest::new("test prompt");

                let results = evaluator
                    .evaluate_completion_parallel(&request)
                    .await
                    .unwrap();
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].text, "good completion");
                assert_eq!(results[0].provider_id, "working");
            }

            #[test]
            fn test_parallel_evaluator_best_response_empty() {
                let providers = vec![];
                let evaluator = ParallelEvaluator::new(providers);

                let results = vec![];
                let best = evaluator.best_response(&results);
                assert!(best.is_none());
            }

            #[test]
            fn test_parallel_evaluator_best_response_single() {
                let providers = vec![];
                let evaluator = ParallelEvaluator::new(providers);

                let results = vec![ParallelEvalResult {
                    text: "only result".to_string(),
                    score: 5.0,
                    time_ms: 100,
                    provider_id: "provider1".to_string(),
                }];

                let best = evaluator.best_response(&results);
                assert!(best.is_some());
                assert_eq!(best.unwrap().text, "only result");
                assert_eq!(best.unwrap().score, 5.0);
            }

            #[test]
            fn test_parallel_evaluator_best_response_multiple() {
                let providers = vec![];
                let evaluator = ParallelEvaluator::new(providers);

                let results = vec![
                    ParallelEvalResult {
                        text: "low score".to_string(),
                        score: 3.0,
                        time_ms: 100,
                        provider_id: "provider1".to_string(),
                    },
                    ParallelEvalResult {
                        text: "high score".to_string(),
                        score: 8.0,
                        time_ms: 200,
                        provider_id: "provider2".to_string(),
                    },
                    ParallelEvalResult {
                        text: "medium score".to_string(),
                        score: 5.0,
                        time_ms: 150,
                        provider_id: "provider3".to_string(),
                    },
                ];

                let best = evaluator.best_response(&results);
                assert!(best.is_some());
                assert_eq!(best.unwrap().text, "high score");
                assert_eq!(best.unwrap().score, 8.0);
                assert_eq!(best.unwrap().provider_id, "provider2");
            }

            #[test]
            fn test_parallel_evaluator_best_response_tie() {
                let providers = vec![];
                let evaluator = ParallelEvaluator::new(providers);

                let results = vec![
                    ParallelEvalResult {
                        text: "first".to_string(),
                        score: 5.0,
                        time_ms: 100,
                        provider_id: "provider1".to_string(),
                    },
                    ParallelEvalResult {
                        text: "second".to_string(),
                        score: 5.0,
                        time_ms: 200,
                        provider_id: "provider2".to_string(),
                    },
                ];

                let best = evaluator.best_response(&results);
                assert!(best.is_some());
                assert_eq!(best.unwrap().score, 5.0);
                // Should return one of them (implementation dependent)
            }

            #[test]
            fn test_parallel_eval_result_debug() {
                let result = ParallelEvalResult {
                    text: "test response".to_string(),
                    score: 10.5,
                    time_ms: 250,
                    provider_id: "test_provider".to_string(),
                };

                let debug_str = format!("{:?}", result);
                assert!(debug_str.contains("ParallelEvalResult"));
                assert!(debug_str.contains("test response"));
                assert!(debug_str.contains("10.5"));
                assert!(debug_str.contains("250"));
                assert!(debug_str.contains("test_provider"));
            }

            #[tokio::test]
            async fn test_parallel_evaluator_timing() {
                let providers = vec![
                    (
                        "fast".to_string(),
                        Box::new(MockLLMProvider::with_delay("fast response", 10))
                            as Box<dyn LLMProvider>,
                    ),
                    (
                        "slow".to_string(),
                        Box::new(MockLLMProvider::with_delay("slow response", 50))
                            as Box<dyn LLMProvider>,
                    ),
                ];
                let evaluator = ParallelEvaluator::new(providers);

                let messages = vec![ChatMessage {
                    role: ChatRole::User,
                    message_type: MessageType::Text,
                    content: "test message".to_string(),
                }];

                let results = evaluator.evaluate_chat_parallel(&messages).await.unwrap();
                assert_eq!(results.len(), 2);

                let fast_result = results.iter().find(|r| r.provider_id == "fast").unwrap();
                let slow_result = results.iter().find(|r| r.provider_id == "slow").unwrap();

                assert!(fast_result.time_ms >= 10);
                assert!(slow_result.time_ms >= 50);
                assert!(fast_result.time_ms < slow_result.time_ms);
            }

            #[tokio::test]
            async fn test_parallel_evaluator_complex_scoring() {
                let providers = vec![
                    (
                        "provider1".to_string(),
                        Box::new(MockLLMProvider::new("good response")) as Box<dyn LLMProvider>,
                    ),
                    (
                        "provider2".to_string(),
                        Box::new(MockLLMProvider::new("bad response")) as Box<dyn LLMProvider>,
                    ),
                ];
                let evaluator = ParallelEvaluator::new(providers)
                    .scoring(|response| response.len() as f32)
                    .scoring(|response| if response.contains("good") { 10.0 } else { 0.0 });

                let messages = vec![ChatMessage {
                    role: ChatRole::User,
                    message_type: MessageType::Text,
                    content: "test message".to_string(),
                }];

                let results = evaluator.evaluate_chat_parallel(&messages).await.unwrap();
                assert_eq!(results.len(), 2);

                let good_result = results.iter().find(|r| r.text == "good response").unwrap();
                let bad_result = results.iter().find(|r| r.text == "bad response").unwrap();

                assert_eq!(good_result.score, 23.0); // 13 (len) + 10 (contains "good")
                assert_eq!(bad_result.score, 12.0); // 12 (len) + 0 (no "good")
            }

            #[tokio::test]
            async fn test_parallel_evaluator_empty_providers() {
                let evaluator = ParallelEvaluator::new(vec![]);

                let messages = vec![ChatMessage {
                    role: ChatRole::User,
                    message_type: MessageType::Text,
                    content: "test message".to_string(),
                }];

                let results = evaluator.evaluate_chat_parallel(&messages).await.unwrap();
                assert_eq!(results.len(), 0);
            }

            #[tokio::test]
            async fn test_parallel_evaluator_all_providers_fail() {
                let providers = vec![
                    (
                        "failing1".to_string(),
                        Box::new(MockLLMProvider::with_failure("fail1")) as Box<dyn LLMProvider>,
                    ),
                    (
                        "failing2".to_string(),
                        Box::new(MockLLMProvider::with_failure("fail2")) as Box<dyn LLMProvider>,
                    ),
                ];
                let evaluator = ParallelEvaluator::new(providers);

                let messages = vec![ChatMessage {
                    role: ChatRole::User,
                    message_type: MessageType::Text,
                    content: "test message".to_string(),
                }];

                let results = evaluator.evaluate_chat_parallel(&messages).await.unwrap();
                assert_eq!(results.len(), 0);
            }

            #[test]
            fn test_parallel_evaluator_provider_id_preservation() {
                let providers = vec![
                    (
                        "custom_provider_1".to_string(),
                        Box::new(MockLLMProvider::new("response1")) as Box<dyn LLMProvider>,
                    ),
                    (
                        "another_provider".to_string(),
                        Box::new(MockLLMProvider::new("response2")) as Box<dyn LLMProvider>,
                    ),
                ];
                let evaluator = ParallelEvaluator::new(providers);

                assert_eq!(evaluator.providers.len(), 2);
                assert_eq!(evaluator.providers[0].0, "custom_provider_1");
                assert_eq!(evaluator.providers[1].0, "another_provider");
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
