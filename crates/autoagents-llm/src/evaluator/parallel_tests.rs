#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::{
        LLMProvider, ToolCall,
        chat::{
            ChatMessage, ChatProvider, ChatResponse, ChatRole, FunctionTool, MessageType, Tool,
        },
        completion::{CompletionProvider, CompletionRequest, CompletionResponse},
        embedding::EmbeddingProvider,
        error::LLMError,
        models::{ModelListRequest, ModelListResponse, ModelsProvider},
    };
    use async_trait::async_trait;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Mock LLM provider for testing
    #[derive(Debug)]
    struct MockLLMProvider {
        id: String,
        response_text: String,
        should_fail: bool,
        delay_ms: u64,
        call_count: Arc<AtomicUsize>,
    }

    impl MockLLMProvider {
        fn new(id: String, response_text: String) -> Self {
            Self {
                id,
                response_text,
                should_fail: false,
                delay_ms: 0,
                call_count: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn with_failure(id: String) -> Self {
            Self {
                id: id.clone(),
                response_text: String::new(),
                should_fail: true,
                delay_ms: 0,
                call_count: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn with_delay(mut self, delay_ms: u64) -> Self {
            self.delay_ms = delay_ms;
            self
        }
    }

    // Mock chat response
    #[derive(Debug)]
    struct MockChatResponse {
        text: String,
    }

    impl ChatResponse for MockChatResponse {
        fn text(&self) -> Option<String> {
            Some(self.text.clone())
        }

        fn tool_calls(&self) -> Option<Vec<ToolCall>> {
            None
        }
    }

    impl std::fmt::Display for MockChatResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.text)
        }
    }

    #[async_trait]
    impl ChatProvider for MockLLMProvider {
        async fn chat(
            &self,
            messages: &[ChatMessage],
            json_schema: Option<crate::chat::StructuredOutputFormat>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            self.chat_with_tools(messages, None, json_schema).await
        }

        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<crate::chat::StructuredOutputFormat>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            self.call_count.fetch_add(1, Ordering::SeqCst);

            if self.delay_ms > 0 {
                tokio::time::sleep(tokio::time::Duration::from_millis(self.delay_ms)).await;
            }

            if self.should_fail {
                return Err(LLMError::ProviderError(format!(
                    "Mock provider {} failed",
                    self.id
                )));
            }

            Ok(Box::new(MockChatResponse {
                text: self.response_text.clone(),
            }))
        }
    }

    #[async_trait]
    impl CompletionProvider for MockLLMProvider {
        async fn complete(
            &self,
            _req: &CompletionRequest,
            _json_schema: Option<crate::chat::StructuredOutputFormat>,
        ) -> Result<CompletionResponse, LLMError> {
            self.call_count.fetch_add(1, Ordering::SeqCst);

            if self.delay_ms > 0 {
                tokio::time::sleep(tokio::time::Duration::from_millis(self.delay_ms)).await;
            }

            if self.should_fail {
                return Err(LLMError::ProviderError(format!(
                    "Mock provider {} failed",
                    self.id
                )));
            }

            Ok(CompletionResponse {
                text: self.response_text.clone(),
            })
        }
    }

    #[async_trait]
    impl EmbeddingProvider for MockLLMProvider {
        async fn embed(&self, _texts: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
            Err(LLMError::ProviderError(
                "Embedding not supported".to_string(),
            ))
        }
    }

    #[async_trait]
    impl ModelsProvider for MockLLMProvider {
        async fn list_models(
            &self,
            _request: Option<&ModelListRequest>,
        ) -> Result<Box<dyn ModelListResponse>, LLMError> {
            Err(LLMError::ProviderError(
                "List Models not supported".to_string(),
            ))
        }
    }

    impl LLMProvider for MockLLMProvider {}

    #[tokio::test]
    async fn test_parallel_evaluator_new() {
        let providers: Vec<(String, Box<dyn LLMProvider>)> = vec![
            (
                "provider1".to_string(),
                Box::new(MockLLMProvider::new(
                    "provider1".to_string(),
                    "Response 1".to_string(),
                )),
            ),
            (
                "provider2".to_string(),
                Box::new(MockLLMProvider::new(
                    "provider2".to_string(),
                    "Response 2".to_string(),
                )),
            ),
        ];

        let _evaluator = ParallelEvaluator::new(providers);
        // We can't access include_timing directly as it's private,
        // but we can test the default behavior indirectly
    }

    #[tokio::test]
    async fn test_parallel_evaluator_with_scoring() {
        let providers: Vec<(String, Box<dyn LLMProvider>)> = vec![(
            "provider1".to_string(),
            Box::new(MockLLMProvider::new(
                "provider1".to_string(),
                "Short".to_string(),
            )),
        )];

        let evaluator = ParallelEvaluator::new(providers)
            .scoring(|text| text.len() as f32)
            .scoring(|text| if text.contains("Short") { 10.0 } else { 0.0 });

        let messages = vec![ChatMessage::user().content("Test").build()];
        let results = evaluator.evaluate_chat_parallel(&messages).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].score, 15.0); // 5 (length) + 10 (contains "Short")
    }

    #[tokio::test]
    async fn test_parallel_evaluator_include_timing() {
        let providers: Vec<(String, Box<dyn LLMProvider>)> = vec![(
            "provider1".to_string(),
            Box::new(
                MockLLMProvider::new("provider1".to_string(), "Response".to_string())
                    .with_delay(50),
            ),
        )];

        let evaluator = ParallelEvaluator::new(providers).include_timing(true);

        let messages = vec![ChatMessage::user().content("Test").build()];
        let results = evaluator.evaluate_chat_parallel(&messages).await.unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].time_ms >= 50);
    }

    #[tokio::test]
    async fn test_evaluate_chat_parallel_multiple_providers() {
        let providers: Vec<(String, Box<dyn LLMProvider>)> = vec![
            (
                "provider1".to_string(),
                Box::new(MockLLMProvider::new(
                    "provider1".to_string(),
                    "Response from provider 1".to_string(),
                )),
            ),
            (
                "provider2".to_string(),
                Box::new(MockLLMProvider::new(
                    "provider2".to_string(),
                    "Response from provider 2".to_string(),
                )),
            ),
            (
                "provider3".to_string(),
                Box::new(MockLLMProvider::new(
                    "provider3".to_string(),
                    "Response from provider 3".to_string(),
                )),
            ),
        ];

        let evaluator = ParallelEvaluator::new(providers);
        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                message_type: MessageType::Text,
                content: "You are a helpful assistant".to_string(),
            },
            ChatMessage::user().content("Hello").build(),
        ];

        let results = evaluator.evaluate_chat_parallel(&messages).await.unwrap();

        assert_eq!(results.len(), 3);
        assert!(results.iter().any(|r| r.provider_id == "provider1"));
        assert!(results.iter().any(|r| r.provider_id == "provider2"));
        assert!(results.iter().any(|r| r.provider_id == "provider3"));
        assert!(results.iter().any(|r| r.text == "Response from provider 1"));
        assert!(results.iter().any(|r| r.text == "Response from provider 2"));
        assert!(results.iter().any(|r| r.text == "Response from provider 3"));
    }

    #[tokio::test]
    async fn test_evaluate_chat_parallel_with_failures() {
        let providers: Vec<(String, Box<dyn LLMProvider>)> = vec![
            (
                "provider1".to_string(),
                Box::new(MockLLMProvider::new(
                    "provider1".to_string(),
                    "Success".to_string(),
                )),
            ),
            (
                "provider2".to_string(),
                Box::new(MockLLMProvider::with_failure("provider2".to_string())),
            ),
            (
                "provider3".to_string(),
                Box::new(MockLLMProvider::new(
                    "provider3".to_string(),
                    "Also success".to_string(),
                )),
            ),
        ];

        let evaluator = ParallelEvaluator::new(providers);
        let messages = vec![ChatMessage::user().content("Test").build()];

        let results = evaluator.evaluate_chat_parallel(&messages).await.unwrap();

        // Should only have 2 results (provider2 failed)
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|r| r.provider_id == "provider1"));
        assert!(results.iter().any(|r| r.provider_id == "provider3"));
        assert!(!results.iter().any(|r| r.provider_id == "provider2"));
    }

    #[tokio::test]
    async fn test_evaluate_chat_with_tools_parallel() {
        let providers: Vec<(String, Box<dyn LLMProvider>)> = vec![
            (
                "provider1".to_string(),
                Box::new(MockLLMProvider::new(
                    "provider1".to_string(),
                    "Tool response 1".to_string(),
                )),
            ),
            (
                "provider2".to_string(),
                Box::new(MockLLMProvider::new(
                    "provider2".to_string(),
                    "Tool response 2".to_string(),
                )),
            ),
        ];

        let evaluator = ParallelEvaluator::new(providers);
        let messages = vec![ChatMessage::user().content("Use a tool").build()];

        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: FunctionTool {
                name: "test_tool".to_string(),
                description: "A test tool".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "param": {"type": "string"}
                    }
                }),
            },
        }];

        let results = evaluator
            .evaluate_chat_with_tools_parallel(&messages, Some(&tools))
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].text.contains("Tool response"));
        assert!(results[1].text.contains("Tool response"));
    }

    #[tokio::test]
    async fn test_evaluate_completion_parallel() {
        let providers: Vec<(String, Box<dyn LLMProvider>)> = vec![
            (
                "provider1".to_string(),
                Box::new(MockLLMProvider::new(
                    "provider1".to_string(),
                    "Completion 1".to_string(),
                )),
            ),
            (
                "provider2".to_string(),
                Box::new(MockLLMProvider::new(
                    "provider2".to_string(),
                    "Completion 2 is longer".to_string(),
                )),
            ),
        ];

        let evaluator = ParallelEvaluator::new(providers).scoring(|text| text.len() as f32);

        let request = CompletionRequest {
            prompt: "Complete this".to_string(),
            max_tokens: Some(100),
            temperature: Some(0.7),
        };

        let results = evaluator
            .evaluate_completion_parallel(&request)
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].text == "Completion 1" || results[0].text == "Completion 2 is longer");
        assert!(results[1].text == "Completion 1" || results[1].text == "Completion 2 is longer");

        // Check scoring worked
        let longer_result = results
            .iter()
            .find(|r| r.text == "Completion 2 is longer")
            .unwrap();
        let shorter_result = results.iter().find(|r| r.text == "Completion 1").unwrap();
        assert!(longer_result.score > shorter_result.score);
    }

    #[tokio::test]
    async fn test_best_response() {
        let providers: Vec<(String, Box<dyn LLMProvider>)> = vec![
            (
                "provider1".to_string(),
                Box::new(MockLLMProvider::new(
                    "provider1".to_string(),
                    "Short".to_string(),
                )),
            ),
            (
                "provider2".to_string(),
                Box::new(MockLLMProvider::new(
                    "provider2".to_string(),
                    "This is a much longer response".to_string(),
                )),
            ),
            (
                "provider3".to_string(),
                Box::new(MockLLMProvider::new(
                    "provider3".to_string(),
                    "Medium response".to_string(),
                )),
            ),
        ];

        let evaluator = ParallelEvaluator::new(providers).scoring(|text| text.len() as f32);

        let messages = vec![ChatMessage::user().content("Test").build()];
        let results = evaluator.evaluate_chat_parallel(&messages).await.unwrap();

        let best = evaluator.best_response(&results).unwrap();
        assert_eq!(best.text, "This is a much longer response");
        assert_eq!(best.provider_id, "provider2");
    }

    #[tokio::test]
    async fn test_best_response_empty_results() {
        let providers: Vec<(String, Box<dyn LLMProvider>)> = vec![];
        let evaluator = ParallelEvaluator::new(providers);

        let results: Vec<ParallelEvalResult> = vec![];
        let best = evaluator.best_response(&results);
        assert!(best.is_none());
    }

    #[tokio::test]
    async fn test_best_response_with_multiple_scoring_functions() {
        let providers: Vec<(String, Box<dyn LLMProvider>)> = vec![
            (
                "provider1".to_string(),
                Box::new(MockLLMProvider::new(
                    "provider1".to_string(),
                    "quality response".to_string(),
                )),
            ),
            (
                "provider2".to_string(),
                Box::new(MockLLMProvider::new(
                    "provider2".to_string(),
                    "poor".to_string(),
                )),
            ),
        ];

        let evaluator = ParallelEvaluator::new(providers)
            .scoring(|text| text.len() as f32)
            .scoring(|text| if text.contains("quality") { 100.0 } else { 0.0 });

        let messages = vec![ChatMessage::user().content("Test").build()];
        let results = evaluator.evaluate_chat_parallel(&messages).await.unwrap();

        let best = evaluator.best_response(&results).unwrap();
        assert_eq!(best.text, "quality response");
        assert_eq!(best.provider_id, "provider1");
    }

    #[tokio::test]
    async fn test_compute_score_no_scoring_functions() {
        let providers: Vec<(String, Box<dyn LLMProvider>)> = vec![(
            "provider1".to_string(),
            Box::new(MockLLMProvider::new(
                "provider1".to_string(),
                "Response".to_string(),
            )),
        )];

        let evaluator = ParallelEvaluator::new(providers);
        let messages = vec![ChatMessage::user().content("Test").build()];
        let results = evaluator.evaluate_chat_parallel(&messages).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].score, 0.0);
    }

    #[tokio::test]
    async fn test_parallel_execution_timing() {
        let providers: Vec<(String, Box<dyn LLMProvider>)> = vec![
            (
                "provider1".to_string(),
                Box::new(
                    MockLLMProvider::new("provider1".to_string(), "Response 1".to_string())
                        .with_delay(100),
                ),
            ),
            (
                "provider2".to_string(),
                Box::new(
                    MockLLMProvider::new("provider2".to_string(), "Response 2".to_string())
                        .with_delay(100),
                ),
            ),
            (
                "provider3".to_string(),
                Box::new(
                    MockLLMProvider::new("provider3".to_string(), "Response 3".to_string())
                        .with_delay(100),
                ),
            ),
        ];

        let evaluator = ParallelEvaluator::new(providers);
        let messages = vec![ChatMessage::user().content("Test").build()];

        let start = std::time::Instant::now();
        let results = evaluator.evaluate_chat_parallel(&messages).await.unwrap();
        let elapsed = start.elapsed().as_millis();

        assert_eq!(results.len(), 3);
        // If running in parallel, should take around 100ms, not 300ms
        assert!(
            elapsed < 200,
            "Expected parallel execution to take less than 200ms, took {elapsed}ms"
        );
    }

    #[tokio::test]
    async fn test_all_providers_fail() {
        let providers: Vec<(String, Box<dyn LLMProvider>)> = vec![
            (
                "provider1".to_string(),
                Box::new(MockLLMProvider::with_failure("provider1".to_string())),
            ),
            (
                "provider2".to_string(),
                Box::new(MockLLMProvider::with_failure("provider2".to_string())),
            ),
        ];

        let evaluator = ParallelEvaluator::new(providers);
        let messages = vec![ChatMessage::user().content("Test").build()];

        let results = evaluator.evaluate_chat_parallel(&messages).await.unwrap();
        assert_eq!(results.len(), 0);

        let best = evaluator.best_response(&results);
        assert!(best.is_none());
    }

    #[tokio::test]
    async fn test_evaluate_completion_with_suffix() {
        let providers: Vec<(String, Box<dyn LLMProvider>)> = vec![(
            "provider1".to_string(),
            Box::new(MockLLMProvider::new(
                "provider1".to_string(),
                "Middle text".to_string(),
            )),
        )];

        let evaluator = ParallelEvaluator::new(providers);
        let request = CompletionRequest {
            prompt: "Start".to_string(),
            max_tokens: Some(50),
            temperature: Some(0.5),
        };

        let results = evaluator
            .evaluate_completion_parallel(&request)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "Middle text");
    }

    #[tokio::test]
    async fn test_provider_call_count() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let provider = MockLLMProvider {
            id: "provider1".to_string(),
            response_text: "Response".to_string(),
            should_fail: false,
            delay_ms: 0,
            call_count: call_count.clone(),
        };

        let providers: Vec<(String, Box<dyn LLMProvider>)> =
            vec![("provider1".to_string(), Box::new(provider))];

        let evaluator = ParallelEvaluator::new(providers);

        // Call chat
        let messages = vec![ChatMessage::user().content("Test").build()];
        let _ = evaluator.evaluate_chat_parallel(&messages).await.unwrap();
        assert_eq!(call_count.load(Ordering::SeqCst), 1);

        // Call again
        let _ = evaluator.evaluate_chat_parallel(&messages).await.unwrap();
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_parallel_eval_result_debug() {
        let result = ParallelEvalResult {
            text: "Test response".to_string(),
            score: 42.5,
            time_ms: 123,
            provider_id: "test_provider".to_string(),
        };

        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("ParallelEvalResult"));
        assert!(debug_str.contains("Test response"));
        assert!(debug_str.contains("42.5"));
        assert!(debug_str.contains("123"));
        assert!(debug_str.contains("test_provider"));
    }
}
