use std::{fmt, pin::Pin, sync::Arc};

use async_trait::async_trait;
use autoagents_llm::{
    LLMProvider,
    chat::{
        ChatMessage, ChatProvider, ChatResponse, StreamChunk, StreamResponse,
        StructuredOutputFormat, Tool,
    },
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::{ModelListRequest, ModelListResponse, ModelsProvider},
};
use futures::Stream;

use crate::{
    engine::GuardrailsEngine,
    guard::{
        ChatGuardInput, ChatGuardOutput, CompletionGuardInput, CompletionGuardOutput, GuardContext,
        GuardOperation, GuardedInput, GuardedOutput, WebSearchGuardInput,
    },
    stream::{StructGuardedStream, TextGuardedStream, ToolGuardedStream},
};

pub(crate) struct GuardedProvider {
    inner: Arc<dyn LLMProvider>,
    engine: Arc<GuardrailsEngine>,
}

impl GuardedProvider {
    pub(crate) fn new(inner: Arc<dyn LLMProvider>, engine: Arc<GuardrailsEngine>) -> Self {
        Self { inner, engine }
    }

    async fn process_chat_output(
        &self,
        response: Box<dyn ChatResponse>,
        context: &GuardContext,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        if !self.engine.has_output_guards() {
            return Ok(response);
        }

        let mut output = GuardedOutput::Chat(ChatGuardOutput {
            text: response.text(),
            tool_calls: response.tool_calls(),
            thinking: response.thinking(),
            usage: response.usage(),
        });

        self.engine.evaluate_output(&mut output, context).await?;

        match output {
            GuardedOutput::Chat(value) => Ok(Box::new(MaterializedChatResponse {
                text: value.text,
                tool_calls: value.tool_calls,
                thinking: value.thinking,
                usage: value.usage,
            })),
            GuardedOutput::Completion(_) => Err(LLMError::ProviderError(
                "unexpected completion output for chat response".to_string(),
            )),
        }
    }

    async fn process_completion_output(
        &self,
        response: CompletionResponse,
        context: &GuardContext,
    ) -> Result<CompletionResponse, LLMError> {
        if !self.engine.has_output_guards() {
            return Ok(response);
        }

        let mut output = GuardedOutput::Completion(CompletionGuardOutput {
            text: response.text,
        });
        self.engine.evaluate_output(&mut output, context).await?;

        match output {
            GuardedOutput::Completion(value) => Ok(CompletionResponse { text: value.text }),
            GuardedOutput::Chat(_) => Err(LLMError::ProviderError(
                "unexpected chat output for completion response".to_string(),
            )),
        }
    }
}

#[derive(Debug, Clone)]
struct MaterializedChatResponse {
    text: Option<String>,
    tool_calls: Option<Vec<autoagents_llm::ToolCall>>,
    thinking: Option<String>,
    usage: Option<autoagents_llm::chat::Usage>,
}

impl ChatResponse for MaterializedChatResponse {
    fn text(&self) -> Option<String> {
        self.text.clone()
    }

    fn tool_calls(&self) -> Option<Vec<autoagents_llm::ToolCall>> {
        self.tool_calls.clone()
    }

    fn thinking(&self) -> Option<String> {
        self.thinking.clone()
    }

    fn usage(&self) -> Option<autoagents_llm::chat::Usage> {
        self.usage.clone()
    }
}

impl fmt::Display for MaterializedChatResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text.as_deref().unwrap_or_default())
    }
}

#[async_trait]
impl ChatProvider for GuardedProvider {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let operation = if tools.is_some() {
            GuardOperation::ChatWithTools
        } else {
            GuardOperation::Chat
        };
        let context = GuardContext::new(operation);

        let response = if self.engine.has_input_guards() {
            let mut input = GuardedInput::Chat(ChatGuardInput {
                messages: messages.to_vec(),
                tools: tools.map(|value| value.to_vec()),
                json_schema,
            });
            self.engine.evaluate_input(&mut input, &context).await?;

            let GuardedInput::Chat(chat) = input else {
                return Err(LLMError::ProviderError(
                    "unexpected input variant for chat".to_string(),
                ));
            };

            self.inner
                .chat_with_tools(&chat.messages, chat.tools.as_deref(), chat.json_schema)
                .await?
        } else {
            self.inner
                .chat_with_tools(messages, tools, json_schema)
                .await?
        };

        self.process_chat_output(response, &context).await
    }

    async fn chat_with_web_search(&self, input: String) -> Result<Box<dyn ChatResponse>, LLMError> {
        let context = GuardContext::new(GuardOperation::ChatWithWebSearch);

        let response = if self.engine.has_input_guards() {
            let mut guarded = GuardedInput::WebSearch(WebSearchGuardInput { input });
            self.engine.evaluate_input(&mut guarded, &context).await?;

            let GuardedInput::WebSearch(web) = guarded else {
                return Err(LLMError::ProviderError(
                    "unexpected input variant for web search".to_string(),
                ));
            };

            self.inner.chat_with_web_search(web.input).await?
        } else {
            self.inner.chat_with_web_search(input).await?
        };

        self.process_chat_output(response, &context).await
    }

    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError> {
        let context = GuardContext::new(GuardOperation::ChatStream);

        let stream = if self.engine.has_input_guards() {
            let mut input = GuardedInput::Chat(ChatGuardInput {
                messages: messages.to_vec(),
                tools: None,
                json_schema,
            });
            self.engine.evaluate_input(&mut input, &context).await?;

            let GuardedInput::Chat(chat) = input else {
                return Err(LLMError::ProviderError(
                    "unexpected input variant for chat_stream".to_string(),
                ));
            };

            self.inner
                .chat_stream(&chat.messages, chat.json_schema)
                .await?
        } else {
            self.inner.chat_stream(messages, json_schema).await?
        };

        if !self.engine.has_output_guards() {
            return Ok(stream);
        }

        Ok(Box::pin(TextGuardedStream::new(
            stream,
            self.engine.clone(),
            context,
        )))
    }

    async fn chat_stream_struct(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>, LLMError>
    {
        let context = GuardContext::new(GuardOperation::ChatStreamStruct);

        let stream = if self.engine.has_input_guards() {
            let mut input = GuardedInput::Chat(ChatGuardInput {
                messages: messages.to_vec(),
                tools: tools.map(|value| value.to_vec()),
                json_schema,
            });
            self.engine.evaluate_input(&mut input, &context).await?;

            let GuardedInput::Chat(chat) = input else {
                return Err(LLMError::ProviderError(
                    "unexpected input variant for chat_stream_struct".to_string(),
                ));
            };

            self.inner
                .chat_stream_struct(&chat.messages, chat.tools.as_deref(), chat.json_schema)
                .await?
        } else {
            self.inner
                .chat_stream_struct(messages, tools, json_schema)
                .await?
        };

        if !self.engine.has_output_guards() {
            return Ok(stream);
        }

        Ok(Box::pin(StructGuardedStream::new(
            stream,
            self.engine.clone(),
            context,
        )))
    }

    async fn chat_stream_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError> {
        let context = GuardContext::new(GuardOperation::ChatStreamWithTools);

        let stream = if self.engine.has_input_guards() {
            let mut input = GuardedInput::Chat(ChatGuardInput {
                messages: messages.to_vec(),
                tools: tools.map(|value| value.to_vec()),
                json_schema,
            });
            self.engine.evaluate_input(&mut input, &context).await?;

            let GuardedInput::Chat(chat) = input else {
                return Err(LLMError::ProviderError(
                    "unexpected input variant for chat_stream_with_tools".to_string(),
                ));
            };

            self.inner
                .chat_stream_with_tools(&chat.messages, chat.tools.as_deref(), chat.json_schema)
                .await?
        } else {
            self.inner
                .chat_stream_with_tools(messages, tools, json_schema)
                .await?
        };

        if !self.engine.has_output_guards() {
            return Ok(stream);
        }

        Ok(Box::pin(ToolGuardedStream::new(
            stream,
            self.engine.clone(),
            context,
        )))
    }
}

#[async_trait]
impl CompletionProvider for GuardedProvider {
    async fn complete(
        &self,
        req: &CompletionRequest,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        let context = GuardContext::new(GuardOperation::Complete);

        let response = if self.engine.has_input_guards() {
            let mut input = GuardedInput::Completion(CompletionGuardInput {
                request: req.clone(),
                json_schema,
            });

            self.engine.evaluate_input(&mut input, &context).await?;

            let GuardedInput::Completion(completion) = input else {
                return Err(LLMError::ProviderError(
                    "unexpected input variant for completion".to_string(),
                ));
            };

            self.inner
                .complete(&completion.request, completion.json_schema)
                .await?
        } else {
            self.inner.complete(req, json_schema).await?
        };

        self.process_completion_output(response, &context).await
    }
}

#[async_trait]
impl EmbeddingProvider for GuardedProvider {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        self.inner.embed(input).await
    }
}

#[async_trait]
impl ModelsProvider for GuardedProvider {
    async fn list_models(
        &self,
        request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        self.inner.list_models(request).await
    }
}

impl LLMProvider for GuardedProvider {}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    };

    use async_trait::async_trait;
    use autoagents_llm::{
        HasConfig, NoConfig,
        chat::{
            ChatMessage, ChatProvider, ChatResponse, StreamChunk, StreamResponse,
            StructuredOutputFormat, Tool,
        },
    };
    use futures::{Stream, StreamExt, stream};

    use crate::{
        engine::Guardrails,
        guard::{GuardDecision, GuardError, GuardViolation, GuardedInput, GuardedOutput},
        policy::{EnforcementPolicy, GuardCategory, GuardSeverity},
    };

    use super::*;

    #[derive(Debug, Clone)]
    struct MockChatResponse(String);

    impl ChatResponse for MockChatResponse {
        fn text(&self) -> Option<String> {
            Some(self.0.clone())
        }

        fn tool_calls(&self) -> Option<Vec<autoagents_llm::ToolCall>> {
            None
        }
    }

    impl std::fmt::Display for MockChatResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(&self.0)
        }
    }

    struct MockProvider {
        calls: Arc<AtomicU32>,
    }

    impl MockProvider {
        fn new() -> Self {
            Self {
                calls: Arc::new(AtomicU32::new(0)),
            }
        }
    }

    #[async_trait]
    impl ChatProvider for MockProvider {
        async fn chat_with_tools(
            &self,
            messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let prompt = messages
                .iter()
                .last()
                .map(|msg| msg.content.clone())
                .unwrap_or_default();
            Ok(Box::new(MockChatResponse(format!("echo:{prompt}"))))
        }

        async fn chat_stream(
            &self,
            messages: &[ChatMessage],
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
        {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let prompt = messages
                .iter()
                .last()
                .map(|msg| msg.content.clone())
                .unwrap_or_default();
            let chunks = vec![Ok("echo:".to_string()), Ok(prompt)];
            Ok(Box::pin(stream::iter(chunks)))
        }

        async fn chat_stream_struct(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>, LLMError>
        {
            Ok(Box::pin(stream::empty()))
        }

        async fn chat_stream_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError>
        {
            Ok(Box::pin(stream::empty()))
        }
    }

    #[async_trait]
    impl CompletionProvider for MockProvider {
        async fn complete(
            &self,
            req: &CompletionRequest,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<CompletionResponse, LLMError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(CompletionResponse {
                text: format!("echo:{}", req.prompt),
            })
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

    struct RejectOnWordInputGuard;

    #[async_trait]
    impl crate::InputGuard for RejectOnWordInputGuard {
        fn name(&self) -> &'static str {
            "reject-on-word-input"
        }

        async fn inspect(
            &self,
            input: &mut GuardedInput,
            _context: &crate::GuardContext,
        ) -> Result<GuardDecision, GuardError> {
            let GuardedInput::Chat(chat) = input else {
                return Ok(GuardDecision::Pass);
            };
            if chat.messages.iter().any(|m| m.content.contains("block-me")) {
                return Ok(GuardDecision::Reject(GuardViolation::new(
                    "block_word",
                    GuardCategory::PromptInjection,
                    GuardSeverity::High,
                    "blocked for test",
                )));
            }
            Ok(GuardDecision::Pass)
        }
    }

    struct RejectOnWordOutputGuard;

    #[async_trait]
    impl crate::OutputGuard for RejectOnWordOutputGuard {
        fn name(&self) -> &'static str {
            "reject-on-word-output"
        }

        async fn inspect(
            &self,
            output: &mut GuardedOutput,
            _context: &crate::GuardContext,
        ) -> Result<GuardDecision, GuardError> {
            let text = match output {
                GuardedOutput::Chat(chat) => chat.text.clone().unwrap_or_default(),
                GuardedOutput::Completion(completion) => completion.text.clone(),
            };
            if text.contains("unsafe") {
                return Ok(GuardDecision::Reject(GuardViolation::new(
                    "unsafe_word",
                    GuardCategory::Toxicity,
                    GuardSeverity::High,
                    "unsafe output for test",
                )));
            }
            Ok(GuardDecision::Pass)
        }
    }

    #[tokio::test]
    async fn input_block_policy_stops_upstream_call() {
        let base = Arc::new(MockProvider::new());
        let calls = base.calls.clone();

        let guardrails = Guardrails::builder()
            .input_guard(RejectOnWordInputGuard)
            .enforcement_policy(EnforcementPolicy::Block)
            .build();

        let guarded = guardrails.wrap(base as Arc<dyn LLMProvider>);
        let messages = vec![ChatMessage::user().content("please block-me").build()];

        let err = guarded.chat(&messages, None).await.unwrap_err();
        assert!(err.to_string().contains("guardrail blocked input"));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn output_sanitize_policy_rewrites_completion() {
        let base = Arc::new(MockProvider::new());
        let guardrails = Guardrails::builder()
            .output_guard(RejectOnWordOutputGuard)
            .enforcement_policy(EnforcementPolicy::Sanitize)
            .build();

        let guarded = guardrails.wrap(base as Arc<dyn LLMProvider>);
        let response = guarded
            .complete(&CompletionRequest::new("unsafe"), None)
            .await
            .unwrap();

        assert_eq!(response.text, "[redacted by guardrails]");
    }

    #[tokio::test]
    async fn streaming_post_aggregate_emits_final_error_on_violation() {
        let base = Arc::new(MockProvider::new());
        let guardrails = Guardrails::builder()
            .output_guard(RejectOnWordOutputGuard)
            .enforcement_policy(EnforcementPolicy::Block)
            .build();

        let guarded = guardrails.wrap(base as Arc<dyn LLMProvider>);
        let messages = vec![ChatMessage::user().content("unsafe").build()];
        let mut stream = guarded.chat_stream(&messages, None).await.unwrap();

        assert_eq!(stream.next().await.unwrap().unwrap(), "echo:");
        assert_eq!(stream.next().await.unwrap().unwrap(), "unsafe");

        let err = stream.next().await.unwrap().unwrap_err();
        assert!(err.to_string().contains("guardrail blocked output"));
        assert!(stream.next().await.is_none());
    }
}
