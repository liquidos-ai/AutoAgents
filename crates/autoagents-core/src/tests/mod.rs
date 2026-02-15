pub(crate) mod actor_integration_tests;
pub(crate) mod agent_integration_tests;

use async_trait::async_trait;
use futures::Stream;
use futures::stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::pin::Pin;
use std::sync::Arc;

use crate::agent::AgentHooks;
use crate::agent::task::Task;
use crate::agent::{AgentDeriveT, AgentExecutor, AgentOutputT, Context, ExecutorConfig};
use crate::tool::ToolT;
use autoagents_llm::builder::LLMBackend;
use autoagents_llm::{
    LLMProvider, ToolCall,
    chat::{
        ChatMessage, ChatProvider, ChatResponse, StreamChunk, StreamResponse,
        StructuredOutputFormat, Tool, Usage,
    },
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::{
        ModelListRequest, ModelListResponse, ModelsProvider, StandardModelEntry,
        StandardModelListResponse, StandardModelListResponseInner,
    },
};

#[derive(Debug, thiserror::Error)]
pub(crate) enum TestError {
    #[error("Test error: {0}")]
    TestError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TestAgentOutput {
    pub(crate) result: String,
}

impl AgentOutputT for TestAgentOutput {
    fn output_schema() -> &'static str {
        r#"{"type":"object","properties":{"result":{"type":"string"}},"required":["result"]}"#
    }

    fn structured_output_format() -> Value {
        serde_json::json!({
            "name": "TestAgentOutput",
            "description": "Test agent output schema",
            "schema": {
                "type": "object",
                "properties": {
                    "result": {"type": "string"}
                },
                "required": ["result"]
            },
            "strict": true
        })
    }
}

impl From<TestAgentOutput> for Value {
    fn from(output: TestAgentOutput) -> Self {
        serde_json::to_value(output).unwrap_or(Value::Null)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MockAgentImpl {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) should_fail: bool,
}

impl MockAgentImpl {
    pub(crate) fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            should_fail: false,
        }
    }

    pub(crate) fn with_failure(mut self, should_fail: bool) -> Self {
        self.should_fail = should_fail;
        self
    }
}

#[async_trait]
impl AgentDeriveT for MockAgentImpl {
    type Output = TestAgentOutput;

    fn description(&self) -> &'static str {
        Box::leak(self.description.clone().into_boxed_str())
    }

    fn output_schema(&self) -> Option<Value> {
        Some(TestAgentOutput::structured_output_format())
    }

    fn name(&self) -> &'static str {
        Box::leak(self.name.clone().into_boxed_str())
    }

    fn tools(&self) -> Vec<Box<dyn ToolT>> {
        vec![]
    }
}

#[async_trait]
impl AgentExecutor for MockAgentImpl {
    type Output = TestAgentOutput;
    type Error = TestError;

    fn config(&self) -> ExecutorConfig {
        ExecutorConfig::default()
    }

    async fn execute(
        &self,
        task: &Task,
        _context: Arc<Context>,
    ) -> Result<Self::Output, Self::Error> {
        if self.should_fail {
            return Err(TestError::TestError("Mock execution failed".to_string()));
        }

        Ok(TestAgentOutput {
            result: format!("Processed: {}", task.prompt),
        })
    }

    async fn execute_stream(
        &self,
        _task: &Task,
        _context: Arc<Context>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Self::Output, Self::Error>> + Send>>, Self::Error>
    {
        unimplemented!()
    }
}

impl AgentHooks for MockAgentImpl {}

#[derive(Debug, Clone)]
pub(crate) struct StaticChatResponse {
    pub(crate) text: Option<String>,
    pub(crate) tool_calls: Option<Vec<ToolCall>>,
    pub(crate) usage: Option<Usage>,
    pub(crate) thinking: Option<String>,
}

impl std::fmt::Display for StaticChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(text) = &self.text {
            write!(f, "{text}")
        } else {
            write!(f, "")
        }
    }
}

impl ChatResponse for StaticChatResponse {
    fn text(&self) -> Option<String> {
        self.text.clone()
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.tool_calls.clone()
    }

    fn thinking(&self) -> Option<String> {
        self.thinking.clone()
    }

    fn usage(&self) -> Option<Usage> {
        self.usage.clone()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ConfigurableLLMProvider {
    pub(crate) chat_response: StaticChatResponse,
    pub(crate) stream_chunks: Vec<StreamChunk>,
    pub(crate) structured_stream: Vec<StreamResponse>,
    pub(crate) completion_response: CompletionResponse,
    pub(crate) embeddings: Vec<Vec<f32>>,
    pub(crate) models: Vec<String>,
}

impl Default for ConfigurableLLMProvider {
    fn default() -> Self {
        Self {
            chat_response: StaticChatResponse {
                text: Some("Mock response".to_string()),
                tool_calls: None,
                usage: None,
                thinking: None,
            },
            stream_chunks: Vec::new(),
            structured_stream: Vec::new(),
            completion_response: CompletionResponse {
                text: "Mock completion".to_string(),
            },
            embeddings: vec![vec![0.1, 0.2, 0.3]],
            models: vec!["test-model".to_string()],
        }
    }
}

#[async_trait]
impl ChatProvider for ConfigurableLLMProvider {
    async fn chat(
        &self,
        _messages: &[ChatMessage],
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        Ok(Box::new(self.chat_response.clone()))
    }

    async fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        Ok(Box::new(self.chat_response.clone()))
    }

    async fn chat_stream_struct(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>, LLMError>
    {
        let stream = stream::iter(self.structured_stream.clone().into_iter().map(Ok));
        Ok(Box::pin(stream))
    }

    async fn chat_stream_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError> {
        let stream = stream::iter(self.stream_chunks.clone().into_iter().map(Ok));
        Ok(Box::pin(stream))
    }
}

#[async_trait]
impl CompletionProvider for ConfigurableLLMProvider {
    async fn complete(
        &self,
        _req: &CompletionRequest,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        Ok(self.completion_response.clone())
    }
}

#[async_trait]
impl EmbeddingProvider for ConfigurableLLMProvider {
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Ok(self.embeddings.clone())
    }
}

#[async_trait]
impl ModelsProvider for ConfigurableLLMProvider {
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        let data = self
            .models
            .iter()
            .cloned()
            .map(|id| StandardModelEntry {
                id,
                created: None,
                extra: Value::Null,
            })
            .collect::<Vec<_>>();
        let response = StandardModelListResponse {
            inner: StandardModelListResponseInner { data },
            backend: LLMBackend::OpenAI,
        };
        Ok(Box::new(response))
    }
}

impl LLMProvider for ConfigurableLLMProvider {}

pub(crate) struct MockLLMProvider;

#[async_trait]
impl ChatProvider for MockLLMProvider {
    async fn chat(
        &self,
        _messages: &[ChatMessage],
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        Ok(Box::new(StaticChatResponse {
            text: Some("Mock response".to_string()),
            tool_calls: None,
            usage: None,
            thinking: None,
        }))
    }

    async fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        Ok(Box::new(StaticChatResponse {
            text: Some("Mock response".to_string()),
            tool_calls: None,
            usage: None,
            thinking: None,
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
        Ok(CompletionResponse {
            text: "Mock completion".to_string(),
        })
    }
}

#[async_trait]
impl EmbeddingProvider for MockLLMProvider {
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Ok(vec![vec![0.1, 0.2, 0.3]])
    }
}

#[async_trait]
impl ModelsProvider for MockLLMProvider {}

impl LLMProvider for MockLLMProvider {}
