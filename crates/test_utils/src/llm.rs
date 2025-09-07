use autoagents::async_trait;
use autoagents_llm::{
    chat::{ChatMessage, ChatProvider, ChatResponse, StructuredOutputFormat},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
    LLMProvider, ToolCall,
};

// Mock LLM Provider
pub struct MockLLMProvider;

#[async_trait]
impl ChatProvider for MockLLMProvider {
    async fn chat(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[autoagents_llm::chat::Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        Ok(Box::new(MockChatResponse {
            text: Some("Mock response".to_string()),
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
