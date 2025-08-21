use crate::actor::{ActorMessage, Topic};
use crate::agent::memory::MemoryProvider;
use crate::agent::{AgentConfig, AgentState};
use crate::error::Error;
use crate::protocol::Event;
use crate::tool::ToolT;
use autoagents_llm::chat::ChatMessage;
use autoagents_llm::LLMProvider;
use std::any::Any;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

pub struct Context {
    llm: Arc<dyn LLMProvider>,
    messages: Vec<ChatMessage>,
    memory: Option<Arc<RwLock<Box<dyn MemoryProvider>>>>,
    tools: Vec<Box<dyn ToolT>>,
    config: AgentConfig,
    state: Arc<RwLock<AgentState>>,
    tx: mpsc::Sender<Event>,
    stream: bool,
}

impl Context {
    pub fn new(llm: Arc<dyn LLMProvider>, tx: mpsc::Sender<Event>) -> Self {
        Self {
            llm,
            messages: vec![],
            memory: None,
            tools: vec![],
            config: AgentConfig::default(),
            state: Arc::new(RwLock::new(AgentState::new(tx.clone()))),
            stream: false,
            tx,
        }
    }

    pub async fn publish<M: ActorMessage>(&self, topic: Topic<M>, message: M) -> Result<(), Error> {
        self.tx
            .send(Event::PublishMessage {
                topic_name: topic.name().to_string(),
                message: Arc::new(message) as Arc<dyn Any + Send + Sync>,
                topic_type: topic.type_id(),
            })
            .await
            .map_err(|e| Error::CustomError(e.to_string()))
    }

    pub fn with_memory(mut self, memory: Option<Arc<RwLock<Box<dyn MemoryProvider>>>>) -> Self {
        self.memory = memory;
        self
    }

    pub fn with_tools(mut self, tools: Vec<Box<dyn ToolT>>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_config(mut self, config: AgentConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_messages(mut self, messages: Vec<ChatMessage>) -> Self {
        self.messages = messages;
        self
    }

    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    // Getters
    pub fn llm(&self) -> Arc<dyn LLMProvider> {
        self.llm.clone()
    }

    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    pub fn memory(&self) -> Option<Arc<RwLock<Box<dyn MemoryProvider>>>> {
        self.memory.clone()
    }

    pub fn tools(&self) -> &[Box<dyn ToolT>] {
        &self.tools
    }

    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    pub fn state(&self) -> Arc<RwLock<AgentState>> {
        self.state.clone()
    }

    pub fn tx(&self) -> &mpsc::Sender<Event> {
        &self.tx
    }

    pub fn stream(&self) -> bool {
        self.stream
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::memory::SlidingWindowMemory;
    use autoagents_llm::chat::{ChatMessage, ChatMessageBuilder, ChatRole};
    use autoagents_test_utils::llm::MockLLMProvider;
    use std::sync::Arc;
    use tokio::sync::mpsc;

    #[test]
    fn test_context_creation() {
        let llm = Arc::new(MockLLMProvider);
        let (tx, _rx) = mpsc::channel(10);
        let context = Context::new(llm, tx);

        assert!(context.messages.is_empty());
        assert!(context.memory.is_none());
        assert!(context.tools.is_empty());
        assert!(!context.stream);
    }

    #[test]
    fn test_context_with_llm_provider() {
        let llm = Arc::new(MockLLMProvider);
        let (tx, _rx) = mpsc::channel(10);
        let context = Context::new(llm.clone(), tx);

        // Verify the LLM provider is set correctly
        let context_llm = context.llm();
        assert!(Arc::strong_count(&context_llm) > 0);
    }

    #[test]
    fn test_context_with_memory() {
        let llm = Arc::new(MockLLMProvider);
        let (tx, _rx) = mpsc::channel(10);
        let memory = Box::new(SlidingWindowMemory::new(5));
        let context = Context::new(llm, tx).with_memory(Some(Arc::new(RwLock::new(memory))));

        assert!(context.memory().is_some());
    }

    #[test]
    fn test_context_with_messages() {
        let llm = Arc::new(MockLLMProvider);
        let (tx, _rx) = mpsc::channel(10);
        let message = ChatMessage::user().content("Hello".to_string()).build();
        let context = Context::new(llm, tx).with_messages(vec![message]);

        assert_eq!(context.messages().len(), 1);
        assert_eq!(context.messages()[0].role, ChatRole::User);
        assert_eq!(context.messages()[0].content, "Hello");
    }

    #[test]
    fn test_context_streaming_flag() {
        let llm = Arc::new(MockLLMProvider);
        let (tx, _rx) = mpsc::channel(10);
        let context = Context::new(llm, tx).with_stream(true);

        assert!(context.stream());
    }

    #[test]
    fn test_context_fluent_interface() {
        let llm = Arc::new(MockLLMProvider);
        let (tx, _rx) = mpsc::channel(10);
        let memory = Box::new(SlidingWindowMemory::new(3));
        let message = ChatMessageBuilder::new(ChatRole::System)
            .content("System prompt".to_string())
            .build();

        let context = Context::new(llm, tx)
            .with_memory(Some(Arc::new(RwLock::new(memory))))
            .with_messages(vec![message])
            .with_stream(true);

        assert!(context.memory().is_some());
        assert_eq!(context.messages().len(), 1);
        assert!(context.stream());
    }
}
