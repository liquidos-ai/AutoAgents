#[cfg(not(target_arch = "wasm32"))]
use crate::actor::{ActorMessage, Topic};
use crate::agent::AgentConfig;
use crate::agent::memory::MemoryProvider;
use crate::agent::state::AgentState;
use crate::tool::{ToolT, to_llm_tool};
use autoagents_llm::LLMProvider;
use autoagents_llm::chat::{ChatMessage, Tool};
use autoagents_protocol::Event;
use std::any::Any;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::{Mutex, mpsc};

#[cfg(target_arch = "wasm32")]
use futures::channel::mpsc;
#[cfg(target_arch = "wasm32")]
use futures::lock::Mutex;

/// Execution context shared across an agent run.
///
/// Holds the configured LLM provider, accumulated chat messages, optional
/// memory and tools, agent configuration, ephemeral execution state, and
/// an optional event transmitter used by actor-based agents to emit protocol
/// `Event`s. Also carries a `stream` flag to indicate streaming mode.
pub struct Context {
    llm: Arc<dyn LLMProvider>,
    messages: Vec<ChatMessage>,
    memory: Option<Arc<Mutex<Box<dyn MemoryProvider>>>>,
    tools: Vec<Box<dyn ToolT>>,
    serialized_tools: Option<Arc<Vec<Tool>>>,
    config: AgentConfig,
    state: Arc<Mutex<AgentState>>,
    tx: Option<mpsc::Sender<Event>>,
    stream: bool,
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum ContextError {
    #[error("Tx value is None, Tx is only set for Actor agents")]
    EmptyTx,
    /// Error when sending events
    #[error("Failed to send event: {0}")]
    EventSendError(String),
}

impl Context {
    pub fn new(llm: Arc<dyn LLMProvider>, tx: Option<mpsc::Sender<Event>>) -> Self {
        Self {
            llm,
            messages: vec![],
            memory: None,
            tools: vec![],
            serialized_tools: None,
            config: AgentConfig::default(),
            state: Arc::new(Mutex::new(AgentState::new())),
            stream: false,
            tx,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn publish<M: ActorMessage>(
        &self,
        topic: Topic<M>,
        message: M,
    ) -> Result<(), ContextError> {
        self.tx
            .as_ref()
            .ok_or(ContextError::EmptyTx)?
            .send(Event::PublishMessage {
                topic_name: topic.name().to_string(),
                message: Arc::new(message) as Arc<dyn Any + Send + Sync>,
                topic_type: topic.type_id(),
            })
            .await
            .map_err(|e| ContextError::EventSendError(e.to_string()))
    }

    pub fn with_memory(mut self, memory: Option<Arc<Mutex<Box<dyn MemoryProvider>>>>) -> Self {
        self.memory = memory;
        self
    }

    pub fn with_tools(mut self, tools: Vec<Box<dyn ToolT>>) -> Self {
        if tools.is_empty() {
            self.serialized_tools = None;
        } else if self.serialized_tools.is_none() {
            let serialized = tools.iter().map(to_llm_tool).collect::<Vec<_>>();
            self.serialized_tools = Some(Arc::new(serialized));
        }
        self.tools = tools;
        self
    }

    pub fn with_serialized_tools(mut self, tools: Option<Arc<Vec<Tool>>>) -> Self {
        self.serialized_tools = tools;
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
    pub fn llm(&self) -> &Arc<dyn LLMProvider> {
        &self.llm
    }

    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    pub fn memory(&self) -> Option<Arc<Mutex<Box<dyn MemoryProvider>>>> {
        self.memory.clone()
    }

    pub fn tools(&self) -> &[Box<dyn ToolT>] {
        &self.tools
    }

    pub fn serialized_tools(&self) -> Option<Arc<Vec<Tool>>> {
        self.serialized_tools.clone()
    }

    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    pub fn state(&self) -> Arc<Mutex<AgentState>> {
        self.state.clone()
    }

    /// Get a clone of the event transmitter (only present for Actor agents).
    pub fn tx(&self) -> Result<mpsc::Sender<Event>, ContextError> {
        Ok(self.tx.as_ref().ok_or(ContextError::EmptyTx)?.clone())
    }

    pub fn stream(&self) -> bool {
        self.stream
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::memory::SlidingWindowMemory;
    use crate::tests::MockLLMProvider;
    use autoagents_llm::chat::{ChatMessage, ChatMessageBuilder, ChatRole};
    use std::sync::Arc;

    #[test]
    fn test_context_with_memory() {
        let llm = Arc::new(MockLLMProvider);
        let memory = Box::new(SlidingWindowMemory::new(5));
        let message = ChatMessage::user().content("Hello".to_string()).build();
        let system_message = ChatMessageBuilder::new(ChatRole::System)
            .content("System prompt".to_string())
            .build();

        let context = Context::new(llm, None)
            .with_memory(Some(Arc::new(Mutex::new(memory))))
            .with_messages(vec![message, system_message])
            .with_stream(true);

        assert!(context.memory().is_some());
        assert_eq!(context.messages().len(), 2);
        assert_eq!(context.messages()[0].role, ChatRole::User);
        assert_eq!(context.messages()[0].content, "Hello");
        assert!(context.stream());
    }

    #[test]
    fn test_context_tx_missing_returns_error() {
        let llm = Arc::new(MockLLMProvider);
        let context = Context::new(llm, None);
        let err = context.tx().unwrap_err();
        assert!(matches!(err, ContextError::EmptyTx));
    }
}
