use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use autoagents_llm::chat::ChatMessage;
use autoagents_llm::LLMProvider;
use crate::agent::{AgentConfig, AgentState};
use crate::agent::memory::MemoryProvider;
use crate::protocol::Event;
use crate::tool::ToolT;

pub struct Context {
    llm: Arc<dyn LLMProvider>,
    messages: Vec<ChatMessage>,
    memory: Option<Arc<RwLock<Box<dyn MemoryProvider>>>>,
    tools: Vec<Box<dyn ToolT>>,
    config: AgentConfig,
    state: Arc<RwLock<AgentState>>,
    tx: mpsc::Sender<Event>,
    stream: bool
}

impl Context {
    pub fn new(llm: Arc<dyn LLMProvider>, tx: mpsc::Sender<Event>) -> Self {
        Self {
            llm,
            messages: vec![],
            memory: None,
            tools: vec![],
            config: AgentConfig::default(),
            state: Arc::new(RwLock::new(AgentState::new())),
            stream: false,
            tx
        }
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