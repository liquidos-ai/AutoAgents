use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};

use super::{MemoryProvider, MemoryType, MessageEvent};
use autoagents_llm::{chat::ChatMessage, error::LLMError};

#[derive(Clone)]
pub struct SharedMemory<T: MemoryProvider> {
    inner: Arc<RwLock<T>>,
    event_sender: Option<broadcast::Sender<MessageEvent>>,
}

impl<T: MemoryProvider> SharedMemory<T> {
    pub fn new(provider: T) -> Self {
        Self {
            inner: Arc::new(RwLock::new(provider)),
            event_sender: None,
        }
    }

    /// Create a new reactive shared memory with the given broadcast channel capacity.
    ///
    /// The capacity determines how many message events can be queued before older
    /// ones get dropped for slow subscribers.
    pub fn new_reactive_with_capacity(provider: T, capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self {
            inner: Arc::new(RwLock::new(provider)),
            event_sender: Some(sender),
        }
    }

    /// Create a new reactive shared memory using the default capacity of **1000**.
    pub fn new_reactive(provider: T) -> Self {
        Self::new_reactive_with_capacity(provider, 1000)
    }

    /// Subscribe to message events (only available for reactive memory)
    pub fn subscribe(&self) -> broadcast::Receiver<MessageEvent> {
        self.event_sender
            .as_ref()
            .expect("subscribe() called on non-reactive memory")
            .subscribe()
    }
}

#[async_trait]
impl<T: MemoryProvider> MemoryProvider for SharedMemory<T> {
    async fn remember(&mut self, message: &ChatMessage) -> Result<(), LLMError> {
        let mut guard = self.inner.write().await;
        guard.remember(message).await
    }

    async fn recall(
        &self,
        query: &str,
        limit: Option<usize>,
    ) -> Result<Vec<ChatMessage>, LLMError> {
        let guard = self.inner.read().await;
        guard.recall(query, limit).await
    }

    async fn clear(&mut self) -> Result<(), LLMError> {
        let mut guard = self.inner.write().await;
        guard.clear().await
    }

    fn memory_type(&self) -> MemoryType {
        MemoryType::SlidingWindow
    }

    fn size(&self) -> usize {
        0
    }

    fn needs_summary(&self) -> bool {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let guard = self.inner.read().await;
                guard.needs_summary()
            })
        })
    }

    fn mark_for_summary(&mut self) {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mut guard = self.inner.write().await;
                guard.mark_for_summary();
            })
        })
    }

    fn replace_with_summary(&mut self, summary: String) {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mut guard = self.inner.write().await;
                guard.replace_with_summary(summary);
            })
        })
    }

    fn get_event_receiver(&self) -> Option<broadcast::Receiver<MessageEvent>> {
        self.event_sender.as_ref().map(|sender| sender.subscribe())
    }

    async fn remember_with_role(
        &mut self,
        message: &ChatMessage,
        role: String,
    ) -> Result<(), LLMError> {
        let mut guard = self.inner.write().await;
        guard.remember(message).await?;

        if let Some(sender) = &self.event_sender {
            let mut msg = message.clone();
            msg.content = msg.content.replace(&format!("[{role}]"), "");
            let event = MessageEvent { role, msg };
            let _ = sender.send(event);
        }

        Ok(())
    }
}
