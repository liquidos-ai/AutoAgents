//! Simple sliding window memory implementation.
//!
//! This module provides a basic FIFO (First In, First Out) memory that maintains
//! a fixed-size window of the most recent conversation messages.
use async_trait::async_trait;
use autoagents_llm::{chat::ChatMessage, error::LLMError};
use std::collections::VecDeque;

use super::{MemoryProvider, MemoryType};

/// Strategy for handling memory when window size limit is reached
#[derive(Debug, Clone)]
pub enum TrimStrategy {
    /// Drop oldest messages (FIFO behavior)
    Drop,
    /// Summarize all messages into one before adding new ones
    Summarize,
}

/// Simple sliding window memory that keeps the N most recent messages.
///
/// This implementation uses a FIFO strategy where old messages are automatically
/// removed when the window size limit is reached. It's suitable for:
/// - Simple conversation contexts
/// - Memory-constrained environments
/// - Cases where only recent context matters
///
#[derive(Debug, Clone)]
pub struct SlidingWindowMemory {
    messages: VecDeque<ChatMessage>,
    window_size: usize,
    trim_strategy: TrimStrategy,
    needs_summary: bool,
}

impl SlidingWindowMemory {
    /// Create a new sliding window memory with the specified window size.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Maximum number of messages to keep in memory
    ///
    /// # Panics
    ///
    /// Panics if `window_size` is 0
    ///
    pub fn new(window_size: usize) -> Self {
        Self::with_strategy(window_size, TrimStrategy::Drop)
    }

    /// Create a new sliding window memory with specified trim strategy
    ///
    /// # Arguments
    ///
    /// * `window_size` - Maximum number of messages to keep in memory
    /// * `strategy` - How to handle overflow when window is full
    pub fn with_strategy(window_size: usize, strategy: TrimStrategy) -> Self {
        if window_size == 0 {
            panic!("Window size must be greater than 0");
        }

        Self {
            messages: VecDeque::with_capacity(window_size),
            window_size,
            trim_strategy: strategy,
            needs_summary: false,
        }
    }

    /// Get the configured window size.
    ///
    /// # Returns
    ///
    /// The maximum number of messages this memory can hold
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Get all stored messages in chronological order.
    ///
    /// # Returns
    ///
    /// A vector containing all messages from oldest to newest
    pub fn messages(&self) -> Vec<ChatMessage> {
        Vec::from(self.messages.clone())
    }

    /// Get the most recent N messages.
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum number of recent messages to return
    ///
    /// # Returns
    ///
    /// A vector containing the most recent messages, up to `limit`
    pub fn recent_messages(&self, limit: usize) -> Vec<ChatMessage> {
        let len = self.messages.len();
        let start = len.saturating_sub(limit);
        self.messages.range(start..).cloned().collect()
    }

    /// Check if memory needs summarization
    pub fn needs_summary(&self) -> bool {
        self.needs_summary
    }

    /// Mark memory as needing summarization
    pub fn mark_for_summary(&mut self) {
        self.needs_summary = true;
    }

    /// Replace all messages with a summary
    ///
    /// # Arguments
    ///
    /// * `summary` - The summary text to replace all messages with
    pub fn replace_with_summary(&mut self, summary: String) {
        self.messages.clear();
        self.messages
            .push_back(ChatMessage::assistant().content(summary).build());
        self.needs_summary = false;
    }
}

#[async_trait]
impl MemoryProvider for SlidingWindowMemory {
    async fn remember(&mut self, message: &ChatMessage) -> Result<(), LLMError> {
        if self.messages.len() >= self.window_size {
            match self.trim_strategy {
                TrimStrategy::Drop => {
                    self.messages.pop_front();
                }
                TrimStrategy::Summarize => {
                    self.mark_for_summary();
                }
            }
        }
        self.messages.push_back(message.clone());
        Ok(())
    }

    async fn recall(
        &self,
        _query: &str,
        limit: Option<usize>,
    ) -> Result<Vec<ChatMessage>, LLMError> {
        let limit = limit.unwrap_or(self.messages.len());
        Ok(self.recent_messages(limit))
    }

    async fn clear(&mut self) -> Result<(), LLMError> {
        self.messages.clear();
        Ok(())
    }

    fn memory_type(&self) -> MemoryType {
        MemoryType::SlidingWindow
    }

    fn size(&self) -> usize {
        self.messages.len()
    }

    fn needs_summary(&self) -> bool {
        self.needs_summary
    }

    fn mark_for_summary(&mut self) {
        self.needs_summary = true;
    }

    fn replace_with_summary(&mut self, summary: String) {
        self.messages.clear();
        self.messages
            .push_back(ChatMessage::assistant().content(summary).build());
        self.needs_summary = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoagents_llm::chat::{ChatMessage, ChatRole, MessageType};

    #[test]
    fn test_new_sliding_window_memory() {
        let memory = SlidingWindowMemory::new(5);
        assert_eq!(memory.window_size(), 5);
        assert_eq!(memory.size(), 0);
        assert!(memory.is_empty());
        assert_eq!(memory.memory_type(), MemoryType::SlidingWindow);
    }

    #[test]
    fn test_sliding_window_memory_with_strategy() {
        let memory = SlidingWindowMemory::with_strategy(3, TrimStrategy::Summarize);
        assert_eq!(memory.window_size(), 3);
        assert_eq!(memory.size(), 0);
        assert!(memory.is_empty());
    }

    #[test]
    #[should_panic(expected = "Window size must be greater than 0")]
    fn test_new_sliding_window_memory_zero_size() {
        SlidingWindowMemory::new(0);
    }

    #[tokio::test]
    async fn test_remember_single_message() {
        let mut memory = SlidingWindowMemory::new(3);
        let message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "Hello".to_string(),
        };

        memory.remember(&message).await.unwrap();
        assert_eq!(memory.size(), 1);
        assert!(!memory.is_empty());

        let messages = memory.messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "Hello");
    }

    #[tokio::test]
    async fn test_remember_multiple_messages() {
        let mut memory = SlidingWindowMemory::new(3);

        for i in 1..=3 {
            let message = ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: format!("Message {}", i),
            };
            memory.remember(&message).await.unwrap();
        }

        assert_eq!(memory.size(), 3);
        let messages = memory.messages();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].content, "Message 1");
        assert_eq!(messages[2].content, "Message 3");
    }

    #[tokio::test]
    async fn test_sliding_window_overflow_drop_strategy() {
        let mut memory = SlidingWindowMemory::with_strategy(2, TrimStrategy::Drop);

        // Add 3 messages to a window of size 2
        for i in 1..=3 {
            let message = ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: format!("Message {}", i),
            };
            memory.remember(&message).await.unwrap();
        }

        // Should only keep the last 2 messages
        assert_eq!(memory.size(), 2);
        let messages = memory.messages();
        assert_eq!(messages[0].content, "Message 2");
        assert_eq!(messages[1].content, "Message 3");
    }

    #[tokio::test]
    async fn test_sliding_window_overflow_summarize_strategy() {
        let mut memory = SlidingWindowMemory::with_strategy(2, TrimStrategy::Summarize);

        // Add first message
        let message1 = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "First message".to_string(),
        };
        memory.remember(&message1).await.unwrap();

        // Add second message
        let message2 = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "Second message".to_string(),
        };
        memory.remember(&message2).await.unwrap();

        // Add third message - should trigger summarize flag
        let message3 = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "Third message".to_string(),
        };
        memory.remember(&message3).await.unwrap();

        assert!(memory.needs_summary());
        assert_eq!(memory.size(), 3); // Still contains all messages until summarized
    }

    #[tokio::test]
    async fn test_recall_all_messages() {
        let mut memory = SlidingWindowMemory::new(3);

        for i in 1..=3 {
            let message = ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: format!("Message {}", i),
            };
            memory.remember(&message).await.unwrap();
        }

        let recalled = memory.recall("", None).await.unwrap();
        assert_eq!(recalled.len(), 3);
        assert_eq!(recalled[0].content, "Message 1");
        assert_eq!(recalled[2].content, "Message 3");
    }

    #[tokio::test]
    async fn test_recall_with_limit() {
        let mut memory = SlidingWindowMemory::new(5);

        for i in 1..=5 {
            let message = ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: format!("Message {}", i),
            };
            memory.remember(&message).await.unwrap();
        }

        let recalled = memory.recall("", Some(2)).await.unwrap();
        assert_eq!(recalled.len(), 2);
        assert_eq!(recalled[0].content, "Message 4");
        assert_eq!(recalled[1].content, "Message 5");
    }

    #[tokio::test]
    async fn test_clear_memory() {
        let mut memory = SlidingWindowMemory::new(3);

        let message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "Test message".to_string(),
        };
        memory.remember(&message).await.unwrap();

        assert_eq!(memory.size(), 1);
        memory.clear().await.unwrap();
        assert_eq!(memory.size(), 0);
        assert!(memory.is_empty());
    }

    #[test]
    fn test_recent_messages() {
        let mut memory = SlidingWindowMemory::new(5);

        // Add messages directly to the internal deque for testing
        for i in 1..=5 {
            let message = ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: format!("Message {}", i),
            };
            memory.messages.push_back(message);
        }

        let recent = memory.recent_messages(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].content, "Message 3");
        assert_eq!(recent[2].content, "Message 5");
    }

    #[test]
    fn test_recent_messages_limit_exceeds_size() {
        let mut memory = SlidingWindowMemory::new(5);

        // Add only 2 messages
        for i in 1..=2 {
            let message = ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: format!("Message {}", i),
            };
            memory.messages.push_back(message);
        }

        let recent = memory.recent_messages(10);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].content, "Message 1");
        assert_eq!(recent[1].content, "Message 2");
    }

    #[test]
    fn test_mark_for_summary() {
        let mut memory = SlidingWindowMemory::new(3);
        assert!(!memory.needs_summary());

        memory.mark_for_summary();
        assert!(memory.needs_summary());
    }

    #[test]
    fn test_replace_with_summary() {
        let mut memory = SlidingWindowMemory::new(3);

        // Add some messages
        for i in 1..=3 {
            let message = ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: format!("Message {}", i),
            };
            memory.messages.push_back(message);
        }

        memory.mark_for_summary();
        assert!(memory.needs_summary());
        assert_eq!(memory.size(), 3);

        memory.replace_with_summary("This is a summary".to_string());

        assert!(!memory.needs_summary());
        assert_eq!(memory.size(), 1);
        let messages = memory.messages();
        assert_eq!(messages[0].content, "This is a summary");
        assert_eq!(messages[0].role, ChatRole::Assistant);
    }

    #[test]
    fn test_memory_provider_trait_methods() {
        let memory = SlidingWindowMemory::new(3);

        // Test trait methods
        assert_eq!(memory.memory_type(), MemoryType::SlidingWindow);
        assert_eq!(memory.size(), 0);
        assert!(memory.is_empty());
        assert!(!memory.needs_summary());
        assert!(memory.get_event_receiver().is_none());
    }
}
