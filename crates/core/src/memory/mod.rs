//! Memory module for storing and retrieving conversation history.
//!
//! This module provides various memory implementations for LLM conversations:
//! - SlidingWindowMemory: Simple FIFO memory with size limit
//! - EmbeddingMemory: Semantic search using embeddings (future)
//! - RAGMemory: Document-based retrieval (future)

pub mod chat_wrapper;
pub mod cond_macros;
pub mod shared_memory;
pub mod sliding_window;

use async_trait::async_trait;
use autoagents_llm::{chat::ChatMessage, error::LLMError};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::broadcast;

/// Event emitted when a message is added to reactive memory
#[derive(Debug, Clone)]
pub struct MessageEvent {
    /// Role of the agent that sent the message
    pub role: String,
    /// The chat message content
    pub msg: ChatMessage,
}

/// Conditions for triggering reactive message handlers
#[derive(Clone)]
pub enum MessageCondition {
    /// Always trigger
    Any,
    /// Trigger if message content equals exact string
    Eq(String),
    /// Trigger if message content contains substring
    Contains(String),
    /// Trigger if message content does not contain substring
    NotContains(String),
    /// Trigger if sender role matches
    RoleIs(String),
    /// Trigger if sender role does not match
    RoleNot(String),
    /// Trigger if message length is greater than specified
    LenGt(usize),
    /// Custom condition function
    Custom(Arc<dyn Fn(&ChatMessage) -> bool + Send + Sync>),
    /// Empty
    Empty,
    /// Trigger if all conditions are met
    All(Vec<MessageCondition>),
    /// Trigger if any condition is met
    AnyOf(Vec<MessageCondition>),
    /// Trigger if message content matches regex
    Regex(String),
}

impl MessageCondition {
    /// Check if the condition is met for the given message event
    pub fn matches(&self, event: &MessageEvent) -> bool {
        match self {
            MessageCondition::Any => true,
            MessageCondition::Eq(text) => event.msg.content == *text,
            MessageCondition::Contains(text) => event.msg.content.contains(text),
            MessageCondition::NotContains(text) => !event.msg.content.contains(text),
            MessageCondition::RoleIs(role) => event.role == *role,
            MessageCondition::RoleNot(role) => event.role != *role,
            MessageCondition::LenGt(len) => event.msg.content.len() > *len,
            MessageCondition::Custom(func) => func(&event.msg),
            MessageCondition::Empty => event.msg.content.is_empty(),
            MessageCondition::All(inner) => inner.iter().all(|c| c.matches(event)),
            MessageCondition::AnyOf(inner) => inner.iter().any(|c| c.matches(event)),
            MessageCondition::Regex(regex) => Regex::new(regex)
                .map(|re| re.is_match(&event.msg.content))
                .unwrap_or(false),
        }
    }
}

pub use chat_wrapper::ChatWithMemory;
pub use shared_memory::SharedMemory;
pub use sliding_window::{SlidingWindowMemory, TrimStrategy};

/// Types of memory implementations available
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryType {
    /// Simple sliding window that keeps the N most recent messages
    SlidingWindow,
}

/// Trait for memory providers that can store and retrieve conversation history.
///
/// Memory providers enable LLMs to maintain context across conversations by:
/// - Storing messages as they are exchanged
/// - Retrieving relevant past messages based on queries
/// - Managing memory size and cleanup
#[async_trait]
pub trait MemoryProvider: Send + Sync {
    /// Store a message in memory.
    ///
    /// # Arguments
    ///
    /// * `message` - The chat message to store
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the message was stored successfully
    /// * `Err(LLMError)` if storage failed
    async fn remember(&mut self, message: &ChatMessage) -> Result<(), LLMError>;

    /// Retrieve relevant messages from memory based on a query.
    ///
    /// # Arguments
    ///
    /// * `query` - The query string to search for relevant messages
    /// * `limit` - Optional maximum number of messages to return
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<ChatMessage>)` containing relevant messages
    /// * `Err(LLMError)` if retrieval failed
    async fn recall(&self, query: &str, limit: Option<usize>)
        -> Result<Vec<ChatMessage>, LLMError>;

    /// Clear all stored messages from memory.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if memory was cleared successfully
    /// * `Err(LLMError)` if clearing failed
    async fn clear(&mut self) -> Result<(), LLMError>;

    /// Get the type of this memory provider.
    ///
    /// # Returns
    ///
    /// The memory type enum variant
    fn memory_type(&self) -> MemoryType;

    /// Get the current number of stored messages.
    ///
    /// # Returns
    ///
    /// The number of messages currently in memory
    fn size(&self) -> usize;

    /// Check if the memory is empty.
    ///
    /// # Returns
    ///
    /// `true` if no messages are stored, `false` otherwise
    fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Check if memory needs summarization
    fn needs_summary(&self) -> bool {
        false
    }

    /// Mark memory as needing summarization
    fn mark_for_summary(&mut self) {}

    /// Replace all messages with a summary
    fn replace_with_summary(&mut self, _summary: String) {}

    /// Get a receiver for reactive events if this memory supports them
    fn get_event_receiver(&self) -> Option<broadcast::Receiver<MessageEvent>> {
        None
    }

    /// Remember a message with a specific role for reactive memory
    async fn remember_with_role(
        &mut self,
        message: &ChatMessage,
        _role: String,
    ) -> Result<(), LLMError> {
        self.remember(message).await
    }
}
