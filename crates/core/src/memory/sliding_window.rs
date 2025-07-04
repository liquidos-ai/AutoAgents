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
