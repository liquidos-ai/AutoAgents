//! Chat functionality stub for liquid-edge
//!
//! This module provides basic chat-related structures and utilities.
//! The full chat implementation should be built using the core traits
//! and runtime components.

#[cfg(feature = "chat")]
use crate::error::{EdgeError, EdgeResult};

/// Chat session configuration
#[cfg(feature = "chat")]
#[derive(Debug, Clone)]
pub struct ChatConfig {
    /// Maximum conversation history to maintain
    pub max_history: usize,
    /// System message to use for the session
    pub system_message: Option<String>,
    /// Chat template to use for formatting
    pub template: Option<String>,
}

#[cfg(feature = "chat")]
impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            max_history: 10,
            system_message: Some("You are a helpful assistant.".to_string()),
            template: None,
        }
    }
}

/// Simple chat session for managing conversation state
#[cfg(feature = "chat")]
#[derive(Debug)]
pub struct ChatSession {
    config: ChatConfig,
    history: Vec<(String, String)>, // (user, assistant) pairs
}

#[cfg(feature = "chat")]
impl ChatSession {
    /// Create a new chat session
    pub fn new(config: ChatConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
        }
    }

    /// Create a chat session with default configuration
    pub fn default() -> Self {
        Self::new(ChatConfig::default())
    }

    /// Add a message exchange to the history
    pub fn add_exchange(&mut self, user_message: String, assistant_response: String) {
        self.history.push((user_message, assistant_response));

        // Trim history if it exceeds max length
        if self.history.len() > self.config.max_history {
            self.history.remove(0);
        }
    }

    /// Get conversation history
    pub fn get_history(&self) -> &[(String, String)] {
        &self.history
    }

    /// Clear conversation history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Get the current configuration
    pub fn config(&self) -> &ChatConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: ChatConfig) {
        self.config = config;
    }
}

// Stub implementations when chat feature is not enabled
#[cfg(not(feature = "chat"))]
/// Stub chat configuration
pub struct ChatConfig;

#[cfg(not(feature = "chat"))]
impl Default for ChatConfig {
    fn default() -> Self {
        Self
    }
}

#[cfg(not(feature = "chat"))]
/// Stub chat session
pub struct ChatSession;

#[cfg(not(feature = "chat"))]
impl ChatSession {
    /// Stub constructor that indicates feature is not available
    pub fn new(_config: ChatConfig) -> Self {
        Self
    }

    /// Stub default constructor
    pub fn default() -> Self {
        Self
    }
}

#[cfg(test)]
#[cfg(feature = "chat")]
mod tests {
    use super::*;

    #[test]
    fn test_chat_session_creation() {
        let config = ChatConfig::default();
        let session = ChatSession::new(config);
        assert_eq!(session.get_history().len(), 0);
    }

    #[test]
    fn test_add_exchange() {
        let mut session = ChatSession::default();
        session.add_exchange("Hello".to_string(), "Hi there!".to_string());

        let history = session.get_history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].0, "Hello");
        assert_eq!(history[0].1, "Hi there!");
    }

    #[test]
    fn test_history_limit() {
        let config = ChatConfig {
            max_history: 2,
            system_message: None,
            template: None,
        };
        let mut session = ChatSession::new(config);

        session.add_exchange("1".to_string(), "A".to_string());
        session.add_exchange("2".to_string(), "B".to_string());
        session.add_exchange("3".to_string(), "C".to_string());

        let history = session.get_history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].0, "2");
        assert_eq!(history[1].0, "3");
    }

    #[test]
    fn test_clear_history() {
        let mut session = ChatSession::default();
        session.add_exchange("Hello".to_string(), "Hi".to_string());
        assert_eq!(session.get_history().len(), 1);

        session.clear_history();
        assert_eq!(session.get_history().len(), 0);
    }
}
