use async_trait::async_trait;
use autoagents_llm::{chat::ChatMessage, error::LLMError};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::broadcast;

mod sliding_window;
pub use sliding_window::SlidingWindowMemory;

#[cfg(test)]
mod tests {
    use super::*;
    use autoagents_llm::chat::{ChatMessage, ChatRole, MessageType};
    use autoagents_llm::error::LLMError;
    use std::sync::Arc;

    // Mock memory provider for testing
    struct MockMemoryProvider {
        messages: Vec<ChatMessage>,
        should_fail: bool,
    }

    impl MockMemoryProvider {
        fn new() -> Self {
            Self {
                messages: Vec::new(),
                should_fail: false,
            }
        }

        fn with_failure() -> Self {
            Self {
                messages: Vec::new(),
                should_fail: true,
            }
        }

        fn with_messages(messages: Vec<ChatMessage>) -> Self {
            Self {
                messages,
                should_fail: false,
            }
        }
    }

    #[async_trait::async_trait]
    impl MemoryProvider for MockMemoryProvider {
        async fn remember(&mut self, message: &ChatMessage) -> Result<(), LLMError> {
            if self.should_fail {
                return Err(LLMError::ProviderError("Mock memory error".to_string()));
            }
            self.messages.push(message.clone());
            Ok(())
        }

        async fn recall(
            &self,
            _query: &str,
            limit: Option<usize>,
        ) -> Result<Vec<ChatMessage>, LLMError> {
            if self.should_fail {
                return Err(LLMError::ProviderError("Mock recall error".to_string()));
            }
            let limit = limit.unwrap_or_else(|| self.messages.len());
            Ok(self.messages.iter().take(limit).cloned().collect())
        }

        async fn clear(&mut self) -> Result<(), LLMError> {
            if self.should_fail {
                return Err(LLMError::ProviderError("Mock clear error".to_string()));
            }
            self.messages.clear();
            Ok(())
        }

        fn memory_type(&self) -> MemoryType {
            MemoryType::SlidingWindow
        }

        fn size(&self) -> usize {
            self.messages.len()
        }

        fn clone_box(&self) -> Box<dyn MemoryProvider> {
            unimplemented!()
        }
    }

    #[test]
    fn test_memory_type_serialization() {
        let sliding_window = MemoryType::SlidingWindow;
        let serialized = serde_json::to_string(&sliding_window).unwrap();
        assert_eq!(serialized, "\"SlidingWindow\"");
    }

    #[test]
    fn test_memory_type_deserialization() {
        let deserialized: MemoryType = serde_json::from_str("\"SlidingWindow\"").unwrap();
        assert_eq!(deserialized, MemoryType::SlidingWindow);
    }

    #[test]
    fn test_memory_type_debug() {
        let sliding_window = MemoryType::SlidingWindow;
        let debug_str = format!("{sliding_window:?}");
        assert!(debug_str.contains("SlidingWindow"));
    }

    #[test]
    fn test_memory_type_clone() {
        let sliding_window = MemoryType::SlidingWindow;
        let cloned = sliding_window.clone();
        assert_eq!(sliding_window, cloned);
    }

    #[test]
    fn test_memory_type_equality() {
        let type1 = MemoryType::SlidingWindow;
        let type2 = MemoryType::SlidingWindow;
        assert_eq!(type1, type2);
    }

    #[test]
    fn test_message_condition_any() {
        let condition = MessageCondition::Any;
        let message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "test message".to_string(),
        };
        let event = MessageEvent {
            role: "user".to_string(),
            msg: message,
        };
        assert!(condition.matches(&event));
    }

    #[test]
    fn test_message_condition_eq() {
        let condition = MessageCondition::Eq("test message".to_string());
        let message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "test message".to_string(),
        };
        let event = MessageEvent {
            role: "user".to_string(),
            msg: message,
        };
        assert!(condition.matches(&event));

        let different_message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "different message".to_string(),
        };
        let different_event = MessageEvent {
            role: "user".to_string(),
            msg: different_message,
        };
        assert!(!condition.matches(&different_event));
    }

    #[test]
    fn test_message_condition_contains() {
        let condition = MessageCondition::Contains("test".to_string());
        let message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "this is a test message".to_string(),
        };
        let event = MessageEvent {
            role: "user".to_string(),
            msg: message,
        };
        assert!(condition.matches(&event));

        let non_matching_message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "this is different".to_string(),
        };
        let non_matching_event = MessageEvent {
            role: "user".to_string(),
            msg: non_matching_message,
        };
        assert!(!condition.matches(&non_matching_event));
    }

    #[test]
    fn test_message_condition_not_contains() {
        let condition = MessageCondition::NotContains("error".to_string());
        let message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "this is a test message".to_string(),
        };
        let event = MessageEvent {
            role: "user".to_string(),
            msg: message,
        };
        assert!(condition.matches(&event));

        let error_message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "this is an error message".to_string(),
        };
        let error_event = MessageEvent {
            role: "user".to_string(),
            msg: error_message,
        };
        assert!(!condition.matches(&error_event));
    }

    #[test]
    fn test_message_condition_role_is() {
        let condition = MessageCondition::RoleIs("user".to_string());
        let message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "test message".to_string(),
        };
        let event = MessageEvent {
            role: "user".to_string(),
            msg: message,
        };
        assert!(condition.matches(&event));

        let assistant_event = MessageEvent {
            role: "assistant".to_string(),
            msg: ChatMessage {
                role: ChatRole::Assistant,
                message_type: MessageType::Text,
                content: "test message".to_string(),
            },
        };
        assert!(!condition.matches(&assistant_event));
    }

    #[test]
    fn test_message_condition_role_not() {
        let condition = MessageCondition::RoleNot("system".to_string());
        let message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "test message".to_string(),
        };
        let event = MessageEvent {
            role: "user".to_string(),
            msg: message,
        };
        assert!(condition.matches(&event));

        let system_event = MessageEvent {
            role: "system".to_string(),
            msg: ChatMessage {
                role: ChatRole::System,
                message_type: MessageType::Text,
                content: "test message".to_string(),
            },
        };
        assert!(!condition.matches(&system_event));
    }

    #[test]
    fn test_message_condition_len_gt() {
        let condition = MessageCondition::LenGt(5);
        let long_message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "this is a long message".to_string(),
        };
        let long_event = MessageEvent {
            role: "user".to_string(),
            msg: long_message,
        };
        assert!(condition.matches(&long_event));

        let short_message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "hi".to_string(),
        };
        let short_event = MessageEvent {
            role: "user".to_string(),
            msg: short_message,
        };
        assert!(!condition.matches(&short_event));
    }

    #[test]
    fn test_message_condition_custom() {
        let condition = MessageCondition::Custom(Arc::new(|msg| msg.content.starts_with("hello")));
        let hello_message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "hello world".to_string(),
        };
        let hello_event = MessageEvent {
            role: "user".to_string(),
            msg: hello_message,
        };
        assert!(condition.matches(&hello_event));

        let goodbye_message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "goodbye world".to_string(),
        };
        let goodbye_event = MessageEvent {
            role: "user".to_string(),
            msg: goodbye_message,
        };
        assert!(!condition.matches(&goodbye_event));
    }

    #[test]
    fn test_message_condition_empty() {
        let condition = MessageCondition::Empty;
        let empty_message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "".to_string(),
        };
        let empty_event = MessageEvent {
            role: "user".to_string(),
            msg: empty_message,
        };
        assert!(condition.matches(&empty_event));

        let non_empty_message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "not empty".to_string(),
        };
        let non_empty_event = MessageEvent {
            role: "user".to_string(),
            msg: non_empty_message,
        };
        assert!(!condition.matches(&non_empty_event));
    }

    #[test]
    fn test_message_condition_all() {
        let condition = MessageCondition::All(vec![
            MessageCondition::Contains("test".to_string()),
            MessageCondition::LenGt(5),
            MessageCondition::RoleIs("user".to_string()),
        ]);

        let matching_message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "this is a test message".to_string(),
        };
        let matching_event = MessageEvent {
            role: "user".to_string(),
            msg: matching_message,
        };
        assert!(condition.matches(&matching_event));

        let non_matching_message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "hi".to_string(),
        };
        let non_matching_event = MessageEvent {
            role: "user".to_string(),
            msg: non_matching_message,
        };
        assert!(!condition.matches(&non_matching_event));
    }

    #[test]
    fn test_message_condition_any_of() {
        let condition = MessageCondition::AnyOf(vec![
            MessageCondition::Contains("hello".to_string()),
            MessageCondition::Contains("goodbye".to_string()),
            MessageCondition::Empty,
        ]);

        let hello_message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "hello world".to_string(),
        };
        let hello_event = MessageEvent {
            role: "user".to_string(),
            msg: hello_message,
        };
        assert!(condition.matches(&hello_event));

        let goodbye_message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "goodbye world".to_string(),
        };
        let goodbye_event = MessageEvent {
            role: "user".to_string(),
            msg: goodbye_message,
        };
        assert!(condition.matches(&goodbye_event));

        let empty_message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "".to_string(),
        };
        let empty_event = MessageEvent {
            role: "user".to_string(),
            msg: empty_message,
        };
        assert!(condition.matches(&empty_event));

        let non_matching_message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "test message".to_string(),
        };
        let non_matching_event = MessageEvent {
            role: "user".to_string(),
            msg: non_matching_message,
        };
        assert!(!condition.matches(&non_matching_event));
    }

    #[test]
    fn test_message_condition_regex() {
        let condition = MessageCondition::Regex(r"\d+".to_string());
        let number_message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "there are 123 items".to_string(),
        };
        let number_event = MessageEvent {
            role: "user".to_string(),
            msg: number_message,
        };
        assert!(condition.matches(&number_event));

        let no_number_message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "no numbers here".to_string(),
        };
        let no_number_event = MessageEvent {
            role: "user".to_string(),
            msg: no_number_message,
        };
        assert!(!condition.matches(&no_number_event));
    }

    #[test]
    fn test_message_condition_regex_invalid() {
        let condition = MessageCondition::Regex("[invalid regex".to_string());
        let message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "test message".to_string(),
        };
        let event = MessageEvent {
            role: "user".to_string(),
            msg: message,
        };
        assert!(!condition.matches(&event));
    }

    #[test]
    fn test_message_event_creation() {
        let message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "test message".to_string(),
        };
        let event = MessageEvent {
            role: "user".to_string(),
            msg: message.clone(),
        };
        assert_eq!(event.role, "user");
        assert_eq!(event.msg.content, "test message");
    }

    #[test]
    fn test_message_event_clone() {
        let message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "test message".to_string(),
        };
        let event = MessageEvent {
            role: "user".to_string(),
            msg: message,
        };
        let cloned = event.clone();
        assert_eq!(event.role, cloned.role);
        assert_eq!(event.msg.content, cloned.msg.content);
    }

    #[test]
    fn test_message_event_debug() {
        let message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "test message".to_string(),
        };
        let event = MessageEvent {
            role: "user".to_string(),
            msg: message,
        };
        let debug_str = format!("{event:?}");
        assert!(debug_str.contains("MessageEvent"));
        assert!(debug_str.contains("user"));
    }

    #[tokio::test]
    async fn test_mock_memory_provider_remember() {
        let mut provider = MockMemoryProvider::new();
        let message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "test message".to_string(),
        };

        let result = provider.remember(&message).await;
        assert!(result.is_ok());
        assert_eq!(provider.size(), 1);
    }

    #[tokio::test]
    async fn test_mock_memory_provider_remember_failure() {
        let mut provider = MockMemoryProvider::with_failure();
        let message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "test message".to_string(),
        };

        let result = provider.remember(&message).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Mock memory error")
        );
    }

    #[tokio::test]
    async fn test_mock_memory_provider_recall() {
        let messages = vec![
            ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: "first message".to_string(),
            },
            ChatMessage {
                role: ChatRole::Assistant,
                message_type: MessageType::Text,
                content: "second message".to_string(),
            },
        ];
        let provider = MockMemoryProvider::with_messages(messages);

        let result = provider.recall("", None).await;
        assert!(result.is_ok());
        let recalled = result.unwrap();
        assert_eq!(recalled.len(), 2);
        assert_eq!(recalled[0].content, "first message");
        assert_eq!(recalled[1].content, "second message");
    }

    #[tokio::test]
    async fn test_mock_memory_provider_recall_with_limit() {
        let messages = vec![
            ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: "first message".to_string(),
            },
            ChatMessage {
                role: ChatRole::Assistant,
                message_type: MessageType::Text,
                content: "second message".to_string(),
            },
        ];
        let provider = MockMemoryProvider::with_messages(messages);

        let result = provider.recall("", Some(1)).await;
        assert!(result.is_ok());
        let recalled = result.unwrap();
        assert_eq!(recalled.len(), 1);
        assert_eq!(recalled[0].content, "first message");
    }

    #[tokio::test]
    async fn test_mock_memory_provider_recall_failure() {
        let provider = MockMemoryProvider::with_failure();

        let result = provider.recall("", None).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Mock recall error")
        );
    }

    #[tokio::test]
    async fn test_mock_memory_provider_clear() {
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "first message".to_string(),
        }];
        let mut provider = MockMemoryProvider::with_messages(messages);
        assert_eq!(provider.size(), 1);

        let result = provider.clear().await;
        assert!(result.is_ok());
        assert_eq!(provider.size(), 0);
    }

    #[tokio::test]
    async fn test_mock_memory_provider_clear_failure() {
        let mut provider = MockMemoryProvider::with_failure();

        let result = provider.clear().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Mock clear error"));
    }

    #[test]
    fn test_mock_memory_provider_memory_type() {
        let provider = MockMemoryProvider::new();
        assert_eq!(provider.memory_type(), MemoryType::SlidingWindow);
    }

    #[test]
    fn test_mock_memory_provider_size() {
        let provider = MockMemoryProvider::new();
        assert_eq!(provider.size(), 0);

        let messages = vec![ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "message".to_string(),
        }];
        let provider_with_messages = MockMemoryProvider::with_messages(messages);
        assert_eq!(provider_with_messages.size(), 1);
    }

    #[test]
    fn test_mock_memory_provider_is_empty() {
        let provider = MockMemoryProvider::new();
        assert!(provider.is_empty());

        let messages = vec![ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "message".to_string(),
        }];
        let provider_with_messages = MockMemoryProvider::with_messages(messages);
        assert!(!provider_with_messages.is_empty());
    }

    #[test]
    fn test_memory_provider_default_methods() {
        let provider = MockMemoryProvider::new();
        assert!(!provider.needs_summary());
        assert!(provider.get_event_receiver().is_none());
    }

    #[tokio::test]
    async fn test_memory_provider_remember_with_role() {
        let mut provider = MockMemoryProvider::new();
        let message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "test message".to_string(),
        };

        let result = provider
            .remember_with_role(&message, "custom_role".to_string())
            .await;
        assert!(result.is_ok());
        assert_eq!(provider.size(), 1);
    }

    #[test]
    fn test_memory_provider_mark_for_summary() {
        let mut provider = MockMemoryProvider::new();
        provider.mark_for_summary(); // Should not panic
        assert!(!provider.needs_summary()); // Default implementation
    }

    #[test]
    fn test_memory_provider_replace_with_summary() {
        let mut provider = MockMemoryProvider::new();
        provider.replace_with_summary("Summary text".to_string()); // Should not panic
        assert_eq!(provider.size(), 0); // Should not change size in default implementation
    }
}

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
    #[cfg(not(target_arch = "wasm32"))]
    fn get_event_receiver(&self) -> Option<broadcast::Receiver<MessageEvent>> {
        None
    }

    #[cfg(target_arch = "wasm32")]
    fn get_event_receiver(&self) -> Option<()> {
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

    /// Clone the memory provider into a new Box
    /// This is needed for persistence across requests
    fn clone_box(&self) -> Box<dyn MemoryProvider>;

    /// Get a unique identifier for this memory instance
    /// Used for caching and persistence
    fn id(&self) -> Option<String> {
        None
    }

    /// Preload memory from a cache or storage
    /// Returns true if memory was successfully preloaded
    fn preload(&mut self, _data: Vec<ChatMessage>) -> bool {
        false
    }

    /// Export memory data for caching or persistence
    /// Returns the messages that should be persisted
    fn export(&self) -> Vec<ChatMessage> {
        Vec::new()
    }
}
