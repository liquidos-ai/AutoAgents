use crate::agent::memory::MemoryProvider;
use crate::tool::ToolCallResult;
use autoagents_llm::ToolCall;
use autoagents_llm::chat::{ChatMessage, ChatRole, MessageType};
use autoagents_protocol::ImageMime;
use std::sync::Arc;

use super::tool_processor::ToolProcessor;

#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::Mutex;

#[cfg(target_arch = "wasm32")]
use futures::lock::Mutex;

/// Helper for managing agent memory operations
pub struct MemoryHelper;

impl MemoryHelper {
    /// Store a message in memory if available
    pub async fn store_message(
        memory: &Option<Arc<Mutex<Box<dyn MemoryProvider>>>>,
        message: ChatMessage,
    ) {
        if let Some(mem) = memory {
            let mut mem = mem.lock().await;
            let _ = mem.remember(&message).await;
        }
    }

    /// Store tool calls and results in memory
    pub async fn store_tool_interaction(
        memory: &Option<Arc<Mutex<Box<dyn MemoryProvider>>>>,
        tool_calls: &[ToolCall],
        tool_results: &[ToolCallResult],
        response_text: &str,
    ) {
        if let Some(mem) = memory {
            let mut mem = mem.lock().await;

            // Record assistant calling tools
            let _ = mem
                .remember(&ChatMessage {
                    role: ChatRole::Assistant,
                    message_type: MessageType::ToolUse(tool_calls.to_vec()),
                    content: response_text.to_string(),
                })
                .await;

            // Create and store tool results
            let result_tool_calls =
                ToolProcessor::create_result_tool_calls(tool_calls, tool_results);

            let _ = mem
                .remember(&ChatMessage {
                    role: ChatRole::Tool,
                    message_type: MessageType::ToolResult(result_tool_calls),
                    content: String::default(),
                })
                .await;
        }
    }

    /// Store user message in memory
    pub async fn store_user_message(
        memory: &Option<Arc<Mutex<Box<dyn MemoryProvider>>>>,
        content: String,
        image: Option<(ImageMime, Vec<u8>)>,
    ) {
        if let Some(mem) = memory {
            let mut mem = mem.lock().await;
            let message = if let Some((mime, data)) = image {
                ChatMessage {
                    role: ChatRole::User,
                    message_type: MessageType::Image((mime.into(), data)),
                    content,
                }
            } else {
                ChatMessage {
                    role: ChatRole::User,
                    message_type: MessageType::Text,
                    content,
                }
            };
            let _ = mem.remember(&message).await;
        }
    }

    /// Store assistant response in memory
    pub async fn store_assistant_response(
        memory: &Option<Arc<Mutex<Box<dyn MemoryProvider>>>>,
        response: String,
    ) {
        Self::store_message(
            memory,
            ChatMessage {
                role: ChatRole::Assistant,
                message_type: MessageType::Text,
                content: response,
            },
        )
        .await;
    }

    /// Recall messages from memory
    pub async fn recall_messages(
        memory: &Option<Arc<Mutex<Box<dyn MemoryProvider>>>>,
    ) -> Vec<ChatMessage> {
        if let Some(mem) = memory
            && let Ok(messages) = mem.lock().await.recall("", None).await
        {
            return messages;
        }
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::memory::SlidingWindowMemory;

    fn make_memory() -> Arc<Mutex<Box<dyn MemoryProvider>>> {
        Arc::new(Mutex::new(Box::new(SlidingWindowMemory::new(10))))
    }

    #[tokio::test]
    async fn test_store_message_with_none() {
        // Should not panic
        MemoryHelper::store_message(
            &None,
            ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: "test".to_string(),
            },
        )
        .await;
    }

    #[tokio::test]
    async fn test_store_message_with_memory() {
        let mem = make_memory();
        MemoryHelper::store_message(
            &Some(mem.clone()),
            ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: "hello".to_string(),
            },
        )
        .await;
        let stored = mem.lock().await.recall("", None).await.unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].content, "hello");
    }

    #[tokio::test]
    async fn test_store_user_message_text() {
        let mem = make_memory();
        MemoryHelper::store_user_message(&Some(mem.clone()), "user msg".to_string(), None).await;
        let stored = mem.lock().await.recall("", None).await.unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].content, "user msg");
        assert!(matches!(stored[0].message_type, MessageType::Text));
    }

    #[tokio::test]
    async fn test_store_user_message_image() {
        let mem = make_memory();
        MemoryHelper::store_user_message(
            &Some(mem.clone()),
            "image msg".to_string(),
            Some((ImageMime::PNG, vec![1, 2, 3])),
        )
        .await;
        let stored = mem.lock().await.recall("", None).await.unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].content, "image msg");
    }

    #[tokio::test]
    async fn test_store_user_message_none_memory() {
        MemoryHelper::store_user_message(&None, "msg".to_string(), None).await;
    }

    #[tokio::test]
    async fn test_store_assistant_response() {
        let mem = make_memory();
        MemoryHelper::store_assistant_response(&Some(mem.clone()), "reply".to_string()).await;
        let stored = mem.lock().await.recall("", None).await.unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].content, "reply");
        assert!(matches!(stored[0].role, ChatRole::Assistant));
    }

    #[tokio::test]
    async fn test_store_assistant_response_none() {
        MemoryHelper::store_assistant_response(&None, "reply".to_string()).await;
    }

    #[tokio::test]
    async fn test_store_tool_interaction() {
        let mem = make_memory();
        let calls = vec![autoagents_llm::ToolCall {
            id: "t1".to_string(),
            call_type: "function".to_string(),
            function: autoagents_llm::FunctionCall {
                name: "tool".to_string(),
                arguments: "{}".to_string(),
            },
        }];
        let results = vec![crate::tool::ToolCallResult {
            tool_name: "tool".to_string(),
            success: true,
            arguments: serde_json::json!({}),
            result: serde_json::json!("ok"),
        }];
        MemoryHelper::store_tool_interaction(&Some(mem.clone()), &calls, &results, "text").await;
        let stored = mem.lock().await.recall("", None).await.unwrap();
        assert_eq!(stored.len(), 2);
    }

    #[tokio::test]
    async fn test_store_tool_interaction_none() {
        MemoryHelper::store_tool_interaction(&None, &[], &[], "text").await;
    }

    #[tokio::test]
    async fn test_recall_messages_empty() {
        let mem = make_memory();
        let messages = MemoryHelper::recall_messages(&Some(mem)).await;
        assert!(messages.is_empty());
    }

    #[tokio::test]
    async fn test_recall_messages_populated() {
        let mem = make_memory();
        MemoryHelper::store_message(
            &Some(mem.clone()),
            ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: "msg1".to_string(),
            },
        )
        .await;
        MemoryHelper::store_message(
            &Some(mem.clone()),
            ChatMessage {
                role: ChatRole::Assistant,
                message_type: MessageType::Text,
                content: "msg2".to_string(),
            },
        )
        .await;
        let messages = MemoryHelper::recall_messages(&Some(mem)).await;
        assert_eq!(messages.len(), 2);
    }

    #[tokio::test]
    async fn test_recall_messages_none() {
        let messages = MemoryHelper::recall_messages(&None).await;
        assert!(messages.is_empty());
    }
}
