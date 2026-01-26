use crate::agent::memory::MemoryProvider;
use crate::tool::ToolCallResult;
use autoagents_llm::ToolCall;
use autoagents_llm::chat::{ChatMessage, ChatRole, ImageMime, MessageType};
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
                    content: String::new(),
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
                    message_type: MessageType::Image((mime, data)),
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
