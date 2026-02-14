use crate::agent::executor::tool_processor::ToolProcessor;
use crate::agent::memory::MemoryProvider;
use crate::agent::task::Task;
use crate::tool::ToolCallResult;
use autoagents_llm::ToolCall;
use autoagents_llm::chat::{ChatMessage, ChatRole, MessageType};
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::Mutex;

#[cfg(target_arch = "wasm32")]
use futures::lock::Mutex;

/// How to select the memory recall query.
#[derive(Debug, Clone, Copy)]
pub enum RecallQuery {
    /// Recall with an empty query.
    Empty,
    /// Recall using the task prompt as the query.
    Prompt,
}

/// Policy describing how an executor should use memory.
#[derive(Debug, Clone)]
pub struct MemoryPolicy {
    pub recall: bool,
    pub recall_query: RecallQuery,
    pub recall_limit: Option<usize>,
    pub store_user: bool,
    pub store_assistant: bool,
    pub store_tool_interactions: bool,
}

impl MemoryPolicy {
    pub fn basic() -> Self {
        Self {
            recall: true,
            recall_query: RecallQuery::Prompt,
            recall_limit: None,
            store_user: true,
            store_assistant: true,
            store_tool_interactions: true,
        }
    }

    pub fn react() -> Self {
        Self {
            recall: true,
            recall_query: RecallQuery::Prompt,
            recall_limit: None,
            store_user: true,
            store_assistant: true,
            store_tool_interactions: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::memory::SlidingWindowMemory;

    #[test]
    fn test_basic_memory_policy_enables_recall_and_tool_interactions() {
        let policy = MemoryPolicy::basic();
        assert!(policy.recall);
        assert!(matches!(policy.recall_query, RecallQuery::Prompt));
        assert!(policy.store_user);
        assert!(policy.store_assistant);
        assert!(policy.store_tool_interactions);
    }

    #[test]
    fn test_react_memory_policy_fields() {
        let policy = MemoryPolicy::react();
        assert!(policy.recall);
        assert!(matches!(policy.recall_query, RecallQuery::Prompt));
        assert_eq!(policy.recall_limit, None);
        assert!(policy.store_user);
        assert!(policy.store_assistant);
        assert!(policy.store_tool_interactions);
    }

    #[test]
    fn test_memory_adapter_without_memory() {
        let adapter = MemoryAdapter::new(None, MemoryPolicy::basic());
        assert!(!adapter.is_enabled());
    }

    #[test]
    fn test_memory_adapter_with_memory() {
        let mem: Box<dyn MemoryProvider> = Box::new(SlidingWindowMemory::new(10));
        let adapter = MemoryAdapter::new(Some(Arc::new(Mutex::new(mem))), MemoryPolicy::basic());
        assert!(adapter.is_enabled());
    }

    #[tokio::test]
    async fn test_recall_disabled() {
        let mem: Box<dyn MemoryProvider> = Box::new(SlidingWindowMemory::new(10));
        let mut policy = MemoryPolicy::basic();
        policy.recall = false;
        let adapter = MemoryAdapter::new(Some(Arc::new(Mutex::new(mem))), policy);
        let task = Task::new("hello");
        let messages = adapter.recall_messages(&task).await;
        assert!(messages.is_empty());
    }

    #[tokio::test]
    async fn test_recall_no_memory() {
        let adapter = MemoryAdapter::new(None, MemoryPolicy::basic());
        let task = Task::new("hello");
        let messages = adapter.recall_messages(&task).await;
        assert!(messages.is_empty());
    }

    #[tokio::test]
    async fn test_recall_with_prompt_query() {
        let mem: Box<dyn MemoryProvider> = Box::new(SlidingWindowMemory::new(10));
        let adapter = MemoryAdapter::new(Some(Arc::new(Mutex::new(mem))), MemoryPolicy::basic());
        let task = Task::new("test prompt");
        // Empty memory => empty result
        let messages = adapter.recall_messages(&task).await;
        assert!(messages.is_empty());
    }

    #[tokio::test]
    async fn test_recall_with_empty_query() {
        let mem: Box<dyn MemoryProvider> = Box::new(SlidingWindowMemory::new(10));
        let mut policy = MemoryPolicy::basic();
        policy.recall_query = RecallQuery::Empty;
        let adapter = MemoryAdapter::new(Some(Arc::new(Mutex::new(mem))), policy);
        let task = Task::new("test");
        let messages = adapter.recall_messages(&task).await;
        assert!(messages.is_empty());
    }

    #[tokio::test]
    async fn test_store_user_enabled() {
        let mem: Box<dyn MemoryProvider> = Box::new(SlidingWindowMemory::new(10));
        let mem_arc = Arc::new(Mutex::new(mem));
        let adapter = MemoryAdapter::new(Some(mem_arc.clone()), MemoryPolicy::basic());
        let task = Task::new("user message");
        adapter.store_user(&task).await;
        let stored = mem_arc.lock().await.recall("", None).await.unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].content, "user message");
    }

    #[tokio::test]
    async fn test_store_user_disabled() {
        let mem: Box<dyn MemoryProvider> = Box::new(SlidingWindowMemory::new(10));
        let mem_arc = Arc::new(Mutex::new(mem));
        let mut policy = MemoryPolicy::basic();
        policy.store_user = false;
        let adapter = MemoryAdapter::new(Some(mem_arc.clone()), policy);
        let task = Task::new("user message");
        adapter.store_user(&task).await;
        let stored = mem_arc.lock().await.recall("", None).await.unwrap();
        assert!(stored.is_empty());
    }

    #[tokio::test]
    async fn test_store_user_no_memory() {
        let adapter = MemoryAdapter::new(None, MemoryPolicy::basic());
        let task = Task::new("user message");
        // Should not panic
        adapter.store_user(&task).await;
    }

    #[tokio::test]
    async fn test_store_assistant_enabled() {
        let mem: Box<dyn MemoryProvider> = Box::new(SlidingWindowMemory::new(10));
        let mem_arc = Arc::new(Mutex::new(mem));
        let adapter = MemoryAdapter::new(Some(mem_arc.clone()), MemoryPolicy::basic());
        adapter.store_assistant("assistant reply").await;
        let stored = mem_arc.lock().await.recall("", None).await.unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].content, "assistant reply");
    }

    #[tokio::test]
    async fn test_store_assistant_disabled() {
        let mem: Box<dyn MemoryProvider> = Box::new(SlidingWindowMemory::new(10));
        let mem_arc = Arc::new(Mutex::new(mem));
        let mut policy = MemoryPolicy::basic();
        policy.store_assistant = false;
        let adapter = MemoryAdapter::new(Some(mem_arc.clone()), policy);
        adapter.store_assistant("reply").await;
        let stored = mem_arc.lock().await.recall("", None).await.unwrap();
        assert!(stored.is_empty());
    }

    #[tokio::test]
    async fn test_store_tool_interaction_enabled() {
        let mem: Box<dyn MemoryProvider> = Box::new(SlidingWindowMemory::new(10));
        let mem_arc = Arc::new(Mutex::new(mem));
        let adapter = MemoryAdapter::new(Some(mem_arc.clone()), MemoryPolicy::basic());
        let tool_calls = vec![ToolCall {
            id: "tc1".to_string(),
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
        adapter
            .store_tool_interaction(&tool_calls, &results, "text")
            .await;
        let stored = mem_arc.lock().await.recall("", None).await.unwrap();
        // Stores 2 messages: assistant ToolUse + Tool ToolResult
        assert_eq!(stored.len(), 2);
    }

    #[tokio::test]
    async fn test_store_tool_interaction_disabled() {
        let mem: Box<dyn MemoryProvider> = Box::new(SlidingWindowMemory::new(10));
        let mem_arc = Arc::new(Mutex::new(mem));
        let mut policy = MemoryPolicy::basic();
        policy.store_tool_interactions = false;
        let adapter = MemoryAdapter::new(Some(mem_arc.clone()), policy);
        adapter.store_tool_interaction(&[], &[], "text").await;
        let stored = mem_arc.lock().await.recall("", None).await.unwrap();
        assert!(stored.is_empty());
    }

    #[test]
    fn test_adapter_policy_accessor() {
        let policy = MemoryPolicy::react();
        let adapter = MemoryAdapter::new(None, policy);
        assert!(adapter.policy().recall);
        assert!(adapter.policy().store_user);
    }
}

/// Adapter that applies a memory policy to a concrete memory provider.
#[derive(Clone)]
pub struct MemoryAdapter {
    memory: Option<Arc<Mutex<Box<dyn MemoryProvider>>>>,
    policy: MemoryPolicy,
}

impl MemoryAdapter {
    pub fn new(memory: Option<Arc<Mutex<Box<dyn MemoryProvider>>>>, policy: MemoryPolicy) -> Self {
        Self { memory, policy }
    }

    pub fn policy(&self) -> &MemoryPolicy {
        &self.policy
    }

    pub fn is_enabled(&self) -> bool {
        self.memory.is_some()
    }

    pub async fn recall_messages(&self, task: &Task) -> Vec<ChatMessage> {
        if !self.policy.recall {
            return Vec::new();
        }
        let Some(memory) = &self.memory else {
            return Vec::new();
        };
        let query = match self.policy.recall_query {
            RecallQuery::Empty => "",
            RecallQuery::Prompt => task.prompt.as_str(),
        };
        memory
            .lock()
            .await
            .recall(query, self.policy.recall_limit)
            .await
            .unwrap_or_default()
    }

    pub async fn store_user(&self, task: &Task) {
        if !self.policy.store_user {
            return;
        }
        let Some(memory) = &self.memory else {
            return;
        };
        let message = if let Some((mime, data)) = &task.image {
            ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Image(((*mime).into(), data.clone())),
                content: task.prompt.clone(),
            }
        } else {
            ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: task.prompt.clone(),
            }
        };
        let _ = memory.lock().await.remember(&message).await;
    }

    pub async fn store_assistant(&self, response: &str) {
        if !self.policy.store_assistant {
            return;
        }
        let Some(memory) = &self.memory else {
            return;
        };
        let message = ChatMessage {
            role: ChatRole::Assistant,
            message_type: MessageType::Text,
            content: response.to_string(),
        };
        let _ = memory.lock().await.remember(&message).await;
    }

    pub async fn store_tool_interaction(
        &self,
        tool_calls: &[ToolCall],
        tool_results: &[ToolCallResult],
        response_text: &str,
    ) {
        if !self.policy.store_tool_interactions {
            return;
        }
        let Some(memory) = &self.memory else {
            return;
        };
        let mut memory = memory.lock().await;
        let _ = memory
            .remember(&ChatMessage {
                role: ChatRole::Assistant,
                message_type: MessageType::ToolUse(tool_calls.to_vec()),
                content: response_text.to_string(),
            })
            .await;

        let result_tool_calls = ToolProcessor::create_result_tool_calls(tool_calls, tool_results);

        let _ = memory
            .remember(&ChatMessage {
                role: ChatRole::Tool,
                message_type: MessageType::ToolResult(result_tool_calls),
                content: String::default(),
            })
            .await;
    }
}
