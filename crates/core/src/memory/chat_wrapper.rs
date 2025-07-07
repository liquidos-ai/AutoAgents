//! Chat wrapper that adds memory capabilities to any ChatProvider.

use async_trait::async_trait;
use autoagents_llm::{
    chat::{ChatMessage, ChatProvider, ChatResponse, ChatRole, Tool},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
    LLMProvider,
};
use std::sync::Arc;
use tokio::sync::RwLock;

use super::{MemoryProvider, MessageCondition};

/// Adds transparent long-term memory to any `ChatProvider`.
pub struct ChatWithMemory {
    provider: Arc<dyn LLMProvider>,
    memory: Arc<RwLock<Box<dyn MemoryProvider>>>,
    role: Option<String>,
    role_triggers: Vec<(String, MessageCondition)>,

    max_cycles: Option<u32>,
    cycle_counter: std::sync::Arc<std::sync::atomic::AtomicU32>,
}

impl ChatWithMemory {
    /// Build a new wrapper.
    ///
    /// * `provider` – Underlying LLM backend
    /// * `memory` – Conversation memory store
    /// * `role` – Optional agent role
    /// * `role_triggers` – Reactive rules
    pub fn new(
        provider: Arc<dyn LLMProvider>,
        memory: Arc<RwLock<Box<dyn MemoryProvider>>>,
        role: Option<String>,
        role_triggers: Vec<(String, MessageCondition)>,
        max_cycles: Option<u32>,
    ) -> Self {
        use std::sync::atomic::AtomicU32;

        let wrapper = Self {
            provider,
            memory: memory.clone(),
            role,
            role_triggers: role_triggers.clone(),
            max_cycles,
            cycle_counter: std::sync::Arc::new(AtomicU32::new(0)),
        };

        if !wrapper.role_triggers.is_empty() {
            wrapper.spawn_reactive_listener();
        }

        wrapper
    }

    /// Spawn a background listener that reacts to memory events.
    fn spawn_reactive_listener(&self) {
        let memory = self.memory.clone();
        let provider = self.provider.clone();
        let role_triggers = self.role_triggers.clone();
        let my_role = self.role.clone();
        let max_cycles = self.max_cycles;
        let cycle_counter = self.cycle_counter.clone();

        tokio::spawn(async move {
            let mut receiver = {
                let guard = memory.read().await;
                match guard.get_event_receiver() {
                    Some(r) => r,
                    None => return,
                }
            };

            while let Ok(event) = receiver.recv().await {
                let event_role = if event.msg.role == ChatRole::User {
                    "user"
                } else {
                    &event.role
                };

                if !role_triggers
                    .iter()
                    .any(|(role, cond)| role == event_role && cond.matches(&event))
                {
                    continue;
                }

                // max_cycles guard
                if let Some(max) = max_cycles {
                    if cycle_counter.load(std::sync::atomic::Ordering::Relaxed) >= max {
                        continue;
                    }
                }

                let context = {
                    let guard = memory.read().await;
                    guard.recall("", None).await.unwrap_or_default()
                };

                let response_text = match provider.chat(&context).await {
                    Ok(r) => r.text().map(|s| s.to_string()),
                    Err(e) => {
                        eprintln!("Reactive chat error: {e}");
                        None
                    }
                };

                if let (Some(txt), Some(role)) = (response_text, &my_role) {
                    let msg = ChatMessage::assistant()
                        .content(format!("[{role}] {txt}"))
                        .build();
                    let mut guard = memory.write().await;
                    if let Err(e) = guard.remember_with_role(&msg, role.clone()).await {
                        eprintln!("Memory save error: {e}");
                    }

                    cycle_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        });
    }

    /// Access the wrapped provider.
    pub fn inner(&self) -> &dyn LLMProvider {
        self.provider.as_ref()
    }

    /// Dump the full memory (debugging).
    pub async fn memory_contents(&self) -> Vec<ChatMessage> {
        let guard = self.memory.read().await;
        guard.recall("", None).await.unwrap_or_default()
    }
}

#[async_trait]
impl ChatProvider for ChatWithMemory {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        // Reset cycle counter when receiving a user-originated message
        if messages.iter().any(|m| matches!(m.role, ChatRole::User)) {
            self.cycle_counter
                .store(0, std::sync::atomic::Ordering::Relaxed);
        }
        {
            // record incoming user messages once
            let mut mem = self.memory.write().await;
            for m in messages {
                mem.remember(&m).await?;
            }
        }

        // build context
        let mut context = {
            let mem = self.memory.read().await;
            mem.recall("", None).await?
        };

        // summarization if needed
        let needs_summary = {
            let mem = self.memory.read().await;
            mem.needs_summary()
        };
        if needs_summary {
            let summary = self.provider.summarize_history(&context).await?;
            let mut mem = self.memory.write().await;
            mem.replace_with_summary(summary);
            context = mem.recall("", None).await?;
        }

        context.extend_from_slice(messages);
        let response = self.provider.chat_with_tools(&context, tools).await?;

        // record assistant reply once
        if let Some(text) = response.text() {
            let memory = self.memory.clone();
            let tag = self.role.clone();
            tokio::spawn(async move {
                let mut mem = memory.write().await;
                let formatted = match &tag {
                    Some(r) => format!("[{r}] {text}"),
                    None => text.to_string(),
                };
                let msg = ChatMessage::assistant().content(formatted).build();
                let res = if let Some(r) = tag {
                    mem.remember_with_role(&msg, r).await
                } else {
                    mem.remember(&msg).await
                };
                if let Err(e) = res {
                    eprintln!("Memory save error: {e}");
                }
            });
        }

        Ok(response)
    }

    async fn memory_contents(&self) -> Option<Vec<ChatMessage>> {
        let guard = self.memory.read().await;
        Some(guard.recall("", None).await.unwrap_or_default())
    }
}

#[async_trait]
impl CompletionProvider for ChatWithMemory {
    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        self.provider.complete(req).await
    }
}

#[async_trait]
impl EmbeddingProvider for ChatWithMemory {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        self.provider.embed(input).await
    }
}

#[async_trait]
impl ModelsProvider for ChatWithMemory {}

impl LLMProvider for ChatWithMemory {
    fn tools(&self) -> Option<&[Tool]> {
        self.provider.tools()
    }
}
