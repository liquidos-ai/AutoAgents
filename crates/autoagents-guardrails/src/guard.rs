use std::{
    fmt,
    sync::atomic::{AtomicU64, Ordering},
    time::SystemTime,
};

use async_trait::async_trait;
use autoagents_llm::{
    ToolCall,
    chat::{ChatMessage, StructuredOutputFormat, Tool, Usage},
    completion::CompletionRequest,
};
use serde_json::Value;

use crate::policy::{GuardCategory, GuardSeverity};

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);
pub const DEFAULT_REDACTED_TEXT: &str = "[redacted by guardrails]";

/// Immutable metadata attached to each guardrails evaluation.
#[derive(Debug, Clone)]
pub struct GuardContext {
    pub request_id: u64,
    pub operation: GuardOperation,
    pub created_at: SystemTime,
}

impl GuardContext {
    pub fn new(operation: GuardOperation) -> Self {
        Self {
            request_id: REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed),
            operation,
            created_at: SystemTime::now(),
        }
    }
}

/// LLM operation currently evaluated by guardrails.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum GuardOperation {
    Chat,
    ChatWithTools,
    ChatWithWebSearch,
    ChatStream,
    ChatStreamStruct,
    ChatStreamWithTools,
    Complete,
}

impl fmt::Display for GuardOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            GuardOperation::Chat => "chat",
            GuardOperation::ChatWithTools => "chat_with_tools",
            GuardOperation::ChatWithWebSearch => "chat_with_web_search",
            GuardOperation::ChatStream => "chat_stream",
            GuardOperation::ChatStreamStruct => "chat_stream_struct",
            GuardOperation::ChatStreamWithTools => "chat_stream_with_tools",
            GuardOperation::Complete => "complete",
        };
        f.write_str(value)
    }
}

/// A rule hit returned by a guard implementation.
#[derive(Debug, Clone)]
pub struct GuardViolation {
    pub rule_id: String,
    pub category: GuardCategory,
    pub severity: GuardSeverity,
    pub message: String,
    pub metadata: Option<Value>,
}

impl GuardViolation {
    pub fn new(
        rule_id: impl Into<String>,
        category: GuardCategory,
        severity: GuardSeverity,
        message: impl Into<String>,
    ) -> Self {
        Self {
            rule_id: rule_id.into(),
            category,
            severity,
            message: message.into(),
            metadata: None,
        }
    }

    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Decision returned by each guard invocation.
#[derive(Debug, Clone)]
pub enum GuardDecision {
    /// No issue found.
    Pass,
    /// Guard mutated payload in-place and wants processing to continue.
    Modify { violation: Option<GuardViolation> },
    /// Guard found a violation and wants policy handling.
    Reject(GuardViolation),
}

impl GuardDecision {
    pub fn pass() -> Self {
        Self::Pass
    }

    pub fn modify() -> Self {
        Self::Modify { violation: None }
    }

    pub fn reject(violation: GuardViolation) -> Self {
        Self::Reject(violation)
    }
}

/// Error emitted by a guard implementation.
#[derive(Debug, Clone)]
pub struct GuardError {
    pub message: String,
}

impl GuardError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for GuardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for GuardError {}

/// Trait implemented by request/input guardrails.
#[async_trait]
pub trait InputGuard: Send + Sync + 'static {
    /// Stable identifier used for diagnostics.
    fn name(&self) -> &'static str;

    /// Inspect and optionally mutate input payload.
    async fn inspect(
        &self,
        input: &mut GuardedInput,
        context: &GuardContext,
    ) -> Result<GuardDecision, GuardError>;
}

/// Trait implemented by response/output guardrails.
#[async_trait]
pub trait OutputGuard: Send + Sync + 'static {
    /// Stable identifier used for diagnostics.
    fn name(&self) -> &'static str;

    /// Inspect and optionally mutate output payload.
    async fn inspect(
        &self,
        output: &mut GuardedOutput,
        context: &GuardContext,
    ) -> Result<GuardDecision, GuardError>;
}

/// Owned chat payload used by input guards.
#[derive(Debug, Clone)]
pub struct ChatGuardInput {
    pub messages: Vec<ChatMessage>,
    pub tools: Option<Vec<Tool>>,
    pub json_schema: Option<StructuredOutputFormat>,
}

/// Owned completion payload used by input guards.
#[derive(Debug, Clone)]
pub struct CompletionGuardInput {
    pub request: CompletionRequest,
    pub json_schema: Option<StructuredOutputFormat>,
}

/// Owned web search payload used by input guards.
#[derive(Debug, Clone)]
pub struct WebSearchGuardInput {
    pub input: String,
}

/// Input payload union passed through input guards.
#[derive(Debug, Clone)]
pub enum GuardedInput {
    Chat(ChatGuardInput),
    Completion(CompletionGuardInput),
    WebSearch(WebSearchGuardInput),
}

impl GuardedInput {
    /// Redact every text field using the default placeholder.
    pub fn redact_all(&mut self) {
        self.redact_with(DEFAULT_REDACTED_TEXT);
    }

    /// Redact every text field with a custom replacement string.
    pub fn redact_with(&mut self, replacement: &str) {
        match self {
            GuardedInput::Chat(chat) => {
                for message in &mut chat.messages {
                    message.content = replacement.to_string();
                }
            }
            GuardedInput::Completion(completion) => {
                completion.request.prompt = replacement.to_string();
            }
            GuardedInput::WebSearch(web) => {
                web.input = replacement.to_string();
            }
        }
    }
}

/// Materialized chat output payload used by output guards.
#[derive(Debug, Clone)]
pub struct ChatGuardOutput {
    pub text: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub thinking: Option<String>,
    pub usage: Option<Usage>,
}

/// Materialized completion output payload used by output guards.
#[derive(Debug, Clone)]
pub struct CompletionGuardOutput {
    pub text: String,
}

/// Output payload union passed through output guards.
#[derive(Debug, Clone)]
pub enum GuardedOutput {
    Chat(ChatGuardOutput),
    Completion(CompletionGuardOutput),
}

impl GuardedOutput {
    /// Redact output content using the default placeholder and clear optional
    /// chat-specific metadata.
    pub fn redact_all(&mut self) {
        self.redact_with(DEFAULT_REDACTED_TEXT);
    }

    /// Redact output content with a custom replacement and clear optional
    /// chat-specific metadata.
    pub fn redact_with(&mut self, replacement: &str) {
        match self {
            GuardedOutput::Chat(chat) => {
                chat.text = Some(replacement.to_string());
                chat.thinking = None;
                chat.tool_calls = None;
            }
            GuardedOutput::Completion(completion) => {
                completion.text = replacement.to_string();
            }
        }
    }

    /// Redact only text fields while preserving non-text chat metadata.
    pub fn redact_text_only(&mut self) {
        match self {
            GuardedOutput::Chat(chat) => {
                chat.text = Some(DEFAULT_REDACTED_TEXT.to_string());
            }
            GuardedOutput::Completion(completion) => {
                completion.text = DEFAULT_REDACTED_TEXT.to_string();
            }
        }
    }
}
