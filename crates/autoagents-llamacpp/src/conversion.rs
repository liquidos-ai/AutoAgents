//! Type conversions between AutoAgents types and llama.cpp types.

use crate::error::LlamaCppProviderError;
use autoagents_llm::chat::{ChatMessage, ChatResponse, ChatRole, MessageType, Usage};
use autoagents_llm::ToolCall;
use llama_cpp_2::model::AddBos;
use serde_json::{json, Value};
use std::fmt;

/// Response wrapper that implements ChatResponse trait.
#[derive(Debug, Clone)]
pub struct LlamaCppResponse {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub usage: Option<Usage>,
}

impl fmt::Display for LlamaCppResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.content, &self.tool_calls) {
            (Some(content), Some(tool_calls)) => {
                for tool_call in tool_calls {
                    write!(f, "{tool_call}")?;
                }
                write!(f, "{content}")
            }
            (Some(content), None) => write!(f, "{content}"),
            (None, Some(tool_calls)) => {
                for tool_call in tool_calls {
                    write!(f, "{tool_call}")?;
                }
                Ok(())
            }
            (None, None) => write!(f, ""),
        }
    }
}

impl ChatResponse for LlamaCppResponse {
    fn text(&self) -> Option<String> {
        self.content.clone()
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.tool_calls.clone()
    }

    fn usage(&self) -> Option<Usage> {
        self.usage.clone()
    }
}

pub(crate) struct PromptData {
    pub prompt: String,
    pub add_bos: AddBos,
}

fn convert_role(role: &ChatRole) -> String {
    match role {
        ChatRole::System => "system".to_string(),
        ChatRole::User => "user".to_string(),
        ChatRole::Assistant => "assistant".to_string(),
        ChatRole::Tool => "user".to_string(),
    }
}

fn convert_content(message: &ChatMessage) -> String {
    match &message.message_type {
        MessageType::Text => message.content.clone(),
        MessageType::Image(_) => format!("[Image: {}]", message.content),
        MessageType::ImageURL(url) => format!("[Image URL: {}] {}", url, message.content),
        MessageType::Pdf(_) => format!("[PDF Document] {}", message.content),
        MessageType::ToolUse(tool_calls) => {
            let tools_str = tool_calls
                .iter()
                .map(|tc| {
                    format!(
                        "Tool: {} with args: {}",
                        tc.function.name, tc.function.arguments
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            format!("{}\n{}", message.content, tools_str)
        }
        MessageType::ToolResult(tool_results) => {
            let results_str = tool_results
                .iter()
                .map(|tc| {
                    format!(
                        "Tool Result: {} = {}",
                        tc.function.name, tc.function.arguments
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            format!("{}\n{}", message.content, results_str)
        }
    }
}

pub(crate) fn build_fallback_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let role = match msg.role {
            ChatRole::System => "System",
            ChatRole::User => "User",
            ChatRole::Assistant => "Assistant",
            ChatRole::Tool => "Tool",
        };
        let content = convert_content(msg);
        prompt.push_str(role);
        prompt.push_str(": ");
        prompt.push_str(&content);
        prompt.push('\n');
    }
    prompt.push_str("Assistant: ");
    prompt
}

fn build_openai_message_value(message: &ChatMessage) -> Result<Value, LlamaCppProviderError> {
    if let MessageType::ToolUse(tool_calls) = &message.message_type {
        let mut tool_values = Vec::with_capacity(tool_calls.len());
        for call in tool_calls {
            let value = serde_json::to_value(call).map_err(|err| {
                LlamaCppProviderError::Template(format!("Failed to serialize tool call: {}", err))
            })?;
            tool_values.push(value);
        }
        let mut obj = serde_json::Map::new();
        obj.insert("role".to_string(), json!("assistant"));
        if !message.content.trim().is_empty() {
            obj.insert("content".to_string(), json!(message.content));
        }
        obj.insert("tool_calls".to_string(), Value::Array(tool_values));
        return Ok(Value::Object(obj));
    }

    let role = convert_role(&message.role);
    let content = convert_content(message);
    Ok(json!({
        "role": role,
        "content": content,
    }))
}

pub(crate) fn build_openai_messages_json(
    messages: &[ChatMessage],
) -> Result<String, LlamaCppProviderError> {
    let mut openai_messages = Vec::new();
    for message in messages {
        if let MessageType::ToolResult(tool_results) = &message.message_type {
            for result in tool_results {
                openai_messages.push(json!({
                    "role": "tool",
                    "tool_call_id": result.id,
                    "content": result.function.arguments,
                }));
            }
            continue;
        }
        openai_messages.push(build_openai_message_value(message)?);
    }

    serde_json::to_string(&openai_messages).map_err(|err| {
        LlamaCppProviderError::Template(format!("Failed to serialize messages: {}", err))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoagents_llm::chat::ChatMessage;

    #[test]
    fn test_fallback_prompt() {
        let messages = vec![ChatMessage::user().content("Hello").build()];
        let prompt = build_fallback_prompt(&messages);
        assert!(prompt.contains("User: Hello"));
        assert!(prompt.contains("Assistant:"));
    }
}
