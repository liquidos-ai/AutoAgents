//! Type conversions between AutoAgents types and llama.cpp types.

use crate::error::LlamaCppProviderError;
use autoagents_llm::chat::{ChatMessage, ChatResponse, ChatRole, MessageType, Usage};
use autoagents_llm::ToolCall;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use std::fmt;

/// Response wrapper that implements ChatResponse trait.
#[derive(Debug, Clone)]
pub struct LlamaCppResponse {
    pub text: String,
    pub usage: Option<Usage>,
}

impl fmt::Display for LlamaCppResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text)
    }
}

impl ChatResponse for LlamaCppResponse {
    fn text(&self) -> Option<String> {
        Some(self.text.clone())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        None
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

fn build_fallback_prompt(messages: &[ChatMessage]) -> String {
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

/// Build a prompt string using the model chat template or a fallback prompt.
pub(crate) fn build_prompt(
    model: &LlamaModel,
    messages: &[ChatMessage],
    template_override: Option<&str>,
) -> Result<PromptData, LlamaCppProviderError> {
    let mut llama_messages = Vec::with_capacity(messages.len());
    for msg in messages {
        let role = convert_role(&msg.role);
        let content = convert_content(msg);
        let llama_msg = LlamaChatMessage::new(role, content).map_err(|err| {
            LlamaCppProviderError::Template(format!("Invalid chat message for template: {}", err))
        })?;
        llama_messages.push(llama_msg);
    }

    let template = if let Some(template) = template_override {
        Some(LlamaChatTemplate::new(template).map_err(|err| {
            LlamaCppProviderError::Template(format!("Invalid chat template override: {}", err))
        })?)
    } else {
        model.chat_template(None).ok()
    };

    if let Some(template) = template {
        let prompt = model
            .apply_chat_template(&template, &llama_messages, true)
            .map_err(|err| {
                LlamaCppProviderError::Template(format!("Failed to apply chat template: {}", err))
            })?;
        Ok(PromptData {
            prompt,
            add_bos: AddBos::Never,
        })
    } else {
        Ok(PromptData {
            prompt: build_fallback_prompt(messages),
            add_bos: AddBos::Always,
        })
    }
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
