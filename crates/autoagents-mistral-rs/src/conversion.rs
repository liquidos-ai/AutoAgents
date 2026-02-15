//! Type conversions between AutoAgents types and mistral.rs types

use autoagents_llm::chat::{ChatMessage, ChatRole, MessageType};
use autoagents_llm::{FunctionCall, ToolCall};
use mistralrs::{TextMessageRole, TextMessages, ToolCallResponse, VisionMessages};
use std::fmt;

/// Response wrapper that implements ChatResponse trait
#[derive(Debug, Clone)]
pub struct MistralRsResponse {
    pub text: String,
    pub tool_calls: Option<Vec<autoagents_llm::ToolCall>>,
}

impl fmt::Display for MistralRsResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text)
    }
}

impl autoagents_llm::chat::ChatResponse for MistralRsResponse {
    fn text(&self) -> Option<String> {
        Some(self.text.clone())
    }

    fn tool_calls(&self) -> Option<Vec<autoagents_llm::ToolCall>> {
        self.tool_calls.clone()
    }
}

/// Convert mistral.rs ToolCallResponse to AutoAgents ToolCall
pub fn convert_tool_calls(tool_calls: &[ToolCallResponse]) -> Vec<ToolCall> {
    tool_calls
        .iter()
        .map(|tc| ToolCall {
            id: tc.id.clone(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: tc.function.name.clone(),
                arguments: tc.function.arguments.clone(),
            },
        })
        .collect()
}

/// Convert AutoAgents ChatRole to mistral.rs TextMessageRole
pub(crate) fn convert_role(role: &ChatRole) -> TextMessageRole {
    match role {
        ChatRole::System => TextMessageRole::System,
        ChatRole::User => TextMessageRole::User,
        ChatRole::Assistant => TextMessageRole::Assistant,
        ChatRole::Tool => TextMessageRole::User,
    }
}

/// Convert AutoAgents ChatMessages to mistral.rs TextMessages
pub(crate) fn convert_messages(messages: &[ChatMessage]) -> TextMessages {
    let mut text_messages = TextMessages::new();

    for msg in messages {
        let role = convert_role(&msg.role);

        // Handle different message types
        let content = match &msg.message_type {
            MessageType::Text => msg.content.clone(),
            MessageType::Image(_) => {
                // mistral.rs doesn't support images in text models yet
                // Include a placeholder or skip
                format!("[Image: {}]", msg.content)
            }
            MessageType::ImageURL(url) => {
                format!("[Image URL: {}] {}", url, msg.content)
            }
            MessageType::Pdf(_) => {
                format!("[PDF Document] {}", msg.content)
            }
            MessageType::ToolUse(tool_calls) => {
                // Format tool calls as text
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
                format!("{}\n{}", msg.content, tools_str)
            }
            MessageType::ToolResult(tool_results) => {
                // Format tool results as text
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
                format!("{}\n{}", msg.content, results_str)
            }
        };

        text_messages = text_messages.add_message(role, content);
    }

    text_messages
}

/// Convert AutoAgents ChatMessages to mistral.rs VisionMessages (supports images)
///
/// Note: This function needs access to the model for proper image message handling
pub(crate) fn convert_vision_messages(
    messages: &[ChatMessage],
    model: &mistralrs::Model,
) -> Result<VisionMessages, anyhow::Error> {
    let mut vision_messages = VisionMessages::new();

    for msg in messages {
        let role = convert_role(&msg.role);

        // Handle different message types
        match &msg.message_type {
            MessageType::Text => {
                // Text-only messages can use add_message
                vision_messages = vision_messages.add_message(role, msg.content.clone());
            }
            MessageType::Image((_, bytes)) => {
                // Load image from raw bytes
                let image = image::load_from_memory(bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to load image: {}", e))?;

                // Add message with image
                vision_messages =
                    vision_messages.add_image_message(role, &msg.content, vec![image], model)?;
            }
            MessageType::ImageURL(_url) => {
                // For now, just add as text message
                // TODO: Download image from URL if needed
                vision_messages = vision_messages.add_message(
                    role,
                    format!("[Image URL not supported yet] {}", msg.content),
                );
            }
            MessageType::Pdf(_) => {
                vision_messages =
                    vision_messages.add_message(role, format!("[PDF Document] {}", msg.content));
            }
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
                vision_messages =
                    vision_messages.add_message(role, format!("{}\n{}", msg.content, tools_str));
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
                vision_messages =
                    vision_messages.add_message(role, format!("{}\n{}", msg.content, results_str));
            }
        };
    }

    Ok(vision_messages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoagents_llm::chat::{ChatMessage, ChatResponse};
    use mistralrs::RequestLike;

    #[test]
    fn test_convert_role() {
        assert_eq!(
            format!("{:?}", convert_role(&ChatRole::System)),
            format!("{:?}", TextMessageRole::System)
        );
        assert_eq!(
            format!("{:?}", convert_role(&ChatRole::User)),
            format!("{:?}", TextMessageRole::User)
        );
        assert_eq!(
            format!("{:?}", convert_role(&ChatRole::Assistant)),
            format!("{:?}", TextMessageRole::Assistant)
        );
    }

    #[test]
    fn test_convert_messages_basic() {
        let messages = vec![
            ChatMessage::user().content("Hello").build(),
            ChatMessage::assistant().content("Hi there!").build(),
        ];

        let text_messages = convert_messages(&messages);
        // We can't directly inspect TextMessages internals, but we can verify it was created
        // This is mainly a smoke test
        drop(text_messages);
    }

    #[test]
    fn test_mistralrs_response_display() {
        let response = MistralRsResponse {
            text: "Test response".to_string(),
            tool_calls: None,
        };
        assert_eq!(response.to_string(), "Test response");
    }

    #[test]
    fn test_mistralrs_response_chat_response_trait() {
        let response = MistralRsResponse {
            text: "Test response".to_string(),
            tool_calls: None,
        };
        assert_eq!(response.text(), Some("Test response".to_string()));
        assert!(response.tool_calls().is_none());
    }

    #[test]
    fn test_convert_messages_with_system() {
        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                message_type: MessageType::Text,
                content: "You are a helpful assistant.".to_string(),
            },
            ChatMessage::user().content("Hello").build(),
        ];

        let text_messages = convert_messages(&messages);
        drop(text_messages);
    }

    #[test]
    fn test_convert_messages_empty() {
        let messages: Vec<ChatMessage> = vec![];
        let text_messages = convert_messages(&messages);
        drop(text_messages);
    }

    #[test]
    fn test_mistralrs_response_clone() {
        let response = MistralRsResponse {
            text: "Original".to_string(),
            tool_calls: None,
        };
        let cloned = response.clone();
        assert_eq!(response.text, cloned.text);
    }

    #[test]
    fn test_convert_tool_calls_maps_fields() {
        let calls = vec![ToolCallResponse {
            index: 0,
            id: "call-1".to_string(),
            tp: mistralrs::ToolCallType::Function,
            function: mistralrs::CalledFunction {
                name: "lookup".to_string(),
                arguments: "{\"q\":\"x\"}".to_string(),
            },
        }];

        let converted = convert_tool_calls(&calls);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].id, "call-1");
        assert_eq!(converted[0].function.name, "lookup");
        assert_eq!(converted[0].function.arguments, "{\"q\":\"x\"}");
    }

    #[test]
    fn test_convert_messages_tool_use_and_result() {
        let tool_call = ToolCall {
            id: "call-1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "lookup".to_string(),
                arguments: "{\"q\":\"x\"}".to_string(),
            },
        };

        let messages = vec![
            ChatMessage {
                role: ChatRole::Assistant,
                message_type: MessageType::ToolUse(vec![tool_call.clone()]),
                content: "call".to_string(),
            },
            ChatMessage {
                role: ChatRole::Tool,
                message_type: MessageType::ToolResult(vec![tool_call]),
                content: "result".to_string(),
            },
        ];

        let text_messages = convert_messages(&messages);
        let stored = text_messages.messages_ref();
        assert_eq!(stored.len(), 2);
        let first = format!("{:?}", stored[0].get("content"));
        assert!(first.contains("Tool: lookup"));
        let second = format!("{:?}", stored[1].get("content"));
        assert!(second.contains("Tool Result: lookup"));
    }
}
