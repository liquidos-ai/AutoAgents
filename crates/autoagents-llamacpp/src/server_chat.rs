use crate::{config::LlamaCppConfig, error::LlamaCppProviderError};
use autoagents_llm::chat::{ChatMessage, ChatRole, MessageType};
use autoagents_llm::{FunctionCall, ToolCall};
use serde_json::{Map, Value, json};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ServerChatToolCall {
    pub(crate) name: String,
    pub(crate) arguments: String,
    pub(crate) id: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ServerChatContentPart {
    pub(crate) part_type: String,
    pub(crate) text: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ServerChatMessage {
    pub(crate) role: String,
    pub(crate) content: String,
    pub(crate) content_parts: Vec<ServerChatContentPart>,
    pub(crate) tool_calls: Vec<ServerChatToolCall>,
    pub(crate) reasoning_content: String,
    pub(crate) tool_name: String,
    pub(crate) tool_call_id: String,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ServerTemplateCaps {
    pub(crate) supports_string_content: bool,
    pub(crate) supports_typed_content: bool,
    pub(crate) supports_system_role: bool,
    pub(crate) supports_tool_calls: bool,
    pub(crate) supports_object_arguments: bool,
    pub(crate) gpt_oss_reasoning_compat: bool,
    pub(crate) ministral_reasoning_blocks_compat: bool,
    pub(crate) lfm2_reasoning_compat: bool,
    pub(crate) gemma4_tool_response_compat: bool,
}

impl Default for ServerTemplateCaps {
    fn default() -> Self {
        Self {
            supports_string_content: true,
            supports_typed_content: true,
            supports_system_role: true,
            supports_tool_calls: false,
            supports_object_arguments: false,
            gpt_oss_reasoning_compat: false,
            ministral_reasoning_blocks_compat: false,
            lfm2_reasoning_compat: false,
            gemma4_tool_response_compat: false,
        }
    }
}

impl ServerChatMessage {
    #[allow(dead_code)]
    pub(crate) fn render_content(&self, delimiter: &str) -> String {
        if !self.content.is_empty() {
            return self.content.clone();
        }

        let blocks = self
            .content_parts
            .iter()
            .filter(|part| part.part_type == "text")
            .map(|part| part.text.clone())
            .collect::<Vec<_>>();
        blocks.join(delimiter)
    }

    fn to_json_oaicompat(&self, concat_typed_text: bool) -> Value {
        let mut object = Map::new();
        object.insert("role".to_string(), Value::String(self.role.clone()));

        if concat_typed_text || self.contains_media_marker() {
            object.insert(
                "content".to_string(),
                Value::String(self.render_content_for_template_concat()),
            );
        } else if !self.content_parts.is_empty() {
            let mut content = Vec::new();
            if !self.content.is_empty() {
                content.push(json!({
                    "type": "text",
                    "text": self.content,
                }));
            }
            content.extend(self.content_parts.iter().map(|part| {
                json!({
                    "type": part.part_type,
                    "text": part.text,
                })
            }));
            object.insert("content".to_string(), Value::Array(content));
        } else if !self.content.is_empty() {
            object.insert("content".to_string(), Value::String(self.content.clone()));
        } else {
            object.insert("content".to_string(), Value::String(String::default()));
        }

        if !self.tool_calls.is_empty() {
            object.insert(
                "tool_calls".to_string(),
                Value::Array(
                    self.tool_calls
                        .iter()
                        .map(|tool_call| {
                            let mut call = Map::new();
                            if !tool_call.id.is_empty() {
                                call.insert("id".to_string(), Value::String(tool_call.id.clone()));
                            }
                            call.insert("type".to_string(), Value::String("function".to_string()));
                            call.insert(
                                "function".to_string(),
                                json!({
                                    "name": tool_call.name,
                                    "arguments": tool_call.arguments,
                                }),
                            );
                            Value::Object(call)
                        })
                        .collect(),
                ),
            );
        }

        if !self.reasoning_content.is_empty() {
            object.insert(
                "reasoning_content".to_string(),
                Value::String(self.reasoning_content.clone()),
            );
        }
        if !self.tool_name.is_empty() {
            object.insert("name".to_string(), Value::String(self.tool_name.clone()));
        }
        if !self.tool_call_id.is_empty() {
            object.insert(
                "tool_call_id".to_string(),
                Value::String(self.tool_call_id.clone()),
            );
        }

        Value::Object(object)
    }

    fn contains_media_marker(&self) -> bool {
        self.content_parts
            .iter()
            .any(|part| part.part_type == "media_marker")
    }

    fn render_content_for_template_concat(&self) -> String {
        if !self.content.is_empty() {
            return self.content.clone();
        }

        let mut text = String::default();
        let mut last_was_media_marker = false;
        for part in &self.content_parts {
            match part.part_type.as_str() {
                "text" => {
                    if !last_was_media_marker && !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(&part.text);
                    last_was_media_marker = false;
                }
                "media_marker" => {
                    text.push_str(&part.text);
                    last_was_media_marker = true;
                }
                _ => {}
            }
        }
        text
    }
}

pub(crate) fn messages_from_autoagents(
    config: &LlamaCppConfig,
    messages: &[ChatMessage],
) -> Result<Vec<ServerChatMessage>, LlamaCppProviderError> {
    let mut values = Vec::new();
    if let Some(system_prompt) = config.system_prompt.as_ref()
        && !messages
            .iter()
            .any(|message| matches!(message.role, ChatRole::System))
    {
        values.push(ServerChatMessage {
            role: "system".to_string(),
            content: system_prompt.clone(),
            ..Default::default()
        });
    }

    for message in messages {
        append_message(&mut values, message)?;
    }

    Ok(values)
}

pub(crate) fn render_messages_to_json(
    messages: &[ServerChatMessage],
    caps: ServerTemplateCaps,
) -> Vec<Value> {
    let only_string_accepted = caps.supports_string_content && !caps.supports_typed_content;
    let only_typed_accepted = !caps.supports_string_content && caps.supports_typed_content;

    messages
        .iter()
        .map(|message| {
            let mut value = if only_string_accepted {
                message.to_json_oaicompat(true)
            } else {
                message.to_json_oaicompat(false)
            };

            if only_typed_accepted
                && let Some(object) = value.as_object_mut()
                && let Some(Value::String(content)) = object.get("content").cloned()
            {
                object.insert(
                    "content".to_string(),
                    Value::Array(vec![json!({
                        "type": "text",
                        "text": content,
                    })]),
                );
            }

            value
        })
        .collect()
}

pub(crate) fn apply_server_workarounds(
    messages: &mut Vec<Value>,
    caps: ServerTemplateCaps,
) -> Result<(), LlamaCppProviderError> {
    if caps.gpt_oss_reasoning_compat {
        convert_reasoning_to_gpt_oss_thinking(messages);
    }
    if caps.lfm2_reasoning_compat {
        copy_reasoning_to_thinking(messages);
    }
    if caps.ministral_reasoning_blocks_compat {
        convert_reasoning_to_ministral_blocks(messages);
    }
    if caps.gemma4_tool_response_compat {
        convert_tool_responses_gemma4(messages);
    }
    if !caps.supports_system_role {
        merge_unsupported_system_messages(messages);
    }
    if caps.supports_tool_calls {
        require_non_null_content(messages);
    }
    if caps.supports_object_arguments {
        convert_function_arguments_to_objects(messages)?;
    }
    Ok(())
}

fn convert_reasoning_to_gpt_oss_thinking(messages: &mut [Value]) {
    for message in messages {
        let Some(object) = message.as_object_mut() else {
            continue;
        };
        let Some(Value::String(reasoning_content)) = object.get("reasoning_content").cloned()
        else {
            continue;
        };

        object.insert("thinking".to_string(), Value::String(reasoning_content));
        if object
            .get("tool_calls")
            .and_then(Value::as_array)
            .is_some_and(|calls| !calls.is_empty())
        {
            object.remove("content");
        }
    }
}

fn copy_reasoning_to_thinking(messages: &mut [Value]) {
    for message in messages {
        let Some(object) = message.as_object_mut() else {
            continue;
        };
        let Some(Value::String(reasoning_content)) = object.get("reasoning_content").cloned()
        else {
            continue;
        };
        object.insert("thinking".to_string(), Value::String(reasoning_content));
    }
}

fn convert_reasoning_to_ministral_blocks(messages: &mut [Value]) {
    for message in messages {
        let Some(object) = message.as_object_mut() else {
            continue;
        };
        let role = object.get("role").and_then(Value::as_str);
        if !matches!(role, Some("system" | "assistant")) {
            continue;
        }

        let mut content = Vec::new();
        if let Some(Value::String(reasoning_content)) = object.get("reasoning_content") {
            content.push(json!({
                "type": "thinking",
                "thinking": reasoning_content,
            }));
        }

        match object.get("content") {
            Some(Value::String(text)) => {
                content.push(json!({
                    "type": "text",
                    "text": text,
                }));
            }
            Some(Value::Array(blocks)) => content.extend(blocks.iter().cloned()),
            Some(Value::Null) | None => {}
            Some(value) => {
                content.push(json!({
                    "type": "text",
                    "text": value.to_string(),
                }));
            }
        }

        object.insert("content".to_string(), Value::Array(content));
        object.remove("reasoning_content");
    }
}

fn convert_tool_responses_gemma4(messages: &mut Vec<Value>) {
    let mut result = Vec::with_capacity(messages.len());
    let mut index = 0;

    while index < messages.len() {
        let message = &messages[index];
        if message.get("role").and_then(Value::as_str) != Some("assistant")
            || message
                .get("tool_calls")
                .and_then(Value::as_array)
                .is_none_or(|calls| calls.is_empty())
        {
            result.push(message.clone());
            index += 1;
            continue;
        }

        let (built, next_index) = build_gemma4_model_turn(messages, index);
        result.push(built);
        index = next_index;
    }

    *messages = result;
}

fn build_gemma4_model_turn(messages: &[Value], start: usize) -> (Value, usize) {
    let mut index = start;
    let first = &messages[index];
    let tool_calls = first
        .get("tool_calls")
        .cloned()
        .unwrap_or_else(|| Value::Array(Vec::default()));
    let reasoning_content = first.get("reasoning_content").cloned();
    let tool_call_names = tool_calls
        .as_array()
        .map(|calls| {
            calls
                .iter()
                .map(|call| {
                    call.get("function")
                        .and_then(|function| function.get("name"))
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string()
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    index += 1;

    let mut tool_responses = Vec::new();
    while index < messages.len()
        && messages[index].get("role").and_then(Value::as_str) == Some("tool")
    {
        let tool_message = &messages[index];
        let response = parse_gemma4_tool_response_content(tool_message.get("content"));
        let response_index = tool_responses.len();
        let name = tool_call_names
            .get(response_index)
            .filter(|name| !name.is_empty())
            .cloned()
            .or_else(|| {
                tool_message
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            })
            .unwrap_or_default();
        tool_responses.push(json!({
            "name": name,
            "response": response,
        }));
        index += 1;
    }

    let mut content = None;
    if index < messages.len()
        && messages[index].get("role").and_then(Value::as_str) == Some("assistant")
        && messages[index]
            .get("tool_calls")
            .and_then(Value::as_array)
            .is_none_or(|calls| calls.is_empty())
        && has_gemma4_content(&messages[index])
    {
        content = messages[index].get("content").cloned();
        index += 1;
    }

    let mut object = Map::new();
    object.insert("role".to_string(), Value::String("assistant".to_string()));
    object.insert("tool_calls".to_string(), tool_calls);
    if !tool_responses.is_empty() {
        object.insert("tool_responses".to_string(), Value::Array(tool_responses));
    }
    if let Some(content) = content {
        object.insert("content".to_string(), content);
    }
    if let Some(reasoning_content) = reasoning_content {
        object.insert("reasoning_content".to_string(), reasoning_content);
    }

    (Value::Object(object), index)
}

fn parse_gemma4_tool_response_content(content: Option<&Value>) -> Value {
    match content {
        Some(Value::String(text)) => {
            serde_json::from_str::<Value>(text).unwrap_or_else(|_| Value::String(text.clone()))
        }
        Some(value) => value.clone(),
        None => Value::Null,
    }
}

fn has_gemma4_content(message: &Value) -> bool {
    match message.get("content") {
        Some(Value::String(text)) => !text.is_empty(),
        Some(Value::Array(items)) => !items.is_empty(),
        Some(Value::Null) | None => false,
        Some(_) => true,
    }
}

fn append_message(
    values: &mut Vec<ServerChatMessage>,
    message: &ChatMessage,
) -> Result<(), LlamaCppProviderError> {
    match &message.message_type {
        MessageType::ToolResult(tool_results) => {
            values.extend(tool_results.iter().map(tool_result_message));
        }
        MessageType::ToolUse(tool_calls) => {
            let mut msg = base_message(message);
            msg.role = "assistant".to_string();
            msg.tool_calls = tool_calls
                .iter()
                .map(server_tool_call_from_tool_call)
                .collect::<Result<Vec<_>, _>>()?;
            values.push(msg);
        }
        MessageType::Image(_) => {
            let mut msg = base_message(message);
            append_text_and_media_marker(&mut msg, &message.content, "<image>");
            values.push(msg);
        }
        MessageType::ImageURL(url) => {
            let mut msg = base_message(message);
            append_text_and_media_marker(&mut msg, &message.content, url);
            values.push(msg);
        }
        MessageType::Pdf(_) => {
            let mut msg = base_message(message);
            append_text_and_media_marker(&mut msg, &message.content, "<pdf>");
            values.push(msg);
        }
        MessageType::Text => values.push(base_message(message)),
    }

    Ok(())
}

fn base_message(message: &ChatMessage) -> ServerChatMessage {
    let role = match message.role {
        ChatRole::System => "system",
        ChatRole::User => "user",
        ChatRole::Assistant => "assistant",
        ChatRole::Tool => "tool",
    };
    let content = if matches!(
        message.message_type,
        MessageType::Text | MessageType::ToolUse(_)
    ) {
        message.content.clone()
    } else {
        String::default()
    };

    ServerChatMessage {
        role: role.to_string(),
        content,
        ..Default::default()
    }
}

fn append_text_and_media_marker(msg: &mut ServerChatMessage, text: &str, marker: &str) {
    if !text.is_empty() {
        msg.content_parts.push(ServerChatContentPart {
            part_type: "text".to_string(),
            text: text.to_string(),
        });
    }
    msg.content_parts.push(ServerChatContentPart {
        part_type: "media_marker".to_string(),
        text: marker.to_string(),
    });
}

fn tool_result_message(tool_call: &ToolCall) -> ServerChatMessage {
    ServerChatMessage {
        role: "tool".to_string(),
        content: tool_call.function.arguments.clone(),
        tool_call_id: tool_call.id.clone(),
        ..Default::default()
    }
}

fn server_tool_call_from_tool_call(
    tool_call: &ToolCall,
) -> Result<ServerChatToolCall, LlamaCppProviderError> {
    if tool_call.call_type != "function" {
        return Err(LlamaCppProviderError::Template(format!(
            "Unsupported tool call type: {}",
            tool_call.call_type
        )));
    }
    if tool_call.function.name.trim().is_empty() {
        return Err(LlamaCppProviderError::Template(
            "Missing tool call name".to_string(),
        ));
    }
    Ok(ServerChatToolCall {
        name: tool_call.function.name.clone(),
        arguments: normalize_function_arguments(&tool_call.function)?,
        id: tool_call.id.clone(),
    })
}

fn normalize_function_arguments(function: &FunctionCall) -> Result<String, LlamaCppProviderError> {
    let arguments = function.arguments.trim();
    if arguments.is_empty() {
        return Ok("{}".to_string());
    }
    if serde_json::from_str::<Value>(arguments).is_err() {
        return Err(LlamaCppProviderError::Template(format!(
            "Tool call arguments for `{}` are not valid JSON",
            function.name
        )));
    }
    Ok(arguments.to_string())
}

fn merge_unsupported_system_messages(messages: &mut Vec<Value>) {
    if messages
        .first()
        .and_then(|message| message.get("role"))
        .and_then(Value::as_str)
        != Some("system")
    {
        return;
    }

    let system = messages.remove(0);
    let Some(system_text) = stringify_content(system.get("content")) else {
        return;
    };
    if system_text.is_empty() {
        return;
    }
    if let Some(first) = messages.first_mut()
        && first.get("role").and_then(Value::as_str) != Some("system")
        && let Some(object) = first.as_object_mut()
    {
        let existing = stringify_content(object.get("content")).unwrap_or_default();
        let content = if existing.is_empty() {
            system_text
        } else {
            format!("{system_text}\n{existing}")
        };
        object.insert("content".to_string(), Value::String(content));
    }
}

fn require_non_null_content(messages: &mut [Value]) {
    for message in messages {
        if let Some(object) = message.as_object_mut()
            && object
                .get("tool_calls")
                .and_then(Value::as_array)
                .is_some_and(|calls| !calls.is_empty())
            && !object.contains_key("content")
        {
            object.insert("content".to_string(), Value::String(String::default()));
        }
    }
}

fn convert_function_arguments_to_objects(
    messages: &mut [Value],
) -> Result<(), LlamaCppProviderError> {
    for message in messages {
        let Some(calls) = message
            .as_object_mut()
            .and_then(|object| object.get_mut("tool_calls"))
            .and_then(Value::as_array_mut)
        else {
            continue;
        };

        for call in calls {
            let Some(arguments) = call
                .get_mut("function")
                .and_then(Value::as_object_mut)
                .and_then(|function| function.get_mut("arguments"))
            else {
                continue;
            };
            if let Some(argument_text) = arguments.as_str() {
                *arguments = serde_json::from_str::<Value>(argument_text).map_err(|err| {
                    LlamaCppProviderError::Template(format!(
                        "Failed to parse tool call arguments as JSON: {err}"
                    ))
                })?;
            }
        }
    }
    Ok(())
}

fn stringify_content(value: Option<&Value>) -> Option<String> {
    match value {
        Some(Value::String(text)) => Some(text.clone()),
        Some(Value::Array(parts)) => {
            let text = parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n\n");
            Some(text)
        }
        Some(Value::Null) | None => None,
        Some(value) => Some(value.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoagents_llm::FunctionCall;

    fn tool_call() -> ToolCall {
        ToolCall {
            id: "call_1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "lookup".to_string(),
                arguments: "{\"q\":\"rust\"}".to_string(),
            },
        }
    }

    #[test]
    fn converts_tool_calls_to_server_messages() {
        let message = ChatMessage {
            role: ChatRole::Assistant,
            message_type: MessageType::ToolUse(vec![tool_call()]),
            content: String::default(),
        };

        let messages = messages_from_autoagents(&LlamaCppConfig::default(), &[message])
            .expect("messages should convert");
        assert_eq!(messages[0].role, "assistant");
        assert_eq!(messages[0].tool_calls[0].name, "lookup");
        assert_eq!(messages[0].tool_calls[0].arguments, "{\"q\":\"rust\"}");
    }

    #[test]
    fn omits_empty_tool_call_id_in_openai_compat_json() {
        let message = ServerChatMessage {
            role: "assistant".to_string(),
            tool_calls: vec![ServerChatToolCall {
                name: "lookup".to_string(),
                arguments: "{}".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };

        let value = message.to_json_oaicompat(false);

        assert_eq!(value["content"], "");
        assert!(value["tool_calls"][0].get("id").is_none());
    }

    #[test]
    fn empty_message_content_defaults_to_empty_string() {
        let message = ServerChatMessage {
            role: "assistant".to_string(),
            ..Default::default()
        };

        let value = message.to_json_oaicompat(false);

        assert_eq!(value["content"], "");
    }

    #[test]
    fn renders_caps_driven_typed_or_string_content() {
        let message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::ImageURL("<image>".to_string()),
            content: "caption".to_string(),
        };
        let messages = messages_from_autoagents(&LlamaCppConfig::default(), &[message])
            .expect("messages should convert");

        let typed = render_messages_to_json(
            &messages,
            ServerTemplateCaps {
                supports_string_content: false,
                supports_typed_content: true,
                ..Default::default()
            },
        );
        assert!(typed[0]["content"].is_array());
        assert_eq!(typed[0]["content"][0]["type"], "text");
        assert_eq!(typed[0]["content"][0]["text"], "caption<image>");

        let string = render_messages_to_json(
            &messages,
            ServerTemplateCaps {
                supports_string_content: true,
                supports_typed_content: false,
                ..Default::default()
            },
        );
        assert_eq!(string[0]["content"], "caption<image>");
    }

    #[test]
    fn text_only_parts_can_render_as_typed_content() {
        let message = ServerChatMessage {
            role: "user".to_string(),
            content_parts: vec![ServerChatContentPart {
                part_type: "text".to_string(),
                text: "caption".to_string(),
            }],
            ..Default::default()
        };

        let typed = render_messages_to_json(
            &[message],
            ServerTemplateCaps {
                supports_string_content: false,
                supports_typed_content: true,
                ..Default::default()
            },
        );

        assert!(typed[0]["content"].is_array());
        assert_eq!(typed[0]["content"][0]["type"], "text");
        assert_eq!(typed[0]["content"][0]["text"], "caption");
    }

    #[test]
    fn applies_server_tool_workarounds() {
        let mut messages = vec![json!({
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": "{\"q\":\"rust\"}"
                }
            }]
        })];

        apply_server_workarounds(
            &mut messages,
            ServerTemplateCaps {
                supports_tool_calls: true,
                supports_object_arguments: true,
                ..Default::default()
            },
        )
        .expect("server workarounds should apply");

        assert_eq!(messages[0]["content"], "");
        assert_eq!(
            messages[0]["tool_calls"][0]["function"]["arguments"]["q"],
            "rust"
        );
    }

    #[test]
    fn object_argument_workaround_rejects_invalid_json_arguments() {
        let mut messages = vec![json!({
            "role": "assistant",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": "{bad"
                }
            }]
        })];

        let err = apply_server_workarounds(
            &mut messages,
            ServerTemplateCaps {
                supports_object_arguments: true,
                ..Default::default()
            },
        )
        .expect_err("invalid object arguments should match server failure");

        assert!(
            err.to_string()
                .contains("Failed to parse tool call arguments as JSON")
        );
    }

    #[test]
    fn tool_content_workaround_only_applies_to_tool_call_messages() {
        let mut messages = vec![
            json!({
                "role": "assistant",
                "content": null,
            }),
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "{}"
                    }
                }]
            }),
            json!({
                "role": "assistant",
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "{}"
                    }
                }]
            }),
        ];

        apply_server_workarounds(
            &mut messages,
            ServerTemplateCaps {
                supports_tool_calls: true,
                ..Default::default()
            },
        )
        .expect("server workarounds should apply");

        assert!(messages[0]["content"].is_null());
        assert!(messages[1]["content"].is_null());
        assert_eq!(messages[2]["content"], "");
    }

    #[test]
    fn gpt_oss_reasoning_workaround_sets_thinking_and_removes_tool_content() {
        let mut messages = vec![json!({
            "role": "assistant",
            "content": "",
            "reasoning_content": "think",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": "{}"
                }
            }]
        })];

        apply_server_workarounds(
            &mut messages,
            ServerTemplateCaps {
                gpt_oss_reasoning_compat: true,
                ..Default::default()
            },
        )
        .expect("server workarounds should apply");

        assert_eq!(messages[0]["thinking"], "think");
        assert!(messages[0].get("content").is_none());
        assert_eq!(messages[0]["reasoning_content"], "think");
    }

    #[test]
    fn lfm2_reasoning_workaround_sets_thinking() {
        let mut messages = vec![json!({
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "think",
        })];

        apply_server_workarounds(
            &mut messages,
            ServerTemplateCaps {
                lfm2_reasoning_compat: true,
                ..Default::default()
            },
        )
        .expect("server workarounds should apply");

        assert_eq!(messages[0]["thinking"], "think");
        assert_eq!(messages[0]["reasoning_content"], "think");
        assert_eq!(messages[0]["content"], "answer");
    }

    #[test]
    fn ministral_reasoning_workaround_converts_content_blocks() {
        let mut messages = vec![
            json!({
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "think",
            }),
            json!({
                "role": "user",
                "content": "question",
                "reasoning_content": "ignored",
            }),
        ];

        apply_server_workarounds(
            &mut messages,
            ServerTemplateCaps {
                ministral_reasoning_blocks_compat: true,
                ..Default::default()
            },
        )
        .expect("server workarounds should apply");

        assert_eq!(messages[0]["content"][0]["type"], "thinking");
        assert_eq!(messages[0]["content"][0]["thinking"], "think");
        assert_eq!(messages[0]["content"][1]["type"], "text");
        assert_eq!(messages[0]["content"][1]["text"], "answer");
        assert!(messages[0].get("reasoning_content").is_none());
        assert_eq!(messages[1]["content"], "question");
        assert_eq!(messages[1]["reasoning_content"], "ignored");
    }

    #[test]
    fn merges_only_leading_system_message_when_system_role_is_unsupported() {
        let mut messages = vec![
            json!({
                "role": "system",
                "content": "policy"
            }),
            json!({
                "role": "user",
                "content": "question"
            }),
            json!({
                "role": "system",
                "content": "later"
            }),
        ];

        apply_server_workarounds(
            &mut messages,
            ServerTemplateCaps {
                supports_system_role: false,
                ..Default::default()
            },
        )
        .expect("server workarounds should apply");

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "policy\nquestion");
        assert_eq!(messages[1]["role"], "system");
    }

    #[test]
    fn converts_old_gemma4_tool_responses_into_model_turn() {
        let mut messages = vec![
            json!({
                "role": "assistant",
                "reasoning_content": "think",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "{\"q\":\"rust\"}"
                    }
                }]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "{\"result\":\"ok\"}"
            }),
            json!({
                "role": "assistant",
                "content": "done"
            }),
        ];

        apply_server_workarounds(
            &mut messages,
            ServerTemplateCaps {
                gemma4_tool_response_compat: true,
                ..Default::default()
            },
        )
        .expect("server workarounds should apply");

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["reasoning_content"], "think");
        assert_eq!(messages[0]["content"], "done");
        assert_eq!(messages[0]["tool_responses"][0]["name"], "lookup");
        assert_eq!(messages[0]["tool_responses"][0]["response"]["result"], "ok");
    }
}
