use std::collections::HashMap;
use std::fmt;

use async_trait::async_trait;
use futures::stream::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{error::LLMError, ToolCall};

/// Role of a participant in a chat conversation.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub enum ChatRole {
    /// System
    System,
    /// The user/human participant in the conversation
    User,
    /// The AI assistant participant in the conversation
    Assistant,
    /// Tool/function response
    Tool,
}

/// The supported MIME type of an image.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[non_exhaustive]
pub enum ImageMime {
    /// JPEG image
    JPEG,
    /// PNG image
    PNG,
    /// GIF image
    GIF,
    /// WebP image
    WEBP,
}

impl ImageMime {
    pub fn mime_type(&self) -> &'static str {
        match self {
            ImageMime::JPEG => "image/jpeg",
            ImageMime::PNG => "image/png",
            ImageMime::GIF => "image/gif",
            ImageMime::WEBP => "image/webp",
        }
    }
}

/// The type of a message in a chat conversation.
#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize, Serialize)]
pub enum MessageType {
    /// A text message
    #[default]
    Text,
    /// An image message
    Image((ImageMime, Vec<u8>)),
    /// PDF message
    Pdf(Vec<u8>),
    /// An image URL message
    ImageURL(String),
    /// A tool use
    ToolUse(Vec<ToolCall>),
    /// Tool result
    ToolResult(Vec<ToolCall>),
}

/// The type of reasoning effort for a message in a chat conversation.
pub enum ReasoningEffort {
    /// Low reasoning effort
    Low,
    /// Medium reasoning effort
    Medium,
    /// High reasoning effort
    High,
}

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// The role of who sent this message (user or assistant)
    pub role: ChatRole,
    /// The type of the message (text, image, audio, video, etc)
    pub message_type: MessageType,
    /// The text content of the message
    pub content: String,
}

/// Represents a parameter in a function tool
#[derive(Debug, Clone, Serialize)]
pub struct ParameterProperty {
    /// The type of the parameter (e.g. "string", "number", "array", etc)
    #[serde(rename = "type")]
    pub property_type: String,
    /// Description of what the parameter does
    pub description: String,
    /// When type is "array", this defines the type of the array items
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<ParameterProperty>>,
    /// When type is "enum", this defines the possible values for the parameter
    #[serde(skip_serializing_if = "Option::is_none", rename = "enum")]
    pub enum_list: Option<Vec<String>>,
}

/// Represents the parameters schema for a function tool
#[derive(Debug, Clone, Serialize)]
pub struct ParametersSchema {
    /// The type of the parameters object (usually "object")
    #[serde(rename = "type")]
    pub schema_type: String,
    /// Map of parameter names to their properties
    pub properties: HashMap<String, ParameterProperty>,
    /// List of required parameter names
    pub required: Vec<String>,
}

/// Represents a function definition for a tool.
///
/// The `parameters` field stores the JSON Schema describing the function
/// arguments.  It is kept as a raw `serde_json::Value` to allow arbitrary
/// complexity (nested arrays/objects, `oneOf`, etc.) without requiring a
/// bespoke Rust structure.
///
/// Builder helpers can still generate simple schemas automatically, but the
/// user may also provide any valid schema directly.
#[derive(Debug, Clone, Serialize)]
pub struct FunctionTool {
    /// Name of the function
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// JSON Schema describing the parameters
    pub parameters: Value,
}

/// Defines rules for structured output responses based on [OpenAI's structured output requirements](https://platform.openai.com/docs/api-reference/chat/create#chat-create-response_format).
/// Individual providers may have additional requirements or restrictions, but these should be handled by each provider's backend implementation.
///
/// If you plan on deserializing into this struct, make sure the source text has a `"name"` field, since that's technically the only thing required by OpenAI.
///
/// ## Example
///
/// ```
/// use autoagents_llm::chat::StructuredOutputFormat;
/// use serde_json::json;
///
/// let response_format = r#"
///     {
///         "name": "Student",
///         "description": "A student object",
///         "schema": {
///             "type": "object",
///             "properties": {
///                 "name": {
///                     "type": "string"
///                 },
///                 "age": {
///                     "type": "integer"
///                 },
///                 "is_student": {
///                     "type": "boolean"
///                 }
///             },
///             "required": ["name", "age", "is_student"]
///         }
///     }
/// "#;
/// let structured_output: StructuredOutputFormat = serde_json::from_str(response_format).unwrap();
/// assert_eq!(structured_output.name, "Student");
/// assert_eq!(structured_output.description, Some("A student object".to_string()));
/// ```
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]

pub struct StructuredOutputFormat {
    /// Name of the schema
    pub name: String,
    /// The description of the schema
    pub description: Option<String>,
    /// The JSON schema for the structured output
    pub schema: Option<Value>,
    /// Whether to enable strict schema adherence
    pub strict: Option<bool>,
}

/// Represents a tool that can be used in chat
#[derive(Debug, Clone, Serialize)]
pub struct Tool {
    /// The type of tool (e.g. "function")
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function definition if this is a function tool
    pub function: FunctionTool,
}

/// Tool choice determines how the LLM uses available tools.
/// The behavior is standardized across different LLM providers.
#[derive(Debug, Clone, Default)]
pub enum ToolChoice {
    /// Model can use any tool, but it must use at least one.
    /// This is useful when you want to force the model to use tools.
    Any,

    /// Model can use any tool, and may elect to use none.
    /// This is the default behavior and gives the model flexibility.
    #[default]
    Auto,

    /// Model must use the specified tool and only the specified tool.
    /// The string parameter is the name of the required tool.
    /// This is useful when you want the model to call a specific function.
    Tool(String),

    /// Explicitly disables the use of tools.
    /// The model will not use any tools even if they are provided.
    None,
}

impl Serialize for ToolChoice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            ToolChoice::Any => serializer.serialize_str("required"),
            ToolChoice::Auto => serializer.serialize_str("auto"),
            ToolChoice::None => serializer.serialize_str("none"),
            ToolChoice::Tool(name) => {
                use serde::ser::SerializeMap;

                // For tool_choice: {"type": "function", "function": {"name": "function_name"}}
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "function")?;

                // Inner function object
                let mut function_obj = std::collections::HashMap::new();
                function_obj.insert("name", name.as_str());

                map.serialize_entry("function", &function_obj)?;
                map.end()
            }
        }
    }
}

pub trait ChatResponse: std::fmt::Debug + std::fmt::Display + Send + Sync {
    fn text(&self) -> Option<String>;
    fn tool_calls(&self) -> Option<Vec<ToolCall>>;
    fn thinking(&self) -> Option<String> {
        None
    }
}

/// Trait for providers that support chat-style interactions.
#[async_trait]
pub trait ChatProvider: Sync + Send {
    /// Sends a chat request to the provider with a sequence of messages.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history as a slice of chat messages
    ///
    /// # Returns
    ///
    /// The provider's response text or an error
    async fn chat(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.chat_with_tools(messages, None, json_schema).await
    }

    /// Sends a chat request to the provider with a sequence of messages and tools.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history as a slice of chat messages
    /// * `tools` - Optional slice of tools to use in the chat
    ///
    /// # Returns
    ///
    /// The provider's response text or an error
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError>;

    /// Sends a streaming chat request to the provider with a sequence of messages.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history as a slice of chat messages
    ///
    /// # Returns
    ///
    /// A stream of text tokens or an error
    async fn chat_stream(
        &self,
        _messages: &[ChatMessage],
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
    {
        Err(LLMError::Generic(
            "Streaming not supported for this provider".to_string(),
        ))
    }

    /// Sends a streaming chat request to the provider with a sequence of messages and tools.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history as a slice of chat messages
    /// * `tools` - Optional slice of tools to use in the chat
    ///
    /// # Returns
    ///
    /// A stream of text tokens or an error
    async fn chat_stream_with_tools(
        &self,
        messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
    {
        // Default implementation falls back to regular chat_stream (ignoring tools)
        self.chat_stream(messages).await
    }

    /// Get current memory contents if provider supports memory
    async fn memory_contents(&self) -> Option<Vec<ChatMessage>> {
        None
    }

    /// Summarizes a conversation history into a concise 2-3 sentence summary
    ///
    /// # Arguments
    /// * `msgs` - The conversation messages to summarize
    ///
    /// # Returns
    /// A string containing the summary or an error if summarization fails
    async fn summarize_history(&self, msgs: &[ChatMessage]) -> Result<String, LLMError> {
        let prompt = format!(
            "Summarize in 2-3 sentences:\n{}",
            msgs.iter()
                .map(|m| format!("{:?}: {}", m.role, m.content))
                .collect::<Vec<_>>()
                .join("\n"),
        );
        let req = [ChatMessage::user().content(prompt).build()];
        self.chat(&req, None)
            .await?
            .text()
            .ok_or(LLMError::Generic("no text in summary response".into()))
    }
}

impl fmt::Display for ReasoningEffort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReasoningEffort::Low => write!(f, "low"),
            ReasoningEffort::Medium => write!(f, "medium"),
            ReasoningEffort::High => write!(f, "high"),
        }
    }
}

impl ChatMessage {
    /// Create a new builder for a user message
    pub fn user() -> ChatMessageBuilder {
        ChatMessageBuilder::new(ChatRole::User)
    }

    /// Create a new builder for an assistant message
    pub fn assistant() -> ChatMessageBuilder {
        ChatMessageBuilder::new(ChatRole::Assistant)
    }
}

/// Builder for ChatMessage
#[derive(Debug)]
pub struct ChatMessageBuilder {
    role: ChatRole,
    message_type: MessageType,
    content: String,
}

impl ChatMessageBuilder {
    /// Create a new ChatMessageBuilder with specified role
    pub fn new(role: ChatRole) -> Self {
        Self {
            role,
            message_type: MessageType::default(),
            content: String::new(),
        }
    }

    /// Set the message content
    pub fn content<S: Into<String>>(mut self, content: S) -> Self {
        self.content = content.into();
        self
    }

    /// Set the message type as Image
    pub fn image(mut self, image_mime: ImageMime, raw_bytes: Vec<u8>) -> Self {
        self.message_type = MessageType::Image((image_mime, raw_bytes));
        self
    }

    /// Set the message type as Image
    pub fn pdf(mut self, raw_bytes: Vec<u8>) -> Self {
        self.message_type = MessageType::Pdf(raw_bytes);
        self
    }

    /// Set the message type as ImageURL
    pub fn image_url(mut self, url: impl Into<String>) -> Self {
        self.message_type = MessageType::ImageURL(url.into());
        self
    }

    /// Set the message type as ToolUse
    pub fn tool_use(mut self, tools: Vec<ToolCall>) -> Self {
        self.message_type = MessageType::ToolUse(tools);
        self
    }

    /// Set the message type as ToolResult
    pub fn tool_result(mut self, tools: Vec<ToolCall>) -> Self {
        self.message_type = MessageType::ToolResult(tools);
        self
    }

    /// Build the ChatMessage
    pub fn build(self) -> ChatMessage {
        ChatMessage {
            role: self.role,
            message_type: self.message_type,
            content: self.content,
        }
    }
}

/// Creates a Server-Sent Events (SSE) stream from an HTTP response.
///
/// # Arguments
///
/// * `response` - The HTTP response from the streaming API
/// * `parser` - Function to parse each SSE chunk into optional text content
///
/// # Returns
///
/// A pinned stream of text tokens or an error
#[allow(dead_code)]
pub(crate) fn create_sse_stream<F>(
    response: reqwest::Response,
    parser: F,
) -> std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>
where
    F: Fn(&str) -> Result<Option<String>, LLMError> + Send + 'static,
{
    let stream = response
        .bytes_stream()
        .map(move |chunk| match chunk {
            Ok(bytes) => {
                let text = String::from_utf8_lossy(&bytes);
                parser(&text)
            }
            Err(e) => Err(LLMError::HttpError(e.to_string())),
        })
        .filter_map(|result| async move {
            match result {
                Ok(Some(content)) => Some(Ok(content)),
                Ok(None) => None,
                Err(e) => Some(Err(e)),
            }
        });

    Box::pin(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::LLMError;
    use serde_json::json;

    #[test]
    fn test_chat_role_serialization() {
        let user_role = ChatRole::User;
        let serialized = serde_json::to_string(&user_role).unwrap();
        assert_eq!(serialized, "\"User\"");

        let assistant_role = ChatRole::Assistant;
        let serialized = serde_json::to_string(&assistant_role).unwrap();
        assert_eq!(serialized, "\"Assistant\"");

        let system_role = ChatRole::System;
        let serialized = serde_json::to_string(&system_role).unwrap();
        assert_eq!(serialized, "\"System\"");

        let tool_role = ChatRole::Tool;
        let serialized = serde_json::to_string(&tool_role).unwrap();
        assert_eq!(serialized, "\"Tool\"");
    }

    #[test]
    fn test_chat_role_deserialization() {
        let deserialized: ChatRole = serde_json::from_str("\"User\"").unwrap();
        assert_eq!(deserialized, ChatRole::User);

        let deserialized: ChatRole = serde_json::from_str("\"Assistant\"").unwrap();
        assert_eq!(deserialized, ChatRole::Assistant);

        let deserialized: ChatRole = serde_json::from_str("\"System\"").unwrap();
        assert_eq!(deserialized, ChatRole::System);

        let deserialized: ChatRole = serde_json::from_str("\"Tool\"").unwrap();
        assert_eq!(deserialized, ChatRole::Tool);
    }

    #[test]
    fn test_image_mime_type() {
        assert_eq!(ImageMime::JPEG.mime_type(), "image/jpeg");
        assert_eq!(ImageMime::PNG.mime_type(), "image/png");
        assert_eq!(ImageMime::GIF.mime_type(), "image/gif");
        assert_eq!(ImageMime::WEBP.mime_type(), "image/webp");
    }

    #[test]
    fn test_message_type_default() {
        let default_type = MessageType::default();
        assert_eq!(default_type, MessageType::Text);
    }

    #[test]
    fn test_message_type_serialization() {
        let text_type = MessageType::Text;
        let serialized = serde_json::to_string(&text_type).unwrap();
        assert_eq!(serialized, "\"Text\"");

        let image_type = MessageType::Image((ImageMime::JPEG, vec![1, 2, 3]));
        let serialized = serde_json::to_string(&image_type).unwrap();
        assert!(serialized.contains("Image"));
    }

    #[test]
    fn test_reasoning_effort_display() {
        assert_eq!(ReasoningEffort::Low.to_string(), "low");
        assert_eq!(ReasoningEffort::Medium.to_string(), "medium");
        assert_eq!(ReasoningEffort::High.to_string(), "high");
    }

    #[test]
    fn test_chat_message_builder_user() {
        let message = ChatMessage::user().content("Hello, world!").build();

        assert_eq!(message.role, ChatRole::User);
        assert_eq!(message.content, "Hello, world!");
        assert_eq!(message.message_type, MessageType::Text);
    }

    #[test]
    fn test_chat_message_builder_assistant() {
        let message = ChatMessage::assistant().content("Hi there!").build();

        assert_eq!(message.role, ChatRole::Assistant);
        assert_eq!(message.content, "Hi there!");
        assert_eq!(message.message_type, MessageType::Text);
    }

    #[test]
    fn test_chat_message_builder_image() {
        let image_data = vec![1, 2, 3, 4, 5];
        let message = ChatMessage::user()
            .content("Check this image")
            .image(ImageMime::PNG, image_data.clone())
            .build();

        assert_eq!(message.role, ChatRole::User);
        assert_eq!(message.content, "Check this image");
        assert_eq!(
            message.message_type,
            MessageType::Image((ImageMime::PNG, image_data))
        );
    }

    #[test]
    fn test_chat_message_builder_pdf() {
        let pdf_data = vec![0x25, 0x50, 0x44, 0x46]; // PDF header
        let message = ChatMessage::user()
            .content("Review this PDF")
            .pdf(pdf_data.clone())
            .build();

        assert_eq!(message.role, ChatRole::User);
        assert_eq!(message.content, "Review this PDF");
        assert_eq!(message.message_type, MessageType::Pdf(pdf_data));
    }

    #[test]
    fn test_chat_message_builder_image_url() {
        let image_url = "https://example.com/image.jpg";
        let message = ChatMessage::user()
            .content("See this image")
            .image_url(image_url)
            .build();

        assert_eq!(message.role, ChatRole::User);
        assert_eq!(message.content, "See this image");
        assert_eq!(
            message.message_type,
            MessageType::ImageURL(image_url.to_string())
        );
    }

    #[test]
    fn test_chat_message_builder_tool_use() {
        let tool_calls = vec![crate::ToolCall {
            id: "call_1".to_string(),
            call_type: "function".to_string(),
            function: crate::FunctionCall {
                name: "get_weather".to_string(),
                arguments: "{\"location\": \"New York\"}".to_string(),
            },
        }];

        let message = ChatMessage::assistant()
            .content("Using weather tool")
            .tool_use(tool_calls.clone())
            .build();

        assert_eq!(message.role, ChatRole::Assistant);
        assert_eq!(message.content, "Using weather tool");
        assert_eq!(message.message_type, MessageType::ToolUse(tool_calls));
    }

    #[test]
    fn test_chat_message_builder_tool_result() {
        let tool_results = vec![crate::ToolCall {
            id: "call_1".to_string(),
            call_type: "function".to_string(),
            function: crate::FunctionCall {
                name: "get_weather".to_string(),
                arguments: "{\"temperature\": 75, \"condition\": \"sunny\"}".to_string(),
            },
        }];

        let message = ChatMessage::user()
            .content("Weather result")
            .tool_result(tool_results.clone())
            .build();

        assert_eq!(message.role, ChatRole::User);
        assert_eq!(message.content, "Weather result");
        assert_eq!(message.message_type, MessageType::ToolResult(tool_results));
    }

    #[test]
    fn test_structured_output_format_serialization() {
        let format = StructuredOutputFormat {
            name: "Person".to_string(),
            description: Some("A person object".to_string()),
            schema: Some(json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
            })),
            strict: Some(true),
        };

        let serialized = serde_json::to_string(&format).unwrap();
        let deserialized: StructuredOutputFormat = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.name, "Person");
        assert_eq!(
            deserialized.description,
            Some("A person object".to_string())
        );
        assert_eq!(deserialized.strict, Some(true));
        assert!(deserialized.schema.is_some());
    }

    #[test]
    fn test_structured_output_format_equality() {
        let format1 = StructuredOutputFormat {
            name: "Test".to_string(),
            description: None,
            schema: None,
            strict: None,
        };

        let format2 = StructuredOutputFormat {
            name: "Test".to_string(),
            description: None,
            schema: None,
            strict: None,
        };

        assert_eq!(format1, format2);
    }

    #[test]
    fn test_tool_choice_serialization() {
        // Test Auto
        let choice = ToolChoice::Auto;
        let serialized = serde_json::to_string(&choice).unwrap();
        assert_eq!(serialized, "\"auto\"");

        // Test Any
        let choice = ToolChoice::Any;
        let serialized = serde_json::to_string(&choice).unwrap();
        assert_eq!(serialized, "\"required\"");

        // Test None
        let choice = ToolChoice::None;
        let serialized = serde_json::to_string(&choice).unwrap();
        assert_eq!(serialized, "\"none\"");

        // Test specific tool
        let choice = ToolChoice::Tool("my_function".to_string());
        let serialized = serde_json::to_string(&choice).unwrap();
        // Parse back to verify structure
        let parsed: serde_json::Value = serde_json::from_str(&serialized).unwrap();
        assert_eq!(parsed["type"], "function");
        assert_eq!(parsed["function"]["name"], "my_function");
    }

    #[test]
    fn test_tool_choice_default() {
        let default_choice = ToolChoice::default();
        assert!(matches!(default_choice, ToolChoice::Auto));
    }

    #[test]
    fn test_parameter_property_serialization() {
        let property = ParameterProperty {
            property_type: "string".to_string(),
            description: "A test parameter".to_string(),
            items: None,
            enum_list: Some(vec!["option1".to_string(), "option2".to_string()]),
        };

        let serialized = serde_json::to_string(&property).unwrap();
        assert!(serialized.contains("\"type\":\"string\""));
        assert!(serialized.contains("\"description\":\"A test parameter\""));
        assert!(serialized.contains("\"enum\":[\"option1\",\"option2\"]"));
    }

    #[test]
    fn test_parameter_property_with_items() {
        let item_property = ParameterProperty {
            property_type: "string".to_string(),
            description: "Array item".to_string(),
            items: None,
            enum_list: None,
        };

        let array_property = ParameterProperty {
            property_type: "array".to_string(),
            description: "An array parameter".to_string(),
            items: Some(Box::new(item_property)),
            enum_list: None,
        };

        let serialized = serde_json::to_string(&array_property).unwrap();
        assert!(serialized.contains("\"type\":\"array\""));
        assert!(serialized.contains("\"items\""));
    }

    #[test]
    fn test_function_tool_serialization() {
        let mut properties = HashMap::new();
        properties.insert(
            "name".to_string(),
            ParameterProperty {
                property_type: "string".to_string(),
                description: "Name parameter".to_string(),
                items: None,
                enum_list: None,
            },
        );

        let schema = ParametersSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["name".to_string()],
        };

        let function = FunctionTool {
            name: "test_function".to_string(),
            description: "A test function".to_string(),
            parameters: serde_json::to_value(schema).unwrap(),
        };

        let serialized = serde_json::to_string(&function).unwrap();
        assert!(serialized.contains("\"name\":\"test_function\""));
        assert!(serialized.contains("\"description\":\"A test function\""));
        assert!(serialized.contains("\"parameters\""));
    }

    #[test]
    fn test_tool_serialization() {
        let function = FunctionTool {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            parameters: json!({"type": "object", "properties": {}}),
        };

        let tool = Tool {
            tool_type: "function".to_string(),
            function,
        };

        let serialized = serde_json::to_string(&tool).unwrap();
        assert!(serialized.contains("\"type\":\"function\""));
        assert!(serialized.contains("\"function\""));
    }

    #[test]
    fn test_chat_message_serialization() {
        let message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "Hello, world!".to_string(),
        };

        let serialized = serde_json::to_string(&message).unwrap();
        let deserialized: ChatMessage = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.role, ChatRole::User);
        assert_eq!(deserialized.message_type, MessageType::Text);
        assert_eq!(deserialized.content, "Hello, world!");
    }

    #[tokio::test]
    async fn test_chat_provider_summarize_history() {
        struct MockChatProvider;

        #[async_trait]
        impl ChatProvider for MockChatProvider {
            async fn chat_with_tools(
                &self,
                messages: &[ChatMessage],
                _tools: Option<&[Tool]>,
                _json_schema: Option<StructuredOutputFormat>,
            ) -> Result<Box<dyn ChatResponse>, LLMError> {
                // Mock implementation that returns the prompt as summary
                let prompt = messages.first().unwrap().content.clone();
                Ok(Box::new(MockChatResponse {
                    text: Some(format!("Summary: {prompt}")),
                }))
            }
        }

        struct MockChatResponse {
            text: Option<String>,
        }

        impl ChatResponse for MockChatResponse {
            fn text(&self) -> Option<String> {
                self.text.clone()
            }

            fn tool_calls(&self) -> Option<Vec<crate::ToolCall>> {
                None
            }
        }

        impl std::fmt::Debug for MockChatResponse {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "MockChatResponse")
            }
        }

        impl std::fmt::Display for MockChatResponse {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.text.as_deref().unwrap_or(""))
            }
        }

        let provider = MockChatProvider;
        let messages = vec![
            ChatMessage::user().content("First message").build(),
            ChatMessage::assistant().content("Response").build(),
        ];

        let summary = provider.summarize_history(&messages).await.unwrap();
        assert!(summary.contains("Summary:"));
        assert!(summary.contains("First message"));
        assert!(summary.contains("Response"));
    }

    #[tokio::test]
    async fn test_chat_provider_default_chat_implementation() {
        struct MockChatProvider;

        #[async_trait]
        impl ChatProvider for MockChatProvider {
            async fn chat_with_tools(
                &self,
                _messages: &[ChatMessage],
                _tools: Option<&[Tool]>,
                _json_schema: Option<StructuredOutputFormat>,
            ) -> Result<Box<dyn ChatResponse>, LLMError> {
                Ok(Box::new(MockChatResponse {
                    text: Some("Default response".to_string()),
                }))
            }
        }

        struct MockChatResponse {
            text: Option<String>,
        }

        impl ChatResponse for MockChatResponse {
            fn text(&self) -> Option<String> {
                self.text.clone()
            }

            fn tool_calls(&self) -> Option<Vec<crate::ToolCall>> {
                None
            }
        }

        impl std::fmt::Debug for MockChatResponse {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "MockChatResponse")
            }
        }

        impl std::fmt::Display for MockChatResponse {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.text.as_deref().unwrap_or(""))
            }
        }

        let provider = MockChatProvider;
        let messages = vec![ChatMessage::user().content("Test").build()];

        let response = provider.chat(&messages, None).await.unwrap();
        assert_eq!(response.text(), Some("Default response".to_string()));
    }

    #[tokio::test]
    async fn test_chat_provider_default_chat_stream_implementation() {
        struct MockChatProvider;

        #[async_trait]
        impl ChatProvider for MockChatProvider {
            async fn chat_with_tools(
                &self,
                _messages: &[ChatMessage],
                _tools: Option<&[Tool]>,
                _json_schema: Option<StructuredOutputFormat>,
            ) -> Result<Box<dyn ChatResponse>, LLMError> {
                unreachable!()
            }
        }

        let provider = MockChatProvider;
        let messages = vec![ChatMessage::user().content("Test").build()];

        let result = provider.chat_stream(&messages).await;
        assert!(result.is_err());
        if let Err(error) = result {
            assert!(error.to_string().contains("Streaming not supported"));
        }
    }

    #[tokio::test]
    async fn test_chat_provider_default_memory_contents() {
        struct MockChatProvider;

        #[async_trait]
        impl ChatProvider for MockChatProvider {
            async fn chat_with_tools(
                &self,
                _messages: &[ChatMessage],
                _tools: Option<&[Tool]>,
                _json_schema: Option<StructuredOutputFormat>,
            ) -> Result<Box<dyn ChatResponse>, LLMError> {
                unreachable!()
            }
        }

        let provider = MockChatProvider;
        let memory_contents = provider.memory_contents().await;
        assert!(memory_contents.is_none());
    }
}
