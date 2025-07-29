//! Defines the Rust data structures that correspond to the .af (Agent File) format schema.
//! These structs are designed for deserialization from JSON using the `serde` framework.
//! 

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// The main struct representing a deserialized .af file.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct AgentFile {
    /// Type of the agent (e.g., "memgpt_agent")
    pub agent_type: String,
    
    /// Core memory blocks (persona, human, etc.)
    pub core_memory: Vec<CoreMemoryBlock>,
    
    /// When the agent was created (ISO 8601 format)
    pub created_at: String,
    
    /// Optional description of the agent
    pub description: Option<String>,
    
    /// Embedding model configuration
    pub embedding_config: EmbeddingConfig,
    
    /// LLM configuration
    pub llm_config: LlmConfig,
    
    /// Whether to clear message buffer automatically
    pub message_buffer_autoclear: bool,
    
    /// Indices of messages to keep in context
    pub in_context_message_indices: Vec<i32>,
    
    /// Conversation history
    pub messages: Vec<Message>,
    
    /// Metadata (must be an object, not null)
    #[serde(rename = "metadata_")]
    pub metadata: Option<HashMap<String, Value>>,
    
    /// Multi-agent group configuration (if any)
    pub multi_agent_group: Option<Value>,
    
    /// Name of the agent
    pub name: String,
    
    /// System prompt
    pub system: String,
    
    /// Tags for categorization
    pub tags: Vec<Tag>,
    
    /// Environment variables for tools
    pub tool_exec_environment_variables: Vec<ToolEnvVar>,
    
    /// Rules for tool execution
    pub tool_rules: Vec<ToolRule>,
    
    /// Available tools
    pub tools: Vec<Tool>,
    
    /// When the agent was last updated (ISO 8601 format)
    pub updated_at: String,
    
    /// Version of the agent file format
    pub version: String,
}

/// Represents a core memory block
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct CoreMemoryBlock {
    /// When the memory block was created (ISO 8601 format)
    pub created_at: String,
    
    /// Description of the memory block
    pub description: Option<String>,
    
    /// Whether this is a template
    pub is_template: bool,
    
    /// Label for the memory block (e.g., "persona", "human")
    pub label: String,
    
    /// Character limit for this memory block
    pub limit: i32,
    
    /// Metadata associated with the memory block
    #[serde(rename = "metadata_")]
    pub metadata: Option<HashMap<String, Value>>,
    
    /// Template name (if this is a template)
    pub template_name: Option<String>,
    
    /// When the memory block was last updated (ISO 8601 format)
    pub updated_at: String,
    
    /// Content of the memory block
    pub value: String,
}

/// Represents embedding model configuration
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct EmbeddingConfig {
    /// Azure deployment name (if using Azure)
    pub azure_deployment: Option<String>,
    
    /// Azure endpoint URL (if using Azure)
    pub azure_endpoint: Option<String>,
    
    /// Azure API version (if using Azure)
    pub azure_version: Option<String>,
    
    /// Size of embedding chunks
    pub embedding_chunk_size: i32,
    
    /// Dimension of embeddings
    pub embedding_dim: i32,
    
    /// Endpoint URL for embedding service
    pub embedding_endpoint: String,
    
    /// Type of embedding endpoint (e.g., "openai")
    pub embedding_endpoint_type: String,
    
    /// Model name for embeddings
    pub embedding_model: String,
    
    /// Handle for the embedding model
    pub handle: String,
}

/// Represents LLM configuration
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct LlmConfig {
    /// Context window size
    pub context_window: i32,
    
    /// Whether to enable reasoning
    pub enable_reasoner: bool,
    
    /// Handle for the LLM model
    pub handle: String,
    
    /// Maximum reasoning tokens
    pub max_reasoning_tokens: i32,
    
    /// Maximum tokens for responses
    pub max_tokens: i32,
    
    /// Model name
    pub model: String,
    
    /// Endpoint URL for the model
    pub model_endpoint: String,
    
    /// Type of model endpoint (e.g., "openai", "anthropic")
    pub model_endpoint_type: String,
    
    /// Model wrapper (if any)
    pub model_wrapper: Option<String>,
    
    /// Whether to put inner thoughts in kwargs
    pub put_inner_thoughts_in_kwargs: bool,
    
    /// Temperature for generation
    pub temperature: f64,
}

/// Represents a message in the conversation
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct Message {
    /// When the message was created (ISO 8601 format)
    pub created_at: String,
    
    /// Optional group ID for message grouping
    pub group_id: Option<String>,
    
    /// Model used to generate this message (if applicable)
    pub model: Option<String>,
    
    /// Name of the message sender (if any)
    pub name: Option<String>,
    
    /// Role of the message sender (e.g., "user", "assistant", "system")
    pub role: String,
    
    /// Content of the message
    pub content: Vec<MessageContent>,
    
    /// ID of the tool call this message is associated with (if any)
    pub tool_call_id: Option<String>,
    
    /// List of tool calls in this message
    pub tool_calls: Vec<Value>,
    
    /// List of tool call returns in this message
    pub tool_returns: Vec<Value>,
    
    /// When the message was last updated (ISO 8601 format)
    pub updated_at: String,
}

/// Represents the content of a message
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum MessageContent {
    /// Text content
    Text { #[serde(rename = "type")] content_type: String, text: String },
    
    /// Structured content
    Structured(Value),
}

/// Represents a tag for categorizing agents
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Tag {
    /// The tag value
    pub tag: String,
}

/// Represents an environment variable for tool execution
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct ToolEnvVar {
    /// When the environment variable was created (ISO 8601 format)
    pub created_at: String,
    
    /// Description of the environment variable
    pub description: Option<String>,
    
    /// Name of the environment variable
    pub key: String,
    
    /// When the environment variable was last updated (ISO 8601 format)
    pub updated_at: String,
    
    /// Value of the environment variable
    pub value: String,
}

/// Represents a rule for tool execution
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum ToolRule {
    /// Base tool rule with just a tool name
    Base {
        tool_name: String,
        #[serde(rename = "type")]
        rule_type: String,
    },
    
    /// Rule with child rules
    Child {
        tool_name: String,
        #[serde(rename = "type")]
        rule_type: String,
        children: Vec<String>,
    },
    
    /// Rule with a maximum count per step
    MaxCountPerStep {
        tool_name: String,
        #[serde(rename = "type")]
        rule_type: String,
        max_count_limit: i32,
    },
    
    /// Conditional rule
    Conditional {
        tool_name: String,
        #[serde(rename = "type")]
        rule_type: String,
        default_child: Option<String>,
        child_output_mapping: HashMap<String, String>,
        require_output_mapping: bool,
    },
}

/// Represents a tool that the agent can use
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct Tool {
    /// JSON schema for the tool's arguments
    pub args_json_schema: Option<Value>,
    
    /// When the tool was created (ISO 8601 format)
    pub created_at: String,
    
    /// Description of what the tool does
    pub description: String,
    
    /// JSON schema for the tool
    pub json_schema: ToolJsonSchema,
    
    /// Name of the tool
    pub name: String,
    
    /// Maximum number of characters to return from the tool
    pub return_char_limit: i32,
    
    /// Source code of the tool (if any)
    pub source_code: Option<String>,
    
    /// Type of the tool source (e.g., "python", "letta_core")
    pub source_type: String,
    
    /// Tags for the tool
    pub tags: Vec<String>,
    
    /// Type of the tool (e.g., "custom", "letta_core")
    pub tool_type: String,
    
    /// When the tool was last updated (ISO 8601 format)
    pub updated_at: String,
    
    /// Metadata associated with the tool
    #[serde(rename = "metadata_")]
    pub metadata: Option<HashMap<String, Value>>,
}

/// JSON schema for a tool
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct ToolJsonSchema {
    /// Name of the tool
    pub name: String,
    
    /// Description of what the tool does
    pub description: String,
    
    /// Parameters schema for the tool
    pub parameters: Parameters,
    
    /// Type of the schema (usually "object")
    #[serde(rename = "type")]
    pub schema_type: Option<String>,
    
    /// List of required parameters
    pub required: Option<Vec<String>>,
}

/// Parameters for a tool
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct Parameters {
    /// Type of the parameters (usually "object")
    #[serde(rename = "type")]
    pub param_type: Option<String>,
    
    /// Properties of the parameters
    pub properties: HashMap<String, ParameterProperties>,
    
    /// List of required parameters
    pub required: Vec<String>,
}

/// Properties of a parameter
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct ParameterProperties {
    /// Type of the parameter
    #[serde(rename = "type")]
    pub param_type: String,
    
    /// Description of the parameter
    pub description: Option<String>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            azure_deployment: None,
            azure_endpoint: None,
            azure_version: None,
            embedding_chunk_size: 300,
            embedding_dim: 1536,
            embedding_endpoint: "https://api.openai.com/v1".to_string(),
            embedding_endpoint_type: "openai".to_string(),
            embedding_model: "text-embedding-ada-002".to_string(),
            handle: "openai/text-embedding-ada-002".to_string(),
        }
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            context_window: 32000,
            enable_reasoner: false,
            handle: "openai/gpt-4o-mini".to_string(),
            max_reasoning_tokens: 0,
            max_tokens: 4096,
            model: "gpt-4o-mini".to_string(),
            model_endpoint: "https://api.openai.com/v1".to_string(),
            model_endpoint_type: "openai".to_string(),
            model_wrapper: None,
            put_inner_thoughts_in_kwargs: true,
            temperature: 0.7,
        }
    }
} 