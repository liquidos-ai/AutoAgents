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
    pub in_context_message_indices: Vec<usize>,
    
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

/// Represents a block of core memory for the agent, such as persona or user info.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct CoreMemoryBlock {
    /// When the memory block was created (ISO 8601 format)
    pub created_at: String,
    
    /// Optional description of the memory block
    pub description: Option<String>,
    
    /// Whether this is a template memory block
    pub is_template: bool,
    
    /// Label identifying the type of memory (e.g., "persona", "human")
    pub label: String,
    
    /// Maximum size limit for this memory block
    pub limit: i64,
    
    /// Metadata associated with this memory block
    #[serde(rename = "metadata_")]
    pub metadata: Option<HashMap<String, Value>>,
    
    /// Name of the template this memory block is based on (if any)
    pub template_name: Option<String>,
    
    /// When the memory block was last updated (ISO 8601 format)
    pub updated_at: String,
    
    /// The actual content of the memory block
    pub value: String,
}

/// Represents a message in the agent's conversation history
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
    Text(String),
    
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
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolRule {
    /// Base tool rule with just a tool name
    Base {
        /// Name of the tool this rule applies to
        tool_name: String,
    },
    
    /// Rule with child rules
    Child {
        /// Name of the tool this rule applies to
        tool_name: String,
        
        /// Child rule names
        children: Vec<String>,
    },
    
    /// Rule with a maximum count per step
    MaxCountPerStep {
        /// Name of the tool this rule applies to
        tool_name: String,
        
        /// Maximum number of times the tool can be called per step
        max_count_limit: i32,
    },
    
    /// Conditional rule
    Conditional {
        /// Name of the tool this rule applies to
        tool_name: String,
        
        /// Default child rule to use
        default_child: Option<String>,
        
        /// Mapping of output values to child rules
        child_output_mapping: HashMap<String, String>,
        
        /// Whether output mapping is required
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
    
    /// Type of the tool source (e.g., "code")
    pub source_type: String,
    
    /// Tags for the tool
    pub tags: Vec<String>,
    
    /// Type of the tool (e.g., "function")
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
    /// Type of the parameter (e.g., "string", "integer")
    #[serde(rename = "type")]
    pub param_type: String,
    
    /// Description of the parameter
    pub description: Option<String>,
}

/// Configuration for the embedding model
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "snake_case")]
pub struct EmbeddingConfig {
    /// Azure deployment name (if using Azure)
    pub azure_deployment: Option<String>,
    
    /// Azure endpoint (if using Azure)
    pub azure_endpoint: Option<String>,
    
    /// Azure API version (if using Azure)
    pub azure_version: Option<String>,
    
    /// Chunk size for embeddings
    pub embedding_chunk_size: i32,
    
    /// Dimensionality of the embeddings
    pub embedding_dim: i32,
    
    /// Endpoint for the embedding service
    pub embedding_endpoint: String,
    
    /// Type of the embedding endpoint (e.g., "openai", "azure")
    pub embedding_endpoint_type: String,
    
    /// Name of the embedding model
    pub embedding_model: String,
    
    /// Handle for the embedding configuration
    pub handle: String,
}

/// Configuration for the language model
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "snake_case")]
pub struct LlmConfig {
    /// Context window size in tokens
    pub context_window: i32,
    
    /// Whether to enable the reasoner
    pub enable_reasoner: bool,
    
    /// Handle for the LLM configuration
    pub handle: String,
    
    /// Maximum number of tokens for reasoning
    pub max_reasoning_tokens: i32,
    
    /// Maximum number of tokens to generate
    pub max_tokens: i32,
    
    /// Name of the model
    pub model: String,
    
    /// Endpoint for the model
    pub model_endpoint: String,
    
    /// Type of the model endpoint (e.g., "openai", "azure")
    pub model_endpoint_type: String,
    
    /// Model wrapper (if any)
    pub model_wrapper: Option<String>,
    
    /// Whether to put inner thoughts in kwargs
    pub put_inner_thoughts_in_kwargs: bool,
    
    /// Sampling temperature
    pub temperature: f64,
}


