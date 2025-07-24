//! Defines the Rust data structures that correspond to the .af (Agent File) format schema.
//! These structs are designed for deserialization from JSON using the `serde` framework.

use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;

/// The main struct representing a deserialized .af file.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AgentFile {
    pub agent_type: String,
    pub core_memory: Vec<CoreMemoryBlock>,
    pub created_at: String,
    pub description: Option<String>,
    pub embedding_config: EmbeddingConfig,
    pub llm_config: LlmConfig,
    pub message_buffer_autoclear: bool,
    pub in_context_message_indices: Vec<i64>,
    pub messages: Vec<Message>,
    #[serde(rename = "metadata_")]
    pub metadata: Option<HashMap<String, Value>>,
    pub multi_agent_group: Option<Value>,
    pub name: String,
    pub system: String,
    pub tags: Vec<Tag>,
    pub tool_exec_environment_variables: Vec<ToolEnvVar>,
    pub tool_rules: Vec<ToolRule>,
    pub tools: Vec<Tool>,
    pub updated_at: String,
    pub version: String,
}

/// Represents a block of core memory for the agent, such as persona or user info.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CoreMemoryBlock {
    pub created_at: String,
    pub description: Option<String>,
    pub is_template: bool,
    pub label: String,
    pub limit: i64,
    #[serde(rename = "metadata_")]
    pub metadata: Option<HashMap<String, Value>>,
    pub template_name: Option<String>,
    pub updated_at: String,
    pub value: String,
}

/// Represents a single message in the agent's conversation history.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Message {
    pub created_at: String,
    pub group_id: Option<String>,
    pub model: Option<String>,
    pub name: Option<String>,
    pub role: String,
    pub content: Vec<MessageContent>,
    pub tool_call_id: Option<String>,
    pub tool_calls: Vec<Value>,
    pub tool_returns: Vec<Value>,
    pub updated_at: String,
}

/// Represents the content of a message, which can be text or other types.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContent {
    Text { text: String },
    // Other content types can be added here as they are identified.
}

/// Represents a tag associated with the agent.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Tag {
    pub tag: String,
}

/// Represents an environment variable for tool execution.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ToolEnvVar {
    pub created_at: String,
    pub description: Option<String>,
    pub key: String,
    pub updated_at: String,
    pub value: String,
}

/// Represents a rule governing tool execution behavior.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolRule {
    ContinueLoop,
    ExitLoop,
    // The Pydantic schema defines more complex rules which can be added here.
    #[serde(rename_all = "snake_case")]
    BaseToolRule {
        tool_name: String,
    },
    #[serde(rename_all = "snake_case")]
    ChildToolRule {
        tool_name: String,
        children: Vec<String>,
    },
    #[serde(rename_all = "snake_case")]
    #[serde(rename = "max_count_per_step")]
    MaxCountPerStepToolRule {
        tool_name: String,
        max_count_limit: i64,
    },
    #[serde(rename_all = "snake_case")]
    #[serde(rename = "conditional")]
    ConditionalToolRule {
        tool_name: String,
        default_child: Option<String>,
        child_output_mapping: HashMap<String, String>,
        require_output_mapping: bool,
    },
    #[serde(rename_all = "snake_case")]
    ConstrainChildTools {
        tool_name: String,
        children: Vec<String>,
    },
}

/// Represents a tool that the agent can use.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Tool {
    pub args_json_schema: Option<Value>,
    pub created_at: String,
    pub description: String,
    pub json_schema: ToolJsonSchema,
    pub name: String,
    pub return_char_limit: i64,
    pub source_code: Option<String>,
    pub source_type: String,
    pub tags: Vec<String>,
    pub tool_type: String,
    pub updated_at: String,
    #[serde(rename = "metadata_")]
    pub metadata: Option<HashMap<String, Value>>,
}

/// The JSON schema definition for a tool's parameters.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ToolJsonSchema {
    pub name: String,
    pub description: String,
    pub parameters: Parameters,
    #[serde(rename = "type")]
    pub schema_type: Option<String>,
    pub required: Option<Vec<String>>,
}

/// The parameters of a tool's JSON schema.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Parameters {
    #[serde(rename = "type")]
    pub schema_type: Option<String>,
    pub properties: HashMap<String, ParameterProperties>,
    #[serde(default)]
    pub required: Vec<String>,
}

/// The properties of a single parameter in a tool's schema.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ParameterProperties {
    #[serde(rename = "type")]
    pub property_type: String,
    pub description: Option<String>,
}

/// Configuration for the embedding model.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct EmbeddingConfig {
    pub embedding_endpoint_type: String,
    pub embedding_endpoint: String,
    pub embedding_model: String,
    pub embedding_dim: i64,
    pub embedding_chunk_size: i64,
    pub handle: String,
    pub azure_endpoint: Option<String>,
    pub azure_version: Option<String>,
    pub azure_deployment: Option<String>,
}

/// Configuration for the language model.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct LlmConfig {
    pub model: String,
    pub model_endpoint_type: String,
    pub model_endpoint: String,
    pub model_wrapper: Option<String>,
    pub context_window: i64,
    pub put_inner_thoughts_in_kwargs: bool,
    pub handle: String,
    pub temperature: f64,
    pub max_tokens: i64,
    pub enable_reasoner: bool,
    pub max_reasoning_tokens: i64,
}
