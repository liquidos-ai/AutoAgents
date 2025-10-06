use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum WorkflowKind {
    Direct,
    Sequential,
    Parallel,
    Routing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowConfig {
    pub kind: WorkflowKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default)]
    pub stream: bool,
    pub workflow: WorkflowSpec,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<EnvironmentConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime: Option<RuntimeConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowSpec {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent: Option<AgentConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agents: Option<Vec<AgentConfig>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub router: Option<AgentConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub handlers: Option<Vec<HandlerConfig>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<OutputConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum ExecutorKind {
    Basic,
    #[serde(rename = "ReAct")]
    ReAct,
}

impl Default for ExecutorKind {
    fn default() -> Self {
        ExecutorKind::ReAct
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub name: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory: Option<MemoryConfig>,
    #[serde(default)]
    pub executor: ExecutorKind,
    pub model: ModelConfig,
    #[serde(default)]
    pub tools: Vec<ToolConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<OutputConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub kind: String, // "llm" for now
    pub backend: BackendConfig,
    pub provider: String, // "OpenAI", "Anthropic", "Ollama", etc.
    pub model_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<ModelParameters>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    pub kind: String, // "Cloud" or "Local"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolConfig {
    #[serde(rename = "type")]
    pub tool_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputType {
    Text,
    Json,
    Structured,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    #[serde(rename = "type")]
    pub output_type: OutputType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub kind: String, // "sliding_window", "buffer", etc.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<MemoryParameters>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_slide: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window_size: Option<usize>,
}

impl MemoryConfig {
    pub fn get_window_size(&self) -> usize {
        if let Some(params) = &self.parameters {
            // Support both n_slide and window_size, prefer n_slide
            params.n_slide.or(params.window_size).unwrap_or(10)
        } else {
            10
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandlerConfig {
    pub condition: String,
    pub agent: AgentConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub working_directory: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_seconds: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    #[serde(rename = "type")]
    pub runtime_type: String, // "single_threaded", "multi_threaded"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_concurrent: Option<usize>,
}
