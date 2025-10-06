use thiserror::Error;

#[derive(Error, Debug)]
pub enum WorkflowError {
    #[error("YAML parsing error: {0}")]
    YamlError(#[from] serde_yaml::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Agent error: {0}")]
    AgentError(#[from] autoagents::core::error::Error),

    #[error("Runnable agent error: {0}")]
    RunnableAgentError(String),

    #[error("Runtime error: {0}")]
    RuntimeError(String),

    #[error("LLM error: {0}")]
    LLMError(#[from] autoagents::llm::error::LLMError),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Execution error: {0}")]
    ExecutionError(String),

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Unsupported workflow kind: {0}")]
    UnsupportedWorkflowKind(String),

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Invalid model configuration: {0}")]
    InvalidModelConfig(String),
}

impl From<autoagents::core::agent::error::RunnableAgentError> for WorkflowError {
    fn from(err: autoagents::core::agent::error::RunnableAgentError) -> Self {
        WorkflowError::RunnableAgentError(err.to_string())
    }
}

impl From<autoagents::core::runtime::RuntimeError> for WorkflowError {
    fn from(err: autoagents::core::runtime::RuntimeError) -> Self {
        WorkflowError::RuntimeError(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, WorkflowError>;
