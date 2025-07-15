use serde::{de::DeserializeOwned, Serialize};

/// Trait for agent output types that can generate structured output schemas
pub trait AgentOutputT: Serialize + DeserializeOwned + Send + Sync {
    /// Get the JSON schema string for this output type
    fn output_schema() -> &'static str;

    /// Get the structured output format as a JSON value
    fn structured_output_format() -> serde_json::Value;
}

/// Implementation of AgentOutputT for String type (when no output is specified)
impl AgentOutputT for String {
    fn output_schema() -> &'static str {
        "{}"
    }

    fn structured_output_format() -> serde_json::Value {
        serde_json::Value::Null
    }
}
