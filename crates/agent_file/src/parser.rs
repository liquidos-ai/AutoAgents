//! Contains the primary parsing logic for deserializing .af files.

use crate::error::AgentFileError;
use crate::schema::AgentFile;

/// Parses a string containing .af JSON data into an `AgentFile` struct.
///
/// # Arguments
///
/// * `s` - A string slice that holds the content of the .af file.
///
/// # Returns
///
/// A `Result` which is `Ok` containing the deserialized `AgentFile` on success,
/// or an `Err` with an `AgentFileError` on failure.
pub fn from_str(s: &str) -> Result<AgentFile, AgentFileError> {
    let agent_file: AgentFile = serde_json::from_str(s)?;
    Ok(agent_file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_memgpt_agent() {
        let agent_json = r#"
        {
    "agent_type": "memgpt_agent",
    "core_memory": [
        {
            "created_at": "2025-04-01T03:47:27",
            "description": null,
            "is_template": false,
            "label": "persona",
            "limit": 5000,
            "metadata_": {},
            "template_name": null,
            "updated_at": "2025-04-01T03:47:27",
            "value": "The following is a starter persona..."
        },
        {
            "created_at": "2025-04-01T03:47:27",
            "description": null,
            "is_template": false,
            "label": "human",
            "limit": 5000,
            "metadata_": {},
            "template_name": null,
            "updated_at": "2025-04-01T03:47:27",
            "value": "This is what I know so far about the user..."
        }
    ],
    "created_at": "2025-04-01T03:47:27",
    "description": "A simple MemGPT agent from the original project release",
    "embedding_config": {
        "embedding_endpoint_type": "openai",
        "embedding_endpoint": "https://api.openai.com/v1",
        "embedding_model": "text-embedding-ada-002",
        "embedding_dim": 1536,
        "embedding_chunk_size": 300,
        "handle": "openai/text-embedding-ada-002",
        "azure_endpoint": null,
        "azure_version": null,
        "azure_deployment": null
    },
    "llm_config": {
        "model": "gpt-4-0613",
        "model_endpoint_type": "openai",
        "model_endpoint": "https://api.openai.com/v1",
        "model_wrapper": null,
        "context_window": 8192,
        "put_inner_thoughts_in_kwargs": true,
        "handle": "openai/gpt-4-june",
        "temperature": 0.7,
        "max_tokens": 4096,
        "enable_reasoner": false,
        "max_reasoning_tokens": 0
    },
    "message_buffer_autoclear": false,
    "in_context_message_indices": [0, 1, 2, 3],
    "messages": [],
    "metadata_": null,
    "multi_agent_group": null,
    "name": "memgpt_agent",
    "system": "You are Letta...",
    "tags": [],
    "tool_exec_environment_variables": [],
    "tool_rules": [
        {"tool_name": "conversation_search", "type": "continue_loop"},
        {"tool_name": "archival_memory_search", "type": "continue_loop"}
    ],
    "tools": [],
    "updated_at": "2025-04-01T03:47:27.514261",
    "version": "0.6.47"
}
        "#;

        let result = from_str(agent_json);
        assert!(result.is_ok());
        let agent_file = result.unwrap();
        assert_eq!(agent_file.name, "memgpt_agent");
        assert_eq!(agent_file.llm_config.model, "gpt-4-0613");
        assert_eq!(agent_file.core_memory.len(), 2);
    }
}


