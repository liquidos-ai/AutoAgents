//! An example of creating an .af file programmatically.

use agent_file::{
    parser::to_string,
    schema::*,
};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create the agent's data structures in Rust.
    let agent = AgentFile {
        agent_type: "simple_agent".to_string(),
        core_memory: vec![
            CoreMemoryBlock {
                created_at: "2025-07-24T15:30:00".to_string(),
                description: Some("The agent's persona".to_string()),
                is_template: false,
                label: "persona".to_string(),
                limit: 1000,
                metadata: Some(HashMap::new()),
                template_name: None,
                updated_at: "2025-07-24T15:30:00".to_string(),
                value: "You are a friendly assistant who greets users.".to_string(),
            }
        ],
        created_at: "2025-07-24T15:30:00".to_string(),
        description: Some("A simple agent that greets the user".to_string()),
        embedding_config: EmbeddingConfig {
            embedding_endpoint_type: "none".to_string(),
            embedding_endpoint: "".to_string(),
            embedding_model: "".to_string(),
            embedding_dim: 0,
            embedding_chunk_size: 0,
            handle: "".to_string(),
            azure_endpoint: None,
            azure_version: None,
            azure_deployment: None,
        },
        llm_config: LlmConfig {
            model: "gpt-3.5-turbo".to_string(),
            model_endpoint_type: "openai".to_string(),
            model_endpoint: "https://api.openai.com/v1".to_string(),
            model_wrapper: None,
            context_window: 4096,
            put_inner_thoughts_in_kwargs: true,
            handle: "openai/gpt-3.5-turbo".to_string(),
            temperature: 0.7,
            max_tokens: 100,
            enable_reasoner: false,
            max_reasoning_tokens: 0,
        },
        message_buffer_autoclear: true,
        in_context_message_indices: vec![],
        messages: vec![],
        metadata: None,
        multi_agent_group: None,
        name: "greeter_agent".to_string(),
        system: "You are a helpful AI assistant.".to_string(),
        tags: vec![],
        tool_exec_environment_variables: vec![],
        tool_rules: vec![],
        tools: vec![],
        updated_at: "2025-07-24T15:30:00".to_string(),
        version: "0.1.0".to_string(),
    };

    // 2. Serialize the struct to a JSON string.
    let json_string = to_string(&agent)?;

    // 3. Save the JSON string to a file.
    let output_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples").join("greeter_agent.af");
    fs::write(&output_path, json_string)?;

    println!("Successfully created agent file at: {:?}", output_path);

    Ok(())
}
