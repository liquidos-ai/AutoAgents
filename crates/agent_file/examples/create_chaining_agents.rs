//! An example of creating .af files for the chaining agents.

use agent_file::{
    parser::to_string,
    schema::*,
};
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- Define Agent 1 ---
    let agent_1 = AgentFile {
        agent_type: "agent_1".to_string(),
        name: "agent_1".to_string(),
        description: Some("A math geek and expert in linear algebra.".to_string()),
        system: "You are a math geek and expert in linear algebra".to_string(),
        tools: vec![],
        tags: vec![
            Tag { tag: "math".to_string() },
            Tag { tag: "linear-algebra".to_string() },
        ],
        // --- Boilerplate fields ---
        core_memory: vec![],
        created_at: "2025-07-24T16:13:00".to_string(),
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
            temperature: 0.0,
            max_tokens: 1024,
            enable_reasoner: false,
            max_reasoning_tokens: 2048,
        },
        message_buffer_autoclear: true,
        in_context_message_indices: vec![],
        messages: vec![],
        metadata: None,
        multi_agent_group: None,
        tool_exec_environment_variables: vec![],
        tool_rules: vec![],
        updated_at: "2025-07-24T16:13:00".to_string(),
        version: "0.1.0".to_string(),
    };

    // --- Define Agent 2 ---
    let agent_2 = AgentFile {
        agent_type: "agent_2".to_string(),
        name: "agent_2".to_string(),
        description: Some("A math professor who reviews content for correctness.".to_string()),
        system: "You are a math professor in linear algebera, Your goal is to review the given content if correct".to_string(),
        tools: vec![],
        tags: vec![
            Tag { tag: "math".to_string() },
            Tag { tag: "review".to_string() },
        ],
        // --- Boilerplate fields (can be shared or customized) ---
        core_memory: vec![],
        created_at: "2025-07-24T16:13:00".to_string(),
        embedding_config: agent_1.embedding_config.clone(), // Reuse config
        llm_config: agent_1.llm_config.clone(), // Reuse config
        message_buffer_autoclear: true,
        in_context_message_indices: vec![],
        messages: vec![],
        metadata: None,
        multi_agent_group: None,
        tool_exec_environment_variables: vec![],
        tool_rules: vec![],
        updated_at: "2025-07-24T16:13:00".to_string(),
        version: "0.1.0".to_string(),
    };

    // --- Serialize and save Agent 1 ---
    let json_string_1 = to_string(&agent_1)?;
    let output_path_1 = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples").join("agent_1.af");
    fs::write(&output_path_1, json_string_1)?;
    println!("Successfully created agent file at: {:?}", output_path_1);

    // --- Serialize and save Agent 2 ---
    let json_string_2 = to_string(&agent_2)?;
    let output_path_2 = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples").join("agent_2.af");
    fs::write(&output_path_2, json_string_2)?;
    println!("Successfully created agent file at: {:?}", output_path_2);

    Ok(())
}
