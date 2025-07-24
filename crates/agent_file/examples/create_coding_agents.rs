//! An example of creating .af files for the coding agents.

use agent_file::{parser::to_string, schema::*};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- Define all 7 tools first ---
    let tools = create_all_tools();

    // --- Define CodingAgent ---
    let coding_agent = AgentFile {
        agent_type: "coding_agent".to_string(),
        name: "coding_agent".to_string(),
        description: Some("A verbose coding agent that follows the ReAct pattern.".to_string()),
        system: "You are a coding agent operating within the AutoAgents framework... [rest of long prompt]".to_string(),
        tools: tools.clone(),
        tags: vec![Tag { tag: "coding".to_string() }, Tag { tag: "react".to_string() }],
        core_memory: vec![],
        created_at: "2025-07-24T16:20:00".to_string(),
        embedding_config: default_embedding_config(),
        llm_config: default_llm_config(),
        message_buffer_autoclear: true,
        in_context_message_indices: vec![],
        messages: vec![],
        metadata: None,
        multi_agent_group: None,
        tool_exec_environment_variables: vec![],
        tool_rules: vec![],
        updated_at: "2025-07-24T16:20:00".to_string(),
        version: "0.1.0".to_string(),
    };

    // --- Define ConciseCodingAgent ---
    let concise_agent = AgentFile {
        agent_type: "concise_coding_agent".to_string(),
        name: "concise_coding_agent".to_string(),
        description: Some("A concise coding assistant for efficient task completion.".to_string()),
        system: "You are a concise coding assistant. Use tools to complete tasks efficiently...".to_string(),
        tools: tools.clone(),
        tags: vec![Tag { tag: "coding".to_string() }, Tag { tag: "concise".to_string() }],
        core_memory: vec![],
        created_at: "2025-07-24T16:20:00".to_string(),
        embedding_config: default_embedding_config(),
        llm_config: default_llm_config(),
        message_buffer_autoclear: true,
        in_context_message_indices: vec![],
        messages: vec![],
        metadata: None,
        multi_agent_group: None,
        tool_exec_environment_variables: vec![],
        tool_rules: vec![],
        updated_at: "2025-07-24T16:20:00".to_string(),
        version: "0.1.0".to_string(),
    };

    let json_string_1 = to_string(&coding_agent)?;
    let output_path_1 = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples").join("coding_agent.af");
    fs::write(&output_path_1, json_string_1)?;
    println!("Successfully created agent file at: {:?}", output_path_1);

    let json_string_2 = to_string(&concise_agent)?;
    let output_path_2 = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples").join("concise_coding_agent.af");
    fs::write(&output_path_2, json_string_2)?;
    println!("Successfully created agent file at: {:?}", output_path_2);

    Ok(())
}

fn create_all_tools() -> Vec<Tool> {
    vec![
        create_tool("FileSearchTool", "Search for files matching a glob pattern", vec![("pattern", "string", "Glob pattern"), ("base_dir", "string", "Base directory")]),
        create_tool("GrepTool", "Search for content in files using regex", vec![("pattern", "string", "Regex pattern"), ("file_pattern", "string", "File glob pattern"), ("base_dir", "string", "Base directory")]),
        create_tool("ReadFileTool", "Read a file's content", vec![("file_path", "string", "Path to the file to read")]),
        create_tool("WriteFileTool", "Write content to a file", vec![("file_path", "string", "Path to write to"), ("content", "string", "Content to write")]),
        create_tool("DeleteFileTool", "Delete a file", vec![("file_path", "string", "Path to the file to delete")]),
        create_tool("ListDirectoryTool", "List directory contents", vec![("dir_path", "string", "Path to the directory")]),
        create_tool("AnalyzeCodeTool", "Analyze code", vec![("path", "string", "Path to analyze"), ("analysis_type", "string", "Type of analysis")]),
    ]
}

fn create_tool(name: &str, description: &str, params: Vec<(&str, &str, &str)>) -> Tool {
    let now = "2025-07-24T16:20:00".to_string();
    let properties: HashMap<String, ParameterProperties> = params.iter().map(|(key, p_type, desc)| {
        (key.to_string(), ParameterProperties { property_type: p_type.to_string(), description: Some(desc.to_string()) })
    }).collect();
    let required: Vec<String> = params.iter().map(|(key, _, _)| key.to_string()).collect();

    Tool {
        name: name.to_string(),
        description: description.to_string(),
        json_schema: ToolJsonSchema {
            name: name.to_string(),
            description: description.to_string(),
            parameters: Parameters {
                schema_type: Some("object".to_string()),
                properties,
                required: required.clone(),
            },
        },
        args_json_schema: None,
        created_at: now.clone(),
        return_char_limit: 0,
        source_code: None,
        source_type: "code".to_string(),
        tags: vec![],
        tool_type: "function".to_string(),
        updated_at: now.clone(),
        metadata: None,
    }
}

fn default_embedding_config() -> EmbeddingConfig {
    EmbeddingConfig {
        embedding_endpoint_type: "none".to_string(),
        embedding_endpoint: "".to_string(),
        embedding_model: "".to_string(),
        embedding_dim: 0,
        embedding_chunk_size: 0,
        handle: "".to_string(),
        azure_endpoint: None,
        azure_version: None,
        azure_deployment: None,
    }
}

fn default_llm_config() -> LlmConfig {
    LlmConfig {
        model: "gpt-4o".to_string(),
        model_endpoint_type: "openai".to_string(),
        model_endpoint: "https://api.openai.com/v1".to_string(),
        model_wrapper: None,
        context_window: 128000,
        put_inner_thoughts_in_kwargs: true,
        handle: "openai/gpt-4o".to_string(),
        temperature: 0.0,
        max_tokens: 4096,
        enable_reasoner: false,
        max_reasoning_tokens: 4096,
    }
}
