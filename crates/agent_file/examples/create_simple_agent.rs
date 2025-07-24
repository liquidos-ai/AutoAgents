//! An example of creating an .af file for the SimpleAgent.

use agent_file::{
    parser::to_string,
    schema::*,
};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define the schema for the tool's parameters.
    let properties = HashMap::from([
        (
            "left".to_string(),
            ParameterProperties {
                property_type: "integer".to_string(),
                description: Some("Left Operand for addition".to_string()),
            },
        ),
        (
            "right".to_string(),
            ParameterProperties {
                property_type: "integer".to_string(),
                description: Some("Right Operand for addition".to_string()),
            },
        ),
    ]);

    let parameters = Parameters {
        schema_type: Some("object".to_string()),
        properties,
        required: vec!["left".to_string(), "right".to_string()],
    };

    let tool_json_schema = ToolJsonSchema {
        name: "Addition".to_string(),
        description: "Calculate the sum of two integers.".to_string(),
        parameters,
        schema_type: Some("object".to_string()),
        required: Some(vec!["left".to_string(), "right".to_string()]),
    };

    // 2. Define the tool itself.
    let addition_tool = Tool {
        name: "Addition".to_string(),
        description: "Calculate the sum of two integers.".to_string(),
        json_schema: tool_json_schema,
        created_at: "2025-07-24T16:05:00".to_string(),
        updated_at: "2025-07-24T16:05:00".to_string(),
        source_type: "code".to_string(),
        source_code: None,
        tool_type: "function".to_string(),
        return_char_limit: 0,
        tags: vec![],
        args_json_schema: None,
        metadata: None,
    };

    // 3. Create the agent's data structures in Rust.
    let agent = AgentFile {
        agent_type: "simple_agent".to_string(),
        core_memory: vec![],
        created_at: "2025-07-24T16:05:00".to_string(),
        description: Some("A simple agent that can perform addition.".to_string()),
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
            max_tokens: 100,
            enable_reasoner: true,
            max_reasoning_tokens: 2048,
        },
        message_buffer_autoclear: true,
        in_context_message_indices: vec![],
        messages: vec![],
        metadata: None,
        multi_agent_group: None,
        name: "simple_agent".to_string(),
        system: "You are a helpful AI assistant that can use tools to perform calculations.".to_string(),
        tags: vec![Tag { tag: "calculator".to_string() }],
        tool_exec_environment_variables: vec![],
        tool_rules: vec![],
        tools: vec![addition_tool],
        updated_at: "2025-07-24T16:05:00".to_string(),
        version: "0.1.0".to_string(),
    };

    // 4. Serialize the struct to a JSON string.
    let json_string = to_string(&agent)?;

    // 5. Save the JSON string to a file.
    let output_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples").join("simple_agent.af");
    fs::write(&output_path, json_string)?;

    println!("Successfully created agent file at: {:?}", output_path);

    Ok(())
}
