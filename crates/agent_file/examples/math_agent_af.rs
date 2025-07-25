//! Example of creating a simple math agent and exporting it to an .af file

use agent_file::export::AgentFileExporter;
use agent_file::schema::*;
use chrono::Utc;
// Removed unused import
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new agent file exporter
    let exporter = AgentFileExporter::new()
        .agent_type("math_agent")
        .name("math_agent")
        .description("A simple math agent that can perform addition")
        .system_prompt("You are a helpful math assistant that can perform addition.")
        .version("0.1.0")
        .add_memory_block(CoreMemoryBlock {
            created_at: Utc::now().to_rfc3339(),
            updated_at: Utc::now().to_rfc3339(),
            label: "persona".to_string(),
            value: "You are a helpful math assistant that can perform addition.".to_string(),
            description: None,
            is_template: false,
            limit: 5000,
            metadata: None,
            template_name: None,
        })
        .add_memory_block(CoreMemoryBlock {
            created_at: Utc::now().to_rfc3339(),
            updated_at: Utc::now().to_rfc3339(),
            label: "human".to_string(),
            value: "User information will be stored here.".to_string(),
            description: None,
            is_template: false,
            limit: 5000,
            metadata: None,
            template_name: None,
        })
        .llm_config(LlmConfig {
            model: "gpt-3.5-turbo".to_string(),
            model_endpoint_type: "openai".to_string(),
            model_endpoint: "https://api.openai.com/v1".to_string(),
            model_wrapper: None,
            context_window: 4096,
            put_inner_thoughts_in_kwargs: false,
            handle: "openai/gpt-3.5-turbo".to_string(),
            temperature: 0.7,
            max_tokens: 1024,
            enable_reasoner: false,
            max_reasoning_tokens: 0,
        })
        .embedding_config(EmbeddingConfig {
            embedding_endpoint_type: "none".to_string(),
            embedding_endpoint: "".to_string(),
            embedding_model: "".to_string(),
            embedding_dim: 0,
            embedding_chunk_size: 0,
            handle: "".to_string(),
            azure_endpoint: None,
            azure_version: None,
            azure_deployment: None,
        })
        .add_tool(Tool {
            name: "add_numbers".to_string(),
            description: "Add two numbers together".to_string(),
            source_type: "rust".to_string(),
            source_code: Some("pub fn add_numbers(left: i32, right: i32) -> i32 {\n    left + right\n}".to_string()),
            tool_type: "function".to_string(),
            json_schema: ToolJsonSchema {
                name: "add_numbers".to_string(),
                description: "Add two numbers together".to_string(),
                parameters: Parameters {
                    param_type: Some("object".to_string()),
                    properties: {
                        let mut map = std::collections::HashMap::new();
                        map.insert("left".to_string(), ParameterProperties {
                            param_type: "integer".to_string(),
                            description: Some("Left operand".to_string()),
                        });
                        map.insert("right".to_string(), ParameterProperties {
                            param_type: "integer".to_string(),
                            description: Some("Right operand".to_string()),
                        });
                        map
                    },
                    required: vec!["left".to_string(), "right".to_string()],
                },
                required: Some(vec!["parameters".to_string()]),
                schema_type: Some("object".to_string()),
            },
            args_json_schema: None,
            return_char_limit: 1000,
            tags: vec!["math".to_string()],
            created_at: Utc::now().to_rfc3339(),
            updated_at: Utc::now().to_rfc3339(),
            metadata: None,
        })
        .message_buffer_autoclear(false)
        .in_context_message_indices(vec![0]);
    // Define the output directory and filename
    let output_dir = "/home/harshalr/yc/testing/task/AutoAgents/crates/agent_file";
    let filename = "math_agent.af";
    let output_path = PathBuf::from(output_dir).join(filename);
    
    // Create the directory if it doesn't exist
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    
    // Save to file
    exporter.save_to_file(&output_path)?;
    
    println!("Successfully created math agent file at: {}", output_path.display());
    Ok(())
}
