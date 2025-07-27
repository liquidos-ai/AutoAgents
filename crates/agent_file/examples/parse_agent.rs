//! An example of parsing an .af file using the agent_file crate.

use agent_file::parser::from_str;
use std::env;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get the file path from the command-line arguments.
    let file_path_arg = env::args().nth(1).ok_or("Please provide a path to an agent file.")?;

    // The path might be relative, so we resolve it against the manifest directory
    // to handle calls like `cargo run --example parse_agent -- examples/my_agent.af`
    let agent_file_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(file_path_arg);

    println!("Attempting to parse file: {:?}", agent_file_path);

    // Read the file content into a string.
    let agent_json = fs::read_to_string(&agent_file_path)?;

    // Parse the JSON string using the library's from_str function.
    match from_str(&agent_json) {
        Ok(agent_file) => {
            println!("\nSuccessfully parsed agent file!");
            println!("--------------------------------");
            println!("Agent Name: {}", agent_file.name);
            println!("Agent Version: {}", agent_file.version);
            println!("Agent Description: {}", agent_file.description.unwrap_or_else(|| "N/A".to_string()));
            println!("\nLLM Configuration:");
            println!("  Model Handle: {}", agent_file.llm_config.handle);
            println!("  Context Window: {}", agent_file.llm_config.context_window);
            
            println!("\nCore Memory Blocks: {}", agent_file.core_memory.len());
            for block in agent_file.core_memory {
                println!("  - Label: '{}', Limit: {} chars", block.label, block.limit);
            }

            println!("\nTools: {}", agent_file.tools.len());
            for tool in agent_file.tools {
                println!("  - Tool: '{}' ({})", tool.name, tool.tool_type);
            }

            println!("\nTool Rules: {}", agent_file.tool_rules.len());
        }
        Err(e) => {
            eprintln!("\nFailed to parse agent file:");
            eprintln!("{}", e);
        }
    }

    Ok(())
}

