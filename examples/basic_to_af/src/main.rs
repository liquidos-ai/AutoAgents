use autoagents_core::agent_file::AgentFileExporter;
use autoagents_core::agent_file::schema::ParameterProperties;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Converting AutoAgents examples to .af files...\n");

    // Convert the simple math agent from examples/basic/src/simple.rs
    create_simple_math_agent()?;
    
    // Convert the chaining agents from examples/basic/src/chaining.rs
    create_chaining_agents()?;

    println!("✅ All .af files created successfully!");
    println!("\nGenerated files:");
    println!("  - simple_math_agent.af (from examples/basic/src/simple.rs)");
    println!("  - agent_1_math_geek.af (from examples/basic/src/chaining.rs)");
    println!("  - agent_2_math_professor.af (from examples/basic/src/chaining.rs)");
    
    Ok(())
}

fn create_simple_math_agent() -> Result<(), Box<dyn std::error::Error>> {
    println!("Creating simple_math_agent.af (from examples/basic/src/simple.rs)...");
    
    // Create parameters for the Addition tool (matching simple.rs)
    let mut addition_params = HashMap::new();
    addition_params.insert("left".to_string(), ParameterProperties {
        param_type: "integer".to_string(),
        description: Some("Left Operand for addition".to_string()),
    });
    addition_params.insert("right".to_string(), ParameterProperties {
        param_type: "integer".to_string(),
        description: Some("Right Operand for addition".to_string()),
    });

    // Create the agent using the programmatic builder (matching MathAgent from simple.rs)
    let agent_file = AgentFileExporter::new()
        .name("math_agent")
        .description("You are a Math agent")
        .system_prompt("You are a Math agent that can perform addition operations. When asked to add numbers, use the Addition tool.")
        .add_memory_block(AgentFileExporter::create_memory_block("persona", "You are a helpful math assistant.", 5000))
        .add_memory_block(AgentFileExporter::create_memory_block("human", "User information will be stored here.", 5000))
        .add_tool(AgentFileExporter::create_simple_tool(
            "Addition",
            "Use this tool to Add two numbers",
            addition_params,
            vec!["left".to_string(), "right".to_string()],
            Some("#[derive(Serialize, Deserialize, ToolInput, Debug)]\npub struct AdditionArgs {\n    #[input(description = \"Left Operand for addition\")]\n    left: i64,\n    #[input(description = \"Right Operand for addition\")]\n    right: i64,\n}\n\n#[tool(\n    name = \"Addition\",\n    description = \"Use this tool to Add two numbers\",\n    input = AdditionArgs,\n)]\nfn add(args: AdditionArgs) -> Result<i64, ToolCallError> {\n    Ok(args.left + args.right)\n}".to_string()),
            "rust",
            "custom"
        ))
        .add_tool_rule(AgentFileExporter::create_tool_rule("run_first", "Addition"))
        .add_tool_rule(AgentFileExporter::create_tool_rule("exit_loop", "Addition"))
        .add_system_message("You are a Math agent that can perform addition operations. When asked to add numbers, use the Addition tool.")
        .add_user_message("What is 2 + 2?")
        .add_user_message("What did I ask before?")
        .build();

    // Save to file
    std::fs::write("simple_math_agent.af", serde_json::to_string_pretty(&agent_file)?)?;
    
    println!("  ✓ Created simple_math_agent.af");
    Ok(())
}

fn create_chaining_agents() -> Result<(), Box<dyn std::error::Error>> {
    println!("Creating chaining_agents.af (from examples/basic/src/chaining.rs)...");
    
    // Create Agent1 (math geek) from chaining.rs
    let agent1_file = AgentFileExporter::new()
        .name("agent_1")
        .description("You are a math geek and expert in linear algebra")
        .system_prompt("You are a math geek and expert in linear algebra. You help with mathematical problems and linear algebra concepts.")
        .add_memory_block(AgentFileExporter::create_memory_block("persona", "You are a math geek and expert in linear algebra.", 5000))
        .add_memory_block(AgentFileExporter::create_memory_block("human", "User information will be stored here.", 5000))
        .add_system_message("You are a math geek and expert in linear algebra. You help with mathematical problems and linear algebra concepts.")
        .add_user_message("Explain the concept of eigenvalues in linear algebra")
        .build();

    // Create Agent2 (math professor) from chaining.rs
    let agent2_file = AgentFileExporter::new()
        .name("agent_2")
        .description("You are a math professor in linear algebra, Your goal is to review the given content if correct")
        .system_prompt("You are a math professor in linear algebra. Your goal is to review the given content and verify if it's correct.")
        .add_memory_block(AgentFileExporter::create_memory_block("persona", "You are a math professor in linear algebra.", 5000))
        .add_memory_block(AgentFileExporter::create_memory_block("human", "User information will be stored here.", 5000))
        .add_system_message("You are a math professor in linear algebra. Your goal is to review the given content and verify if it's correct.")
        .add_user_message("Review this explanation of eigenvalues")
        .build();

    // Save both agents to separate files
    std::fs::write("agent_1_math_geek.af", serde_json::to_string_pretty(&agent1_file)?)?;
    std::fs::write("agent_2_math_professor.af", serde_json::to_string_pretty(&agent2_file)?)?;
    
    println!("  ✓ Created agent_1_math_geek.af");
    println!("  ✓ Created agent_2_math_professor.af");
    Ok(())
} 