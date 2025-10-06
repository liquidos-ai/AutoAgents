use autoagents::init_logging;
use autoagents_serve::WorkflowBuilder;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_logging();

    println!("==============================================");
    println!("   AutoAgents Workflow Examples");
    println!("==============================================\n");

    // Example 1: Direct Workflow
    println!("Example 1: Direct Workflow (Single Agent)");
    println!("----------------------------------------------");
    run_workflow(
        "examples/serve/workflows/direct.yaml",
        "What is 15 multiplied by 8?",
    )
    .await?;

    println!("\n");

    // Example 2: Sequential Workflow
    println!("Example 2: Sequential Workflow (Chain)");
    println!("----------------------------------------------");
    run_workflow(
        "examples/serve/workflows/sequential.yaml",
        "Rust is a systems programming language that is blazingly fast and memory-efficient. It has no runtime or garbage collector and can power performance-critical services.",
    ).await?;

    println!("\n");

    // Example 3: Parallel Workflow
    println!("Example 3: Parallel Workflow (Concurrent)");
    println!("----------------------------------------------");
    run_workflow(
        "examples/serve/workflows/parallel.yaml",
        "Artificial Intelligence and Machine Learning",
    )
    .await?;

    println!("\n");

    // Example 4: Routing Workflow (Math)
    println!("Example 4a: Routing Workflow (Math Route)");
    println!("----------------------------------------------");
    run_workflow(
        "examples/serve/workflows/routing.yaml",
        "Calculate the sum of 25 and 17",
    )
    .await?;

    println!("\n");

    // Example 5: Routing Workflow (Creative)
    println!("Example 4b: Routing Workflow (Creative Route)");
    println!("----------------------------------------------");
    run_workflow(
        "examples/serve/workflows/routing.yaml",
        "Write a short poem about the ocean",
    )
    .await?;

    println!("\n==============================================");
    println!("   All examples completed successfully! ✓");
    println!("==============================================\n");

    Ok(())
}

async fn run_workflow(yaml_path: &str, input: &str) -> anyhow::Result<()> {
    println!("Loading workflow: {}", yaml_path);
    let workflow = WorkflowBuilder::from_yaml_file(yaml_path)?.build()?;

    println!("Input: {}", input);
    println!("Executing...\n");

    let result = workflow.run(input.to_string()).await?;

    match result {
        autoagents_serve::WorkflowOutput::Single(output) => {
            println!("Output:");
            println!("{}", output);
        }
        autoagents_serve::WorkflowOutput::Multiple(outputs) => {
            println!("Outputs from {} agents:", outputs.len());
            for (idx, output) in outputs.iter().enumerate() {
                println!("\n  Agent {}:", idx + 1);
                println!("  {}", output);
            }
        }
    }

    Ok(())
}
