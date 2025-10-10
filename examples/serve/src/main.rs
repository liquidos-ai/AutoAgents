use autoagents::init_logging;
use autoagents_serve::WorkflowBuilder;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "workflow-runner")]
#[command(about = "AutoAgents Workflow Runner - Run workflow examples", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a specific workflow
    Run {
        /// Workflow to run: direct, sequential, parallel, routing, local, or path to YAML file
        #[arg(short, long)]
        workflow: String,

        /// Input text for the workflow
        #[arg(short, long)]
        input: String,
    },
    /// Run all example workflows
    All,
    /// List available workflows
    List,
}

// Workflow definitions with default inputs
struct WorkflowExample {
    name: &'static str,
    path: &'static str,
    description: &'static str,
    default_inputs: &'static [&'static str],
}

fn get_workflows() -> Vec<WorkflowExample> {
    vec![
        WorkflowExample {
            name: "direct",
            path: "examples/serve/workflows/direct.yaml",
            description: "Single agent execution",
            default_inputs: &["What is 15 multiplied by 8?"],
        },
        WorkflowExample {
            name: "sequential",
            path: "examples/serve/workflows/sequential.yaml",
            description: "Sequential chain of agents",
            default_inputs: &["Rust is a systems programming language that is blazingly fast and memory-efficient. It has no runtime or garbage collector and can power performance-critical services."],
        },
        WorkflowExample {
            name: "parallel",
            path: "examples/serve/workflows/parallel.yaml",
            description: "Parallel execution of multiple agents",
            default_inputs: &["Artificial Intelligence and Machine Learning"],
        },
        WorkflowExample {
            name: "routing",
            path: "examples/serve/workflows/routing.yaml",
            description: "Router-based workflow with conditional handlers",
            default_inputs: &[
                "Calculate the sum of 25 and 17",
                "Write a short poem about the ocean",
            ],
        },
        WorkflowExample {
            name: "local",
            path: "examples/serve/workflows/local.yaml",
            description: "Local model using MistralRs",
            default_inputs: &["What is 5 + 5?"],
        },
        WorkflowExample {
            name: "research",
            path: "examples/serve/workflows/research.yaml",
            description: "Research Agent",
            default_inputs: &["What is 5 + 5?"],
        },
    ]
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_logging();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Run { workflow, input }) => {
            run_specific_workflow(&workflow, &input).await?;
        }
        Some(Commands::All) => {
            run_all_workflows().await?;
        }
        Some(Commands::List) => {
            list_workflows();
        }
        None => {
            // Default: run all workflows
            run_all_workflows().await?;
        }
    }

    Ok(())
}

async fn run_specific_workflow(workflow_name: &str, input: &str) -> anyhow::Result<()> {
    println!("==============================================");
    println!("   AutoAgents Workflow Runner");
    println!("==============================================\n");

    // Check if it's a path to a YAML file
    let workflow_path = if workflow_name.ends_with(".yaml") || workflow_name.ends_with(".yml") {
        workflow_name.to_string()
    } else {
        // Look up predefined workflow
        let workflows = get_workflows();
        let workflow = workflows
            .iter()
            .find(|w| w.name == workflow_name.to_lowercase())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Unknown workflow: {}. Use 'list' command to see available workflows.",
                    workflow_name
                )
            })?;
        workflow.path.to_string()
    };

    run_workflow(&workflow_path, input, Some(workflow_name)).await?;

    println!("\n==============================================");
    println!("   Workflow completed successfully! ✓");
    println!("==============================================\n");

    Ok(())
}

async fn run_all_workflows() -> anyhow::Result<()> {
    println!("==============================================");
    println!("   AutoAgents Workflow Examples");
    println!("==============================================\n");

    let workflows = get_workflows();
    for (idx, workflow) in workflows.iter().enumerate() {
        for (input_idx, input) in workflow.default_inputs.iter().enumerate() {
            let title = if workflow.default_inputs.len() > 1 {
                format!(
                    "Example {}.{}: {} - {}",
                    idx + 1,
                    (b'a' + input_idx as u8) as char,
                    workflow.name,
                    workflow.description
                )
            } else {
                format!(
                    "Example {}: {} - {}",
                    idx + 1,
                    workflow.name,
                    workflow.description
                )
            };

            println!("{}", title);
            println!("{}", "-".repeat(title.len() + 3));
            run_workflow(workflow.path, input, Some(workflow.name)).await?;
            println!("\n");
        }
    }

    println!("==============================================");
    println!("   All examples completed successfully! ✓");
    println!("==============================================\n");

    Ok(())
}

fn list_workflows() {
    println!("==============================================");
    println!("   Available Workflows");
    println!("==============================================\n");

    let workflows = get_workflows();
    for workflow in workflows {
        println!("{}", workflow.name);
        println!("   Description: {}", workflow.description);
        println!("   Path: {}", workflow.path);
        println!("   Default inputs:");
        for (idx, input) in workflow.default_inputs.iter().enumerate() {
            let preview = if input.len() > 60 {
                format!("{}...", &input[..60])
            } else {
                input.to_string()
            };
            println!("     {}. {}", idx + 1, preview);
        }
        println!();
    }

    println!("Usage examples:");
    println!("  cargo run --example serve -- run -w direct -i \"What is 5 + 3?\"");
    println!("  cargo run --example serve -- run -w routing -i \"Write a haiku\"");
    println!("  cargo run --example serve -- run -w path/to/custom.yaml -i \"Your input\"");
    println!("  cargo run --example serve -- all");
    println!();
}

async fn run_workflow(
    yaml_path: &str,
    input: &str,
    workflow_name: Option<&str>,
) -> anyhow::Result<()> {
    if let Some(name) = workflow_name {
        println!("Loading workflow: {} ({})", name, yaml_path);
    } else {
        println!("Loading workflow: {}", yaml_path);
    }

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
