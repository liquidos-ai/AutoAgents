use autoagents::init_logging;
use autoagents_serve::workflow::WorkflowStreamEvent;
use autoagents_serve::WorkflowBuilder;
use clap::{Parser, Subcommand};
use std::io::{self, Write};
use tokio_stream::StreamExt;

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
        /// Workflow path to YAML file
        #[arg(short, long)]
        workflow: String,

        /// Input text for the workflow
        #[arg(short, long)]
        input: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_logging();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Run { workflow, input }) => {
            run_specific_workflow(&workflow, &input).await?;
        }
        None => {
            panic!("Invalid Command")
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
        "".into()
    };

    run_workflow(&workflow_path, input).await?;

    println!("\n==============================================");
    println!("   Workflow completed successfully! ✓");
    println!("==============================================\n");

    Ok(())
}

async fn run_workflow(yaml_path: &str, input: &str) -> anyhow::Result<()> {
    let workflow = WorkflowBuilder::from_yaml_file(yaml_path)?.build()?;

    println!("Input: {}", input);
    println!("Executing...\n");

    if workflow.stream_enabled() {
        match workflow.run_stream(input.to_string()).await {
            Ok(mut stream) => {
                let mut aggregated = String::new();
                println!("Streaming output:\n");

                while let Some(event) = stream.next().await {
                    match event {
                        Ok(WorkflowStreamEvent::Chunk { content }) => {
                            print!("{}", content);
                            let _ = io::stdout().flush();
                            aggregated.push_str(&content);
                        }
                        Ok(WorkflowStreamEvent::ToolCall { tool_name, payload }) => {
                            println!(
                                "\n\n[tool-call] {} => {}\n",
                                tool_name,
                                serde_json::to_string_pretty(&payload).unwrap_or_default()
                            );
                        }
                        Ok(WorkflowStreamEvent::ToolCallComplete { tool_name, .. }) => {
                            println!("\n\n[tool-call-complete] {}\n", tool_name,);
                        }
                        Ok(WorkflowStreamEvent::Complete) => {
                            println!("\n\n[stream complete]");
                            break;
                        }
                        Err(e) => {
                            println!("\n\n[stream error] {}", e);
                            break;
                        }
                    }
                }

                if !aggregated.is_empty() {
                    println!("\nFinal response:\n{}", aggregated);
                }
                return Ok(());
            }
            Err(e) => {
                panic!(
                    "Streaming unsupported for this workflow run ({}). Falling back to blocking execution.\n",
                    e
                );
            }
        }
    }

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
