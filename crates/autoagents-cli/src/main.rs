use anyhow::{Context, Result};
use autoagents_serve::{serve, ServerConfig, WorkflowBuilder};
use clap::{Parser, Subcommand};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "autoagents")]
#[command(about = "AutoAgents CLI - Run and serve agent workflows", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a workflow from a YAML file
    Run {
        /// Path to the workflow YAML file
        #[arg(short, long)]
        workflow: PathBuf,

        /// Input text for the workflow
        #[arg(short, long)]
        input: String,
    },
    /// Serve workflows over HTTP REST API
    Serve {
        /// Path to a single workflow YAML file (conflicts with --directory)
        #[arg(short, long, conflicts_with = "directory")]
        workflow: Option<PathBuf>,

        /// Path to a directory containing workflow YAML files (conflicts with --workflow)
        #[arg(short, long, conflicts_with = "workflow")]
        directory: Option<PathBuf>,

        /// Name for the workflow (only used with --workflow, defaults to filename)
        #[arg(short, long)]
        name: Option<String>,

        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to bind to
        #[arg(short, long, default_value = "8080")]
        port: u16,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run { workflow, input } => {
            run_workflow(workflow, input).await?;
        }
        Commands::Serve {
            workflow,
            directory,
            name,
            host,
            port,
        } => {
            if workflow.is_none() && directory.is_none() {
                anyhow::bail!("Either --workflow or --directory must be specified");
            }
            serve_workflows(workflow, directory, name, host, port).await?;
        }
    }

    Ok(())
}

async fn run_workflow(workflow_path: PathBuf, input: String) -> Result<()> {
    log::info!("Loading workflow from {:?}", workflow_path);

    let workflow = WorkflowBuilder::from_yaml_file(&workflow_path)?.build()?;

    log::info!("Executing workflow with input: {}", input);
    let result = workflow.run(input).await?;

    println!("\n========== Workflow Result ==========");
    match result {
        autoagents_serve::WorkflowOutput::Single(output) => {
            println!("{}", output);
        }
        autoagents_serve::WorkflowOutput::Multiple(outputs) => {
            for (idx, output) in outputs.iter().enumerate() {
                println!("\n--- Agent {} Output ---", idx + 1);
                println!("{}", output);
            }
        }
    }
    println!("=====================================\n");

    Ok(())
}

async fn serve_workflows(
    workflow_path: Option<PathBuf>,
    directory_path: Option<PathBuf>,
    name: Option<String>,
    host: String,
    port: u16,
) -> Result<()> {
    let config = ServerConfig { host, port };
    let mut workflows = HashMap::new();

    if let Some(workflow_path) = workflow_path {
        // Single workflow mode
        let workflow_name = name.unwrap_or_else(|| {
            workflow_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("default")
                .to_string()
        });

        workflows.insert(
            workflow_name.clone(),
            workflow_path.to_string_lossy().to_string(),
        );

        log::info!(
            "Starting HTTP server with workflow '{}' at {}:{}",
            workflow_name,
            config.host,
            config.port
        );
    } else if let Some(directory_path) = directory_path {
        // Directory mode
        log::info!(
            "Scanning directory for workflow files: {}",
            directory_path.display()
        );

        workflows = scan_workflow_directory(&directory_path)
            .context("Failed to scan workflow directory")?;

        if workflows.is_empty() {
            anyhow::bail!(
                "No workflow YAML files found in directory: {}",
                directory_path.display()
            );
        }

        log::info!(
            "Starting HTTP server with {} workflows at {}:{}",
            workflows.len(),
            config.host,
            config.port
        );
        log::info!("Loaded workflows:");
        for workflow_name in workflows.keys() {
            log::info!("  - {}", workflow_name);
        }
    }

    log::info!("Endpoints:");
    log::info!("  GET  /health");
    log::info!("  GET  /api/v1/workflows");
    for workflow_name in workflows.keys() {
        log::info!("  POST /api/v1/workflows/{}/execute", workflow_name);
    }

    serve(config, workflows).await?;

    Ok(())
}

fn scan_workflow_directory(directory_path: &PathBuf) -> Result<HashMap<String, String>> {
    use std::fs;

    let mut workflows = HashMap::new();

    if !directory_path.exists() {
        anyhow::bail!("Directory does not exist: {}", directory_path.display());
    }

    if !directory_path.is_dir() {
        anyhow::bail!("Path is not a directory: {}", directory_path.display());
    }

    let entries = fs::read_dir(directory_path).context("Failed to read directory")?;

    for entry in entries {
        let entry = entry.context("Failed to read directory entry")?;
        let path = entry.path();

        // Only process files with .yaml or .yml extension
        if path.is_file() {
            if let Some(extension) = path.extension() {
                if extension == "yaml" || extension == "yml" {
                    if let Some(file_name) = path.file_stem() {
                        if let Some(name) = file_name.to_str() {
                            let workflow_name = name.to_string();
                            let workflow_path = path.to_string_lossy().to_string();

                            log::debug!("Found workflow: {} -> {}", workflow_name, workflow_path);
                            workflows.insert(workflow_name, workflow_path);
                        }
                    }
                }
            }
        }
    }

    Ok(workflows)
}
