pub mod api;
pub mod state;

use crate::{error::Result, WorkflowBuilder};
use api::create_router;
use state::AppState;
use std::path::Path;
use tower_http::{cors::CorsLayer, trace::TraceLayer};

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
        }
    }
}

/// Start the HTTP server with the given workflows
///
/// # Arguments
///
/// * `config` - Server configuration (host, port)
/// * `workflow_paths` - Map of workflow names to YAML file paths
///
/// # Example
///
/// ```no_run
/// use autoagents_serve::{serve, ServerConfig};
/// use std::collections::HashMap;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let config = ServerConfig::default();
///     let mut workflows = HashMap::new();
///     workflows.insert("my_workflow".to_string(), "workflow.yaml".to_string());
///
///     serve(config, workflows).await?;
///     Ok(())
/// }
/// ```
pub async fn serve(
    config: ServerConfig,
    workflow_paths: std::collections::HashMap<String, String>,
) -> Result<()> {
    log::info!("Initializing AutoAgents HTTP server");
    log::debug!("Server configuration: {:?}", config);

    let state = AppState::new();

    // Load all workflows
    log::info!("Loading {} workflow(s)", workflow_paths.len());
    for (name, path) in workflow_paths {
        log::info!("Loading workflow '{}' from '{}'", name, path);

        match WorkflowBuilder::from_yaml_file(Path::new(&path)) {
            Ok(builder) => match builder.build() {
                Ok(workflow) => {
                    state.add_workflow(name.clone(), workflow).await;
                    log::info!("✓ Workflow '{}' loaded successfully", name);
                }
                Err(e) => {
                    log::error!("✗ Failed to build workflow '{}': {}", name, e);
                    return Err(e);
                }
            },
            Err(e) => {
                log::error!(
                    "✗ Failed to parse workflow '{}' from '{}': {}",
                    name,
                    path,
                    e
                );
                return Err(e);
            }
        }
    }

    let workflows = state.list_workflows().await;
    log::info!("Total workflows loaded: {}", workflows.len());
    for workflow_name in workflows {
        log::info!("  - {}", workflow_name);
    }

    // Create router with middleware
    let app = create_router(state)
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive());

    let addr = format!("{}:{}", config.host, config.port);
    log::info!("Starting HTTP server on {}", addr);
    log::info!("Server is ready to accept connections");
    log::info!("Available endpoints:");
    log::info!("  - GET  http://{}/health", addr);
    log::info!("  - GET  http://{}/api/v1/workflows", addr);
    log::info!("  - POST http://{}/api/v1/workflows/:name/execute", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| {
        log::error!("Failed to bind to {}: {}", addr, e);
        crate::error::WorkflowError::IoError(e)
    })?;

    log::info!("🚀 Server started successfully!");

    if let Err(e) = axum::serve(listener, app).await {
        log::error!("Server error: {}", e);
        return Err(crate::error::WorkflowError::ExecutionError(e.to_string()));
    }

    Ok(())
}
