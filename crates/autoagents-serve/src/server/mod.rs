pub mod api;
pub mod state;

use crate::config::ModelConfig;
use crate::{
    config::WorkflowConfig, error::Result, workflow::llm_factory::LLMFactory, WorkflowBuilder,
};
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

    // Load all workflows and preload models
    log::info!("Loading {} workflow(s)", workflow_paths.len());

    let mut workflow_configs: Vec<(String, WorkflowConfig, String)> = Vec::new();

    for (name, path) in workflow_paths {
        log::info!("Loading workflow '{}' from '{}'", name, path);

        match WorkflowBuilder::from_yaml_file(Path::new(&path)) {
            Ok(builder) => {
                // Store config for model preloading
                workflow_configs.push((name.clone(), builder.config.clone(), path.clone()));

                match builder
                    .build_with_caches(state.model_cache.clone(), state.memory_cache.clone())
                {
                    Ok(workflow) => {
                        state.add_workflow(name.clone(), workflow).await;
                        log::info!("Workflow '{}' loaded successfully", name);
                    }
                    Err(e) => {
                        log::error!("Failed to build workflow '{}': {}", name, e);
                        return Err(e);
                    }
                }
            }
            Err(e) => {
                log::error!("Failed to parse workflow '{}' from '{}': {}", name, path, e);
                return Err(e);
            }
        }
    }

    // Preload models that have preload=true
    log::info!("Checking for models to preload...");
    preload_models(&state, &workflow_configs).await?;

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

    log::info!("Server started successfully!");

    if let Err(e) = axum::serve(listener, app).await {
        log::error!("Server error: {}", e);
        return Err(crate::error::WorkflowError::ExecutionError(e.to_string()));
    }

    Ok(())
}

/// Preload models that have preload=true in their configuration
async fn preload_models(
    state: &AppState,
    workflow_configs: &[(String, WorkflowConfig, String)],
) -> Result<()> {
    use crate::config::{ModelConfig, WorkflowKind};

    let mut models_to_preload: std::collections::HashMap<String, ModelConfig> =
        std::collections::HashMap::new();

    // Extract all models that need preloading
    for (_workflow_name, config, _path) in workflow_configs {
        match &config.kind {
            WorkflowKind::Direct => {
                if let Some(agent) = &config.workflow.agent {
                    if agent.model.preload {
                        let key = generate_model_key(&agent.model);
                        models_to_preload.insert(key, agent.model.clone());
                    }
                }
            }
            WorkflowKind::Sequential | WorkflowKind::Parallel => {
                if let Some(agents) = &config.workflow.agents {
                    for agent in agents {
                        if agent.model.preload {
                            let key = generate_model_key(&agent.model);
                            models_to_preload.insert(key, agent.model.clone());
                        }
                    }
                }
            }
            WorkflowKind::Routing => {
                // Check router model
                if let Some(router) = &config.workflow.router {
                    if router.model.preload {
                        let key = generate_model_key(&router.model);
                        models_to_preload.insert(key, router.model.clone());
                    }
                }
                // Check handler models
                if let Some(handlers) = &config.workflow.handlers {
                    for handler in handlers {
                        if handler.agent.model.preload {
                            let key = generate_model_key(&handler.agent.model);
                            models_to_preload.insert(key, handler.agent.model.clone());
                        }
                    }
                }
            }
        }
    }

    if models_to_preload.is_empty() {
        log::info!("No models configured for preloading");
        return Ok(());
    }

    log::info!("Preloading {} unique model(s)...", models_to_preload.len());

    for (key, model_config) in models_to_preload {
        log::info!(
            "  Preloading model: {} (provider: {})",
            key,
            model_config.provider
        );

        match LLMFactory::create_llm(&model_config).await {
            Ok(model) => {
                state.add_model(key.clone(), model).await;
                log::info!("    ✓ Model '{}' preloaded successfully", key);
            }
            Err(e) => {
                log::error!("    ✗ Failed to preload model '{}': {}", key, e);
                return Err(e);
            }
        }
    }

    log::info!("All models preloaded successfully");
    Ok(())
}

/// Generate a unique key for a model configuration
pub fn generate_model_key(config: &ModelConfig) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Create a deterministic hash based on model configuration
    let mut hasher = DefaultHasher::new();

    config.provider.hash(&mut hasher);
    config.backend.kind.hash(&mut hasher);

    if let Some(model_name) = &config.model_name {
        model_name.hash(&mut hasher);
    }

    if let Some(source) = &config.source {
        source.hash(&mut hasher);
    }

    if let Some(params) = &config.parameters {
        if let Some(model_dir) = &params.model_dir {
            model_dir.hash(&mut hasher);
        }
        if let Some(quant) = &params.quant {
            quant.hash(&mut hasher);
        }
    }

    let hash = hasher.finish();

    // Create a readable key
    let provider = &config.provider;
    let model_id = config
        .model_name
        .as_ref()
        .or(config.source.as_ref())
        .map(|s| {
            // Take last part of path/repo
            s.split('/').last().unwrap_or(s).to_string()
        })
        .unwrap_or_else(|| "unknown".to_string());

    format!("{}_{}_{}", provider, model_id, &format!("{:x}", hash)[..8])
}
