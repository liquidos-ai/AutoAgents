pub mod api;
pub mod state;

use crate::utils::generate_model_key;
use crate::{
    config::WorkflowConfig, error::Result, workflow::llm_factory::LLMFactory, WorkflowBuilder,
};
use api::create_router;
use state::AppState;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
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

pub struct HTTPServer {
    state: Arc<AppState>,
    config: ServerConfig,
    workflows: HashMap<String, String>,
}

impl HTTPServer {
    pub fn new(config: ServerConfig, workflows: HashMap<String, String>) -> Self {
        Self {
            state: Arc::new(AppState::new()),
            config,
            workflows,
        }
    }

    // Start the HTTP server with the given workflows
    pub async fn serve(&self) -> Result<()> {
        log::info!("Initializing AutoAgents HTTP server");
        log::debug!("Server configuration: {:?}", self.config);

        // Load all workflows and preload models
        log::info!("Loading {} workflow(s)", self.workflows.len());

        let mut workflow_configs: Vec<(String, WorkflowConfig, String)> = Vec::new();

        for (name, path) in self.workflows.iter() {
            log::info!("Loading workflow '{}' from '{}'", name, path);

            match WorkflowBuilder::from_yaml_file(Path::new(&path)) {
                Ok(builder) => {
                    // Store config for model preloading
                    workflow_configs.push((name.clone(), builder.config.clone(), path.clone()));

                    match builder.build_with_cache(self.state.model_cache.clone()) {
                        Ok(workflow) => {
                            self.state.add_workflow(name.clone(), workflow).await;
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

        self.preload_models(&workflow_configs).await?;

        let workflows = self.state.list_workflows().await;
        log::info!("Total workflows loaded: {}", workflows.len());
        for workflow_name in workflows {
            log::info!("  - {}", workflow_name);
        }

        // Create router with middleware
        let app = create_router(self.state.clone())
            .layer(TraceLayer::new_for_http())
            .layer(CorsLayer::permissive());

        let addr = format!("{}:{}", self.config.host, self.config.port);
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
        &self,
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
                    self.state.add_model(key.clone(), model).await;
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
}
