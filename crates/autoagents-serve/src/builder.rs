use crate::{
    config::{parse_yaml_file, parse_yaml_str, validate_workflow, WorkflowConfig},
    error::{Result, WorkflowError},
    workflow::{MemoryCache, ModelCache, Workflow, WorkflowOutput, WorkflowStream},
};
use std::path::Path;

/// Builder for constructing and executing workflows from YAML configurations
pub struct WorkflowBuilder {
    pub config: WorkflowConfig,
}

impl WorkflowBuilder {
    /// Create a new WorkflowBuilder from a YAML file path
    #[allow(clippy::result_large_err)]
    pub fn from_yaml_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config = parse_yaml_file(path)?;
        validate_workflow(&config)?;
        Ok(Self { config })
    }

    /// Create a new WorkflowBuilder from a YAML string
    #[allow(clippy::result_large_err)]
    pub fn from_yaml_str(yaml: &str) -> Result<Self> {
        let config = parse_yaml_str(yaml)?;
        validate_workflow(&config)?;
        Ok(Self { config })
    }

    /// Build the workflow from the configuration
    #[allow(clippy::result_large_err)]
    pub fn build(self) -> Result<BuiltWorkflow> {
        let workflow = Workflow::from_config(self.config.clone())?;
        let memory_persistence_enabled = self.config.memory_persistence.is_some();
        Ok(BuiltWorkflow {
            workflow,
            model_cache: None,
            memory_cache: None,
            workflow_name: self.config.name.clone(),
            memory_persistence_enabled,
            stream_enabled: self.config.stream,
        })
    }

    /// Build the workflow with a model cache for preloaded models
    #[allow(clippy::result_large_err)]
    pub fn build_with_cache(self, model_cache: ModelCache) -> Result<BuiltWorkflow> {
        let workflow = Workflow::from_config(self.config.clone())?;
        let memory_persistence_enabled = self.config.memory_persistence.is_some();
        Ok(BuiltWorkflow {
            workflow,
            model_cache: Some(model_cache.clone()),
            memory_cache: None,
            workflow_name: self.config.name.clone(),
            memory_persistence_enabled,
            stream_enabled: self.config.stream,
        })
    }

    /// Build the workflow with both model and memory caches
    #[allow(clippy::result_large_err)]
    pub fn build_with_caches(
        self,
        model_cache: ModelCache,
        memory_cache: MemoryCache,
    ) -> Result<BuiltWorkflow> {
        let workflow = Workflow::from_config(self.config.clone())?;
        let memory_persistence_enabled = self.config.memory_persistence.is_some();
        Ok(BuiltWorkflow {
            workflow,
            model_cache: Some(model_cache),
            memory_cache: Some(memory_cache),
            workflow_name: self.config.name.clone(),
            memory_persistence_enabled,
            stream_enabled: self.config.stream,
        })
    }
}

/// A fully constructed workflow ready for execution
pub struct BuiltWorkflow {
    workflow: Workflow,
    pub model_cache: Option<ModelCache>,
    pub memory_cache: Option<MemoryCache>,
    pub workflow_name: Option<String>,
    pub memory_persistence_enabled: bool,
    pub stream_enabled: bool,
}

impl BuiltWorkflow {
    pub fn stream_enabled(&self) -> bool {
        self.stream_enabled && self.workflow.stream_enabled()
    }

    /// Execute the workflow with the given input
    pub async fn run(&self, input: String) -> Result<WorkflowOutput> {
        self.workflow
            .execute(
                input,
                self.model_cache.as_ref(),
                self.memory_cache.as_ref(),
                self.workflow_name.as_deref(),
                self.memory_persistence_enabled,
            )
            .await
    }

    pub async fn run_stream(&self, input: String) -> Result<WorkflowStream> {
        if !self.stream_enabled() {
            return Err(WorkflowError::ExecutionError(
                "Streaming is not enabled for this workflow".to_string(),
            ));
        }

        self.workflow
            .execute_stream(
                input,
                self.model_cache.as_ref(),
                self.memory_cache.as_ref(),
                self.workflow_name.as_deref(),
                self.memory_persistence_enabled,
            )
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_from_yaml_str() {
        let yaml = r#"
kind: Direct
workflow:
  agent:
    name: test_agent
    description: A test agent
    model:
      kind: llm
      backend:
        kind: Cloud
      provider: OpenAI
      model_name: gpt-4
    tools: []
"#;
        let result = WorkflowBuilder::from_yaml_str(yaml);
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_invalid_yaml() {
        let yaml = r#"
kind: Direct
workflow:
  # Missing agent configuration
"#;
        let result = WorkflowBuilder::from_yaml_str(yaml);
        assert!(result.is_err());
    }
}
