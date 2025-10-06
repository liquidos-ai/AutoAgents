use crate::{
    config::{parse_yaml_file, parse_yaml_str, validate_workflow, WorkflowConfig},
    error::Result,
    workflow::{Workflow, WorkflowOutput},
};
use std::path::Path;

/// Builder for constructing and executing workflows from YAML configurations
///
/// # Examples
///
/// ```no_run
/// use autoagents_serve::WorkflowBuilder;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let workflow = WorkflowBuilder::from_yaml_file("workflow.yaml")?
///         .build()?;
///     
///     let result = workflow.run("What is 2+2?".to_string()).await?;
///     println!("Result: {:?}", result);
///     Ok(())
/// }
/// ```
pub struct WorkflowBuilder {
    config: WorkflowConfig,
}

impl WorkflowBuilder {
    /// Create a new WorkflowBuilder from a YAML file path
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the YAML workflow configuration file
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be read
    /// - The YAML is malformed
    /// - The configuration is invalid
    pub fn from_yaml_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config = parse_yaml_file(path)?;
        validate_workflow(&config)?;
        Ok(Self { config })
    }

    /// Create a new WorkflowBuilder from a YAML string
    ///
    /// # Arguments
    ///
    /// * `yaml` - YAML string containing the workflow configuration
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The YAML is malformed
    /// - The configuration is invalid
    pub fn from_yaml_str(yaml: &str) -> Result<Self> {
        let config = parse_yaml_str(yaml)?;
        validate_workflow(&config)?;
        Ok(Self { config })
    }

    /// Build the workflow from the configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the workflow cannot be constructed from the configuration
    pub fn build(self) -> Result<BuiltWorkflow> {
        let workflow = Workflow::from_config(self.config)?;
        Ok(BuiltWorkflow { workflow })
    }
}

/// A fully constructed workflow ready for execution
pub struct BuiltWorkflow {
    workflow: Workflow,
}

impl BuiltWorkflow {
    /// Execute the workflow with the given input
    ///
    /// # Arguments
    ///
    /// * `input` - The input string to process through the workflow
    ///
    /// # Returns
    ///
    /// Returns the workflow output which can be either:
    /// - `WorkflowOutput::Single(String)` for Direct, Sequential, and Routing workflows
    /// - `WorkflowOutput::Multiple(Vec<String>)` for Parallel workflows
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The workflow execution fails
    /// - Any agent in the workflow fails
    /// - The workflow times out
    pub async fn run(&self, input: String) -> Result<WorkflowOutput> {
        self.workflow.execute(input).await
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
