use super::schema::WorkflowConfig;
use crate::error::Result;
use std::path::Path;

pub fn parse_yaml_file<P: AsRef<Path>>(path: P) -> Result<WorkflowConfig> {
    let content = std::fs::read_to_string(path)?;
    parse_yaml_str(&content)
}

pub fn parse_yaml_str(yaml: &str) -> Result<WorkflowConfig> {
    let config: WorkflowConfig = serde_yaml::from_str(yaml)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_direct_workflow() {
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
        let result = parse_yaml_str(yaml);
        assert!(result.is_ok());
    }
}
