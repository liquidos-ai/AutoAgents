use crate::protocol::ActorID;
use autoagents_llm::chat::StructuredOutputFormat;

#[derive(Clone, Default)]
pub struct AgentConfig {
    /// The agent's name
    pub name: String,
    /// The agent's description
    pub description: String,
    /// The Agent ID
    pub id: ActorID,
    /// The output schema for the agent
    pub output_schema: Option<StructuredOutputFormat>,
}

impl AgentConfig {
    pub fn new(name: String, description: String) -> Self {
        Self {
            name,
            description,
            id: ActorID::new_v4(),
            output_schema: None,
        }
    }

    pub fn with_output_schema(mut self, schema: StructuredOutputFormat) -> Self {
        self.output_schema = Some(schema);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_config_default() {
        let config = AgentConfig::default();
        assert!(config.name.is_empty());
        assert!(config.description.is_empty());
        assert!(config.id.is_nil());
        assert!(config.output_schema.is_none());
    }

    #[test]
    fn test_agent_config_new() {
        let name = "TestAgent".to_string();
        let description = "A test agent for unit tests".to_string();
        let config = AgentConfig::new(name.clone(), description.clone());

        assert_eq!(config.name, name);
        assert_eq!(config.description, description);
        assert!(!config.id.is_nil());
        assert!(config.output_schema.is_none());
    }

    #[test]
    fn test_agent_config_with_output_schema() {
        let config = AgentConfig::new("Agent".to_string(), "Description".to_string());
        let schema = StructuredOutputFormat {
            name: "TestSchema".to_string(),
            description: Some("Test schema".to_string()),
            schema: Some(serde_json::json!({"type": "object"})),
            strict: Some(true),
        };
        let config_with_schema = config.with_output_schema(schema.clone());

        assert!(config_with_schema.output_schema.is_some());
        if let Some(actual_schema) = config_with_schema.output_schema {
            assert_eq!(actual_schema.name, "TestSchema");
        }
    }

    #[test]
    fn test_agent_config_clone() {
        let original = AgentConfig::new("Original".to_string(), "Original description".to_string());
        let cloned = original.clone();

        assert_eq!(original.name, cloned.name);
        assert_eq!(original.description, cloned.description);
        assert_eq!(original.id, cloned.id);
    }

    #[test]
    fn test_agent_config_unique_ids() {
        let config1 = AgentConfig::new("Agent1".to_string(), "Description1".to_string());
        let config2 = AgentConfig::new("Agent2".to_string(), "Description2".to_string());

        assert_ne!(config1.id, config2.id);
    }
}
