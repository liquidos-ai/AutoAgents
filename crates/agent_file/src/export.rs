//! Provides a builder pattern for exporting AgentFile with flexible options.

use crate::schema::{AgentFile, CoreMemoryBlock, EmbeddingConfig, LlmConfig, Message, Tag, Tool, ToolEnvVar, ToolRule};
use chrono::Utc;
use serde_json::Value;
use std::collections::HashMap;

/// Builder for creating and exporting AgentFile instances
#[derive(Debug, Clone)]
pub struct AgentFileExporter {
    agent_file: AgentFile,
    embedding_config: Option<EmbeddingConfig>,
    llm_config: Option<LlmConfig>,
    tools: Vec<Tool>,
    tool_rules: Vec<ToolRule>,
    tool_env_vars: Vec<ToolEnvVar>,
}

impl Default for AgentFileExporter {
    fn default() -> Self {
        let now = Utc::now().to_rfc3339();
        
        Self {
            agent_file: AgentFile {
                agent_type: "memgpt_agent".to_string(),
                core_memory: Vec::new(),
                created_at: now.clone(),
                description: None,
                embedding_config: EmbeddingConfig::default(),
                llm_config: LlmConfig::default(),
                message_buffer_autoclear: false,
                in_context_message_indices: Vec::new(),
                messages: Vec::new(),
                metadata: None,
                multi_agent_group: None,
                name: "".to_string(),
                system: "".to_string(),
                tags: Vec::new(),
                tool_exec_environment_variables: Vec::new(),
                tool_rules: Vec::new(),
                tools: Vec::new(),
                updated_at: now,
                version: "1.0".to_string(),
            },
            embedding_config: None,
            llm_config: None,
            tools: Vec::new(),
            tool_rules: Vec::new(),
            tool_env_vars: Vec::new(),
        }
    }
}

impl AgentFileExporter {
    /// Create a new AgentFileExporter with default options
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the agent type (default: "memgpt_agent")
    pub fn agent_type(mut self, agent_type: impl Into<String>) -> Self {
        self.agent_file.agent_type = agent_type.into();
        self
    }

    /// Set the name of the agent
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.agent_file.name = name.into();
        self
    }

    /// Set the description of the agent
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.agent_file.description = Some(description.into());
        self
    }

    /// Add a metadata entry
    pub fn add_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        if self.agent_file.metadata.is_none() {
            self.agent_file.metadata = Some(HashMap::new());
        }
        if let Some(metadata) = &mut self.agent_file.metadata {
            metadata.insert(key.into(), value);
        }
        self
    }

    /// Set the embedding configuration
    pub fn embedding_config(mut self, config: EmbeddingConfig) -> Self {
        self.embedding_config = Some(config);
        self
    }

    /// Set the LLM configuration
    pub fn llm_config(mut self, config: LlmConfig) -> Self {
        self.llm_config = Some(config);
        self
    }

    /// Add a memory block to the agent
    pub fn add_memory_block(mut self, block: CoreMemoryBlock) -> Self {
        self.agent_file.core_memory.push(block);
        self
    }

    /// Add a tool to the agent
    pub fn add_tool(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }

    /// Add a tool rule to the agent
    pub fn add_tool_rule(mut self, rule: ToolRule) -> Self {
        self.tool_rules.push(rule);
        self
    }

    /// Add an environment variable for tool execution
    pub fn add_tool_env_var(mut self, env_var: ToolEnvVar) -> Self {
        self.tool_env_vars.push(env_var);
        self
    }

    /// Set the system prompt
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.agent_file.system = prompt.into();
        self
    }

    /// Add a tag to the agent
    pub fn add_tag(mut self, tag: Tag) -> Self {
        self.agent_file.tags.push(tag);
        self
    }

    /// Add a message to the conversation history
    pub fn add_message(mut self, message: Message) -> Self {
        self.agent_file.messages.push(message);
        self
    }

    /// Set the message buffer auto-clear flag
    pub fn message_buffer_autoclear(mut self, autoclear: bool) -> Self {
        self.agent_file.message_buffer_autoclear = autoclear;
        self
    }

    /// Set the in-context message indices
    pub fn in_context_message_indices(mut self, indices: Vec<usize>) -> Self {
        self.agent_file.in_context_message_indices = indices;
        self
    }

    /// Set the multi-agent group configuration
    pub fn multi_agent_group(mut self, config: Value) -> Self {
        self.agent_file.multi_agent_group = Some(config);
        self
    }

    /// Set the version
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.agent_file.version = version.into();
        self
    }

    /// Build the AgentFile
    pub fn build(mut self) -> AgentFile {
        // Update the timestamp
        self.agent_file.updated_at = Utc::now().to_rfc3339();

        // Set the embedding config if provided
        if let Some(embedding_config) = self.embedding_config {
            self.agent_file.embedding_config = embedding_config;
        }

        // Set the LLM config if provided
        if let Some(llm_config) = self.llm_config {
            self.agent_file.llm_config = llm_config;
        }

        // Add all tools
        self.agent_file.tools = self.tools;

        // Add all tool rules
        self.agent_file.tool_rules = self.tool_rules;

        // Add all tool environment variables
        self.agent_file.tool_exec_environment_variables = self.tool_env_vars;

        self.agent_file
    }

    /// Export the AgentFile as a JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        let agent_file = self.clone().build();
        serde_json::to_string_pretty(&agent_file)
    }

    /// Save the AgentFile to a file
    pub fn save_to_file(&self, path: impl AsRef<std::path::Path>) -> Result<(), Box<dyn std::error::Error>> {
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    #[test]
    fn test_export_basic() {
        // Create a basic agent file with required fields
        let exporter = AgentFileExporter::new()
            .agent_type("test_agent")
            .description("Test description")
            .name("Test Agent")
            .add_metadata("key", json!("value"));

        let json = exporter.to_json().unwrap();
        let parsed: Value = serde_json::from_str(&json).unwrap();

        // Verify required fields are present
        assert_eq!(parsed["agent_type"], "test_agent");
        assert_eq!(parsed["description"], "Test description");
        assert_eq!(parsed["name"], "Test Agent");
        assert!(parsed.get("metadata").is_some());
    }
}
