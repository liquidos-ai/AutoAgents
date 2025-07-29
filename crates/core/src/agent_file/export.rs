//! Provides a builder pattern for exporting AgentFile with flexible options.

use crate::agent_file::error::AgentFileError;
use crate::agent_file::schema::{AgentFile, CoreMemoryBlock, EmbeddingConfig, LlmConfig, Message, MessageContent, Tag, Tool, ToolEnvVar, ToolRule, ToolJsonSchema, Parameters, ParameterProperties};
use chrono::Utc;
use serde_json::Value;
use std::collections::HashMap;
use tempfile::TempDir;

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
                description: Some("".to_string()),
                embedding_config: EmbeddingConfig::default(),
                llm_config: LlmConfig::default(),
                message_buffer_autoclear: false,
                in_context_message_indices: vec![0, 1, 2, 3],
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
                version: "0.6.47".to_string(),
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

    /// Add a system message with text content
    pub fn add_system_message(mut self, text: impl Into<String>) -> Self {
        let now = Utc::now().to_rfc3339();
        let message = Message {
            created_at: now.clone(),
            group_id: None,
            model: Some("gpt-4o-mini".to_string()),
            name: None,
            role: "system".to_string(),
            content: vec![MessageContent::Text {
                content_type: "text".to_string(),
                text: text.into(),
            }],
            tool_call_id: None,
            tool_calls: vec![],
            tool_returns: vec![],
            updated_at: now,
        };
        self.agent_file.messages.push(message);
        self
    }

    /// Add a user message with text content
    pub fn add_user_message(mut self, text: impl Into<String>) -> Self {
        let now = Utc::now().to_rfc3339();
        let message = Message {
            created_at: now.clone(),
            group_id: None,
            model: None,
            name: None,
            role: "user".to_string(),
            content: vec![MessageContent::Text {
                content_type: "text".to_string(),
                text: text.into(),
            }],
            tool_call_id: None,
            tool_calls: vec![],
            tool_returns: vec![],
            updated_at: now,
        };
        self.agent_file.messages.push(message);
        self
    }

    /// Set the message buffer auto-clear flag
    pub fn message_buffer_autoclear(mut self, autoclear: bool) -> Self {
        self.agent_file.message_buffer_autoclear = autoclear;
        self
    }

    /// Set the in-context message indices
    pub fn in_context_message_indices(mut self, indices: Vec<i32>) -> Self {
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
    pub fn to_json(&self) -> Result<String, AgentFileError> {
        let agent_file = self.clone().build();
        let json_string = serde_json::to_string_pretty(&agent_file)?;
        Ok(json_string)
    }

    /// Save the AgentFile to a file
    pub fn save_to_file(&self, path: impl AsRef<std::path::Path>) -> Result<(), AgentFileError> {
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Create a temporary directory and save the AgentFile there
    pub fn save_to_temp_dir(&self, filename: &str) -> Result<TempDir, AgentFileError> {
        let temp_dir = tempfile::tempdir()?;
        let file_path = temp_dir.path().join(filename);
        self.save_to_file(file_path)?;
        Ok(temp_dir)
    }

    /// Create a simple tool with basic configuration
    pub fn create_simple_tool(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: HashMap<String, ParameterProperties>,
        required: Vec<String>,
        source_code: Option<String>,
        source_type: impl Into<String>,
        tool_type: impl Into<String>,
    ) -> Tool {
        let now = Utc::now().to_rfc3339();
        let name = name.into();
        let description = description.into();
        
        Tool {
            args_json_schema: None,
            created_at: now.clone(),
            description: description.clone(),
            json_schema: ToolJsonSchema {
                name: name.clone(),
                description,
                parameters: Parameters {
                    param_type: Some("object".to_string()),
                    properties: parameters,
                    required,
                },
                schema_type: None,
                required: None,
            },
            name,
            return_char_limit: 6000,
            source_code,
            source_type: source_type.into(),
            tags: vec![],
            tool_type: tool_type.into(),
            updated_at: now,
            metadata: None,
        }
    }

    /// Create a simple memory block
    pub fn create_memory_block(
        label: impl Into<String>,
        value: impl Into<String>,
        limit: i32,
    ) -> CoreMemoryBlock {
        let now = Utc::now().to_rfc3339();
        
        CoreMemoryBlock {
            created_at: now.clone(),
            description: None,
            is_template: false,
            label: label.into(),
            limit,
            metadata: Some(HashMap::new()),
            template_name: None,
            updated_at: now,
            value: value.into(),
        }
    }

    /// Create a simple tool rule
    pub fn create_tool_rule(rule_type: &str, tool_name: impl Into<String>) -> ToolRule {
        let tool_name = tool_name.into();
        
        match rule_type {
            "run_first" => ToolRule::Base { tool_name, rule_type: "run_first".to_string() },
            "exit_loop" => ToolRule::Base { tool_name, rule_type: "exit_loop".to_string() },
            "continue_loop" => ToolRule::Base { tool_name, rule_type: "continue_loop".to_string() },
            _ => ToolRule::Base { tool_name, rule_type: rule_type.to_string() },
        }
    }

    /// Create a conditional tool rule
    pub fn create_conditional_tool_rule(
        tool_name: impl Into<String>,
        default_child: Option<String>,
        child_output_mapping: HashMap<String, String>,
        require_output_mapping: bool,
    ) -> ToolRule {
        ToolRule::Conditional {
            tool_name: tool_name.into(),
            rule_type: "conditional".to_string(),
            default_child,
            child_output_mapping,
            require_output_mapping,
        }
    }

    /// Create a constrain child tools rule
    pub fn create_constrain_child_tools_rule(
        tool_name: impl Into<String>,
        children: Vec<String>,
    ) -> ToolRule {
        ToolRule::Child {
            tool_name: tool_name.into(),
            rule_type: "constrain_child_tools".to_string(),
            children,
        }
    }

    /// Create a max count per step rule
    pub fn create_max_count_per_step_rule(
        tool_name: impl Into<String>,
        max_count_limit: i32,
    ) -> ToolRule {
        ToolRule::MaxCountPerStep {
            tool_name: tool_name.into(),
            rule_type: "max_count_per_step".to_string(),
            max_count_limit,
        }
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
        assert!(parsed.get("metadata_").is_some());
    }

    #[test]
    fn test_create_simple_tool() {
        let mut parameters = HashMap::new();
        parameters.insert(
            "message".to_string(),
            ParameterProperties {
                param_type: "string".to_string(),
                description: Some("Message to send".to_string()),
            },
        );

        let tool = AgentFileExporter::create_simple_tool(
            "send_message",
            "Send a message to the user",
            parameters,
            vec!["message".to_string()],
            None,
            "letta_core",
            "letta_core",
        );

        assert_eq!(tool.name, "send_message");
        assert_eq!(tool.tool_type, "letta_core");
        assert_eq!(tool.source_type, "letta_core");
    }

    #[test]
    fn test_create_memory_block() {
        let block = AgentFileExporter::create_memory_block(
            "persona",
            "You are a helpful assistant",
            5000,
        );

        assert_eq!(block.label, "persona");
        assert_eq!(block.value, "You are a helpful assistant");
        assert_eq!(block.limit, 5000);
        assert!(!block.is_template);
    }

    #[test]
    fn test_save_to_temp_dir() {
        let exporter = AgentFileExporter::new()
            .name("temp_test_agent")
            .description("Test agent for temp dir");

        let temp_dir = exporter.save_to_temp_dir("test_agent.af").unwrap();
        let file_path = temp_dir.path().join("test_agent.af");
        
        assert!(file_path.exists());
        
        // Clean up
        temp_dir.close().unwrap();
    }
} 