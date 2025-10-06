use crate::{
    config::{AgentConfig, ExecutorKind, OutputType},
    error::Result,
    tools::ToolRegistry,
    workflow::llm_factory::LLMFactory,
};
use autoagents::core::{
    agent::{
        memory::SlidingWindowMemory,
        prebuilt::executor::{BasicAgent, ReActAgent},
        task::Task,
        AgentBuilder, AgentDeriveT, AgentHooks, DirectAgent,
    },
    tool::{shared_tools_to_boxes, ToolT},
};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct DynamicAgent {
    name: String,
    description: String,
    tools: Vec<Arc<dyn ToolT>>,
    output_schema: Option<serde_json::Value>,
}

impl DynamicAgent {
    pub fn new(
        name: String,
        description: String,
        tools: Vec<Arc<dyn ToolT>>,
        output_schema: Option<serde_json::Value>,
    ) -> Self {
        Self {
            name,
            description,
            tools,
            output_schema,
        }
    }
}

impl AgentDeriveT for DynamicAgent {
    type Output = String;

    fn description(&self) -> &'static str {
        // We need to return a static str, so we'll use Box::leak to create a static string
        // This is acceptable for workflow agents that are long-lived
        Box::leak(self.description.clone().into_boxed_str())
    }

    fn name(&self) -> &'static str {
        Box::leak(self.name.clone().into_boxed_str())
    }

    fn output_schema(&self) -> Option<serde_json::Value> {
        self.output_schema.clone()
    }

    fn tools(&self) -> Vec<Box<dyn ToolT>> {
        shared_tools_to_boxes(&self.tools)
    }
}

impl AgentHooks for DynamicAgent {}

pub struct DirectWorkflow {
    agent_config: AgentConfig,
    stream: bool,
}

impl DirectWorkflow {
    pub fn new(agent_config: AgentConfig, stream: bool) -> Self {
        Self {
            agent_config,
            stream,
        }
    }

    pub async fn run(&self, input: String) -> Result<String> {
        // Create LLM provider
        let llm = LLMFactory::create_llm(&self.agent_config.model).await?;

        // Create tools
        let tools = ToolRegistry::create_tools(&self.agent_config.tools)?;

        // Determine output schema based on output configuration
        let output_schema = if let Some(output_config) = &self.agent_config.output {
            match output_config.output_type {
                OutputType::Structured => output_config.schema.clone(),
                OutputType::Json => output_config.schema.clone(),
                OutputType::Text => None,
            }
        } else {
            None
        };

        // Create dynamic agent
        let dynamic_agent = DynamicAgent::new(
            self.agent_config.name.clone(),
            self.agent_config
                .instructions
                .as_ref()
                .unwrap_or(&self.agent_config.description)
                .clone(),
            tools,
            output_schema,
        );

        // Configure memory
        let window_size = self
            .agent_config
            .memory
            .as_ref()
            .map(|m| m.get_window_size())
            .unwrap_or(10);
        let memory = Box::new(SlidingWindowMemory::new(window_size));

        // Create executor based on executor kind
        match self.agent_config.executor {
            ExecutorKind::ReAct => {
                let agent = ReActAgent::new(dynamic_agent);
                let agent_handle = AgentBuilder::<_, DirectAgent>::new(agent)
                    .llm(llm)
                    .memory(memory)
                    .stream(self.stream)
                    .build()
                    .await?;

                let result = agent_handle.agent.run(Task::new(input)).await?;
                Ok(result)
            }
            ExecutorKind::Basic => {
                let agent = BasicAgent::new(dynamic_agent);
                let agent_handle = AgentBuilder::<_, DirectAgent>::new(agent)
                    .llm(llm)
                    .memory(memory)
                    .stream(self.stream)
                    .build()
                    .await?;

                let result = agent_handle.agent.run(Task::new(input)).await?;
                Ok(result)
            }
        }
    }
}
