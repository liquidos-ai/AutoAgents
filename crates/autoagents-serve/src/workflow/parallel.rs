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
pub struct ParallelAgent {
    name: String,
    description: String,
    tools: Vec<Arc<dyn ToolT>>,
    output_schema: Option<serde_json::Value>,
}

impl ParallelAgent {
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

impl AgentDeriveT for ParallelAgent {
    type Output = String;

    fn description(&self) -> &'static str {
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

impl AgentHooks for ParallelAgent {}

pub struct ParallelWorkflow {
    agent_configs: Vec<AgentConfig>,
}

impl ParallelWorkflow {
    pub fn new(agent_configs: Vec<AgentConfig>) -> Self {
        Self { agent_configs }
    }

    pub async fn run(&self, input: String) -> Result<Vec<String>> {
        let mut handles = vec![];

        for agent_config in &self.agent_configs {
            let llm = LLMFactory::create_llm(&agent_config.model).await?;
            let tools = ToolRegistry::create_tools(&agent_config.tools)?;

            // Determine output schema
            let output_schema = if let Some(output_config) = &agent_config.output {
                match output_config.output_type {
                    OutputType::Structured | OutputType::Json => output_config.schema.clone(),
                    OutputType::Text => None,
                }
            } else {
                None
            };

            let parallel_agent = ParallelAgent::new(
                agent_config.name.clone(),
                agent_config
                    .instructions
                    .as_ref()
                    .unwrap_or(&agent_config.description)
                    .clone(),
                tools,
                output_schema,
            );

            let window_size = agent_config
                .memory
                .as_ref()
                .map(|m| m.get_window_size())
                .unwrap_or(10);
            let memory = Box::new(SlidingWindowMemory::new(window_size));

            let task_input = input.clone();

            // Build agent based on executor type and spawn task
            let handle = match agent_config.executor {
                ExecutorKind::ReAct => {
                    let react_agent = ReActAgent::new(parallel_agent.clone());
                    let agent_handle = AgentBuilder::<_, DirectAgent>::new(react_agent)
                        .llm(llm)
                        .memory(memory)
                        .build()
                        .await?;

                    tokio::spawn(async move { agent_handle.agent.run(Task::new(task_input)).await })
                }
                ExecutorKind::Basic => {
                    let basic_agent = BasicAgent::new(parallel_agent.clone());
                    let agent_handle = AgentBuilder::<_, DirectAgent>::new(basic_agent)
                        .llm(llm)
                        .memory(memory)
                        .build()
                        .await?;

                    tokio::spawn(async move { agent_handle.agent.run(Task::new(task_input)).await })
                }
            };

            handles.push(handle);
        }

        // Wait for all agents to complete
        let mut results = vec![];
        for handle in handles {
            let result = handle
                .await
                .map_err(|e| crate::error::WorkflowError::ExecutionError(e.to_string()))??;
            results.push(result);
        }

        Ok(results)
    }
}
