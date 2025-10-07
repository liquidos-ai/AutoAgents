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

    pub async fn run(
        &self,
        input: String,
        model_cache: Option<&crate::workflow::ModelCache>,
        memory_cache: Option<
            &std::sync::Arc<
                tokio::sync::RwLock<
                    std::collections::HashMap<String, Vec<autoagents::llm::chat::ChatMessage>>,
                >,
            >,
        >,
        workflow_name: Option<&str>,
        memory_persistence: bool,
    ) -> Result<String> {
        // Get cached model or create new one
        let cached_model = if self.agent_config.model.preload {
            if let Some(cache) = model_cache {
                let key = {
                    #[cfg(feature = "http-serve")]
                    {
                        crate::server::generate_model_key(&self.agent_config.model)
                    }
                    #[cfg(not(feature = "http-serve"))]
                    String::new()
                };
                let cache_read = cache.read().await;
                cache_read.get(&key).cloned()
            } else {
                None
            }
        } else {
            None
        };

        // Create LLM provider (uses cache if available)
        let llm = LLMFactory::create_llm_with_cache(&self.agent_config.model, cached_model).await?;

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

        // Configure memory - check if persistence is enabled
        let use_persistence = memory_persistence
            && self
                .agent_config
                .memory
                .as_ref()
                .and_then(|m| m.persistence.as_ref())
                .map(|p| p.enable)
                .unwrap_or(true); // Default to enabled if workflow has persistence

        let window_size = self
            .agent_config
            .memory
            .as_ref()
            .map(|m| m.get_window_size())
            .unwrap_or(10);

        // Create memory and preload from cache if persistence is enabled
        let mut memory: Box<dyn autoagents::core::agent::memory::MemoryProvider> =
            Box::new(SlidingWindowMemory::new(window_size));

        let memory_key = if use_persistence {
            Some(format!(
                "{}_{}",
                workflow_name.unwrap_or("default"),
                self.agent_config.name
            ))
        } else {
            None
        };

        // Preload memory from cache if available
        if let (Some(key), Some(cache)) = (&memory_key, memory_cache) {
            let cache_read = cache.read().await;
            if let Some(cached_messages) = cache_read.get(key) {
                log::debug!(
                    "Preloading persistent memory for agent: {} ({} messages)",
                    self.agent_config.name,
                    cached_messages.len()
                );
                memory.preload(cached_messages.clone());
            } else {
                log::debug!(
                    "No cached memory found for agent: {}, starting fresh",
                    self.agent_config.name
                );
            }
        }

        // Create executor based on executor kind
        let result = match self.agent_config.executor {
            ExecutorKind::ReAct => {
                let agent = ReActAgent::new(dynamic_agent);
                let agent_handle = AgentBuilder::<_, DirectAgent>::new(agent)
                    .llm(llm)
                    .memory(memory)
                    .stream(self.stream)
                    .build()
                    .await?;

                let result = agent_handle.agent.run(Task::new(input)).await?;

                // Export memory back to cache if persistence is enabled
                if let (Some(key), Some(cache)) = (&memory_key, memory_cache) {
                    if let Some(mem) = &agent_handle.agent.memory() {
                        let exported_messages = mem.lock().await.export();
                        let mut cache_write = cache.write().await;
                        cache_write.insert(key.clone(), exported_messages.clone());
                        log::debug!(
                            "Exported {} messages to persistent cache for agent: {}",
                            exported_messages.len(),
                            self.agent_config.name
                        );
                    }
                }

                result
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

                // Export memory back to cache if persistence is enabled
                if let (Some(key), Some(cache)) = (&memory_key, memory_cache) {
                    if let Some(mem) = &agent_handle.agent.memory() {
                        let exported_messages = mem.lock().await.export();
                        let mut cache_write = cache.write().await;
                        cache_write.insert(key.clone(), exported_messages.clone());
                        log::debug!(
                            "Exported {} messages to persistent cache for agent: {}",
                            exported_messages.len(),
                            self.agent_config.name
                        );
                    }
                }

                result
            }
        };

        Ok(result)
    }
}
