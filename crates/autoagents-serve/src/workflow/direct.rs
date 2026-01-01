use crate::{
    config::{AgentConfig, ExecutorKind, OutputType},
    error::{Result, WorkflowError},
    tools::ToolRegistry,
    workflow::{
        llm_factory::LLMFactory,
        types::{WorkflowStream, WorkflowStreamEvent},
    },
};
use autoagents::core::{
    agent::{
        memory::{MemoryProvider, SlidingWindowMemory},
        prebuilt::executor::{BasicAgent, ReActAgent},
        task::Task,
        AgentBuilder, AgentDeriveT, AgentExecutor, AgentHooks, DirectAgent, DirectAgentHandle,
    },
    protocol::Event,
    tool::{shared_tools_to_boxes, ToolT},
};
use autoagents::llm::chat::StreamChunk;
use futures::StreamExt;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

type MemoryCache =
    Arc<tokio::sync::RwLock<HashMap<String, Vec<autoagents::llm::chat::ChatMessage>>>>;

#[derive(Debug, Clone)]
pub struct DynamicAgent {
    name: String,
    description: String,
    tools: Vec<Arc<dyn ToolT>>,
    output_schema: Option<serde_json::Value>,
}

async fn persist_memory(
    agent_name: &str,
    memory_key: &Option<String>,
    memory_cache: &Option<MemoryCache>,
    agent_memory: Option<Arc<tokio::sync::Mutex<Box<dyn MemoryProvider>>>>,
) {
    if let (Some(key), Some(cache), Some(memory)) = (memory_key, memory_cache, agent_memory) {
        let exported_messages = memory.lock().await.export();
        let mut cache_write = cache.write().await;
        cache_write.insert(key.clone(), exported_messages.clone());
        log::debug!(
            "Exported {} messages to persistent cache for agent: {}",
            exported_messages.len(),
            agent_name
        );
    }
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

struct PreparedAgent {
    llm: Arc<dyn autoagents::llm::LLMProvider>,
    dynamic_agent: DynamicAgent,
    memory: Box<dyn MemoryProvider>,
    memory_key: Option<String>,
    memory_cache: Option<MemoryCache>,
}

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

    pub fn stream_enabled(&self) -> bool {
        self.stream
    }

    async fn build_direct_stream<T>(
        &self,
        agent_handle: DirectAgentHandle<T>,
        input: String,
        memory_key: Option<String>,
        memory_cache: Option<MemoryCache>,
    ) -> Result<WorkflowStream>
    where
        T: AgentDeriveT + AgentExecutor + AgentHooks + Send + Sync + 'static,
        <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output> + Into<String>,
    {
        if !self.stream {
            return Err(WorkflowError::ExecutionError(
                "Streaming is not enabled for this workflow".to_string(),
            ));
        }

        let mut event_stream = agent_handle.rx;
        let agent_memory = agent_handle.agent.memory();
        let agent_name = self.agent_config.name.clone();

        let (tx, rx) = mpsc::channel::<Result<WorkflowStreamEvent>>(64);

        let mut agent_stream = agent_handle.agent.run_stream(Task::new(input)).await?;
        let completion_flag = Arc::new(AtomicBool::new(false));
        let event_chunk_seen = Arc::new(AtomicBool::new(false));

        let tx_for_stream = tx.clone();
        let completion_for_stream = completion_flag.clone();
        let memory_cache_for_stream = memory_cache.clone();
        let memory_key_for_stream = memory_key.clone();
        let agent_memory_for_stream = agent_memory.clone();
        let agent_name_for_stream = agent_name.clone();
        let event_chunk_seen_stream = event_chunk_seen.clone();

        tokio::spawn(async move {
            while let Some(chunk) = agent_stream.next().await {
                match chunk {
                    Ok(content) => {
                        let text: String = content.into();

                        if event_chunk_seen_stream.load(Ordering::SeqCst) || text.is_empty() {
                            continue;
                        }

                        if tx_for_stream
                            .send(Ok(WorkflowStreamEvent::Chunk { content: text }))
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                    Err(err) => {
                        completion_for_stream.store(true, Ordering::SeqCst);
                        let _ = tx_for_stream
                            .send(Err(WorkflowError::ExecutionError(err.to_string())))
                            .await;
                        return;
                    }
                }
            }

            if completion_for_stream
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                persist_memory(
                    &agent_name_for_stream,
                    &memory_key_for_stream,
                    &memory_cache_for_stream,
                    agent_memory_for_stream,
                )
                .await;
                let _ = tx_for_stream.send(Ok(WorkflowStreamEvent::Complete)).await;
            }
        });

        let tx_for_events = tx.clone();
        let memory_cache_clone = memory_cache.clone();
        let memory_key_clone = memory_key.clone();
        let agent_memory_clone = agent_memory.clone();
        let agent_name_clone = agent_name.clone();
        let completion_for_events = completion_flag.clone();
        let event_chunk_seen_events = event_chunk_seen.clone();

        tokio::spawn(async move {
            while let Some(event) = event_stream.next().await {
                match event {
                    Event::StreamChunk { chunk, .. } => {
                        if let StreamChunk::Text(content) = chunk {
                            if !content.is_empty() {
                                if tx_for_events
                                    .send(Ok(WorkflowStreamEvent::Chunk { content }))
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                            }
                            event_chunk_seen_events.store(true, Ordering::SeqCst);
                        }
                    }
                    Event::StreamToolCall { tool_call, .. } => {
                        let tool_name = tool_call
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string();
                        if tx_for_events
                            .send(Ok(WorkflowStreamEvent::ToolCall {
                                tool_name,
                                payload: tool_call,
                            }))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    Event::StreamComplete { .. } => {
                        if completion_for_events
                            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                            .is_ok()
                        {
                            persist_memory(
                                &agent_name_clone,
                                &memory_key_clone,
                                &memory_cache_clone,
                                agent_memory_clone.clone(),
                            )
                            .await;
                            let _ = tx_for_events.send(Ok(WorkflowStreamEvent::Complete)).await;
                        }
                        break;
                    }
                    Event::ToolCallCompleted {
                        tool_name, result, ..
                    } => {
                        if tx_for_events
                            .send(Ok(WorkflowStreamEvent::ToolCallComplete {
                                tool_name,
                                result,
                            }))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    Event::TaskError { error, .. } => {
                        completion_for_events.store(true, Ordering::SeqCst);
                        let _ = tx_for_events
                            .send(Err(WorkflowError::ExecutionError(error)))
                            .await;
                        break;
                    }
                    _ => {}
                }
            }
        });

        drop(tx);

        Ok(Box::pin(ReceiverStream::new(rx)))
    }

    async fn prepare_agent(
        &self,
        model_cache: Option<&crate::workflow::ModelCache>,
        memory_cache: Option<&MemoryCache>,
        workflow_name: Option<&str>,
        memory_persistence: bool,
    ) -> Result<PreparedAgent> {
        let cached_model = if self.agent_config.model.preload {
            if let Some(cache) = model_cache {
                let key = {
                    #[cfg(feature = "http-serve")]
                    {
                        crate::utils::generate_model_key(&self.agent_config.model)
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

        let llm = LLMFactory::create_llm_with_cache(&self.agent_config.model, cached_model).await?;

        let tools = ToolRegistry::create_tools(&self.agent_config.tools)?;

        let output_schema = if let Some(output_config) = &self.agent_config.output {
            match output_config.output_type {
                OutputType::Structured | OutputType::Json => output_config.schema.clone(),
                OutputType::Text => None,
            }
        } else {
            None
        };

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

        let use_persistence = memory_persistence
            && self
                .agent_config
                .memory
                .as_ref()
                .and_then(|m| m.persistence.as_ref())
                .map(|p| p.enable)
                .unwrap_or(true);

        let window_size = self
            .agent_config
            .memory
            .as_ref()
            .map(|m| m.get_window_size())
            .unwrap_or(10);

        let mut memory: Box<dyn MemoryProvider> = Box::new(SlidingWindowMemory::new(window_size));

        let memory_key = if use_persistence {
            Some(format!(
                "{}_{}",
                workflow_name.unwrap_or("default"),
                self.agent_config.name
            ))
        } else {
            None
        };

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

        Ok(PreparedAgent {
            llm,
            dynamic_agent,
            memory,
            memory_key,
            memory_cache: memory_cache.cloned(),
        })
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
        let PreparedAgent {
            llm,
            dynamic_agent,
            memory,
            memory_key,
            memory_cache,
        } = self
            .prepare_agent(model_cache, memory_cache, workflow_name, memory_persistence)
            .await?;

        match self.agent_config.executor {
            ExecutorKind::ReAct => {
                let agent = ReActAgent::new(dynamic_agent);
                let agent_handle = AgentBuilder::<_, DirectAgent>::new(agent)
                    .llm(llm)
                    .memory(memory)
                    .stream(self.stream)
                    .build()
                    .await?;

                let output = if self.stream {
                    let mut stream = agent_handle
                        .agent
                        .run_stream(Task::new(input.clone()))
                        .await?;
                    let mut aggregated = String::new();

                    while let Some(chunk) = stream.next().await {
                        let chunk =
                            chunk.map_err(|e| WorkflowError::ExecutionError(e.to_string()))?;
                        aggregated.push_str(&chunk);
                    }

                    aggregated
                } else {
                    agent_handle.agent.run(Task::new(input)).await?
                };

                persist_memory(
                    &self.agent_config.name,
                    &memory_key,
                    &memory_cache,
                    agent_handle.agent.memory(),
                )
                .await;

                Ok(output)
            }
            ExecutorKind::Basic => {
                let agent = BasicAgent::new(dynamic_agent);
                let agent_handle = AgentBuilder::<_, DirectAgent>::new(agent)
                    .llm(llm)
                    .memory(memory)
                    .stream(self.stream)
                    .build()
                    .await?;

                let output = if self.stream {
                    let mut stream = agent_handle
                        .agent
                        .run_stream(Task::new(input.clone()))
                        .await?;
                    let mut aggregated = String::new();

                    while let Some(chunk) = stream.next().await {
                        let chunk =
                            chunk.map_err(|e| WorkflowError::ExecutionError(e.to_string()))?;
                        aggregated.push_str(&chunk);
                    }

                    aggregated
                } else {
                    agent_handle.agent.run(Task::new(input)).await?
                };

                persist_memory(
                    &self.agent_config.name,
                    &memory_key,
                    &memory_cache,
                    agent_handle.agent.memory(),
                )
                .await;

                Ok(output)
            }
        }
    }

    pub async fn run_stream(
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
    ) -> Result<WorkflowStream> {
        if !self.stream {
            return Err(WorkflowError::ExecutionError(
                "Streaming is disabled for this workflow".to_string(),
            ));
        }

        let PreparedAgent {
            llm,
            dynamic_agent,
            memory,
            memory_key,
            memory_cache,
        } = self
            .prepare_agent(model_cache, memory_cache, workflow_name, memory_persistence)
            .await?;

        let input_clone = input.clone();
        let memory_key_clone = memory_key.clone();
        let memory_cache_clone = memory_cache.clone();

        match self.agent_config.executor {
            ExecutorKind::ReAct => {
                let agent = ReActAgent::new(dynamic_agent);
                let agent_handle = AgentBuilder::<_, DirectAgent>::new(agent)
                    .llm(llm)
                    .memory(memory)
                    .stream(true)
                    .build()
                    .await?;

                self.build_direct_stream(
                    agent_handle,
                    input_clone,
                    memory_key_clone,
                    memory_cache_clone,
                )
                .await
            }
            ExecutorKind::Basic => {
                let agent = BasicAgent::new(dynamic_agent);
                let agent_handle = AgentBuilder::<_, DirectAgent>::new(agent)
                    .llm(llm)
                    .memory(memory)
                    .stream(true)
                    .build()
                    .await?;

                self.build_direct_stream(agent_handle, input, memory_key, memory_cache)
                    .await
            }
        }
    }
}
