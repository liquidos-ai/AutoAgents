use crate::{
    config::AgentConfig,
    error::{Result, WorkflowError},
    tools::ToolRegistry,
    workflow::{
        llm_factory::LLMFactory,
        types::{WorkflowStream, WorkflowStreamEvent},
    },
};
use autoagents::{
    async_trait,
    core::{
        actor::Topic,
        agent::{
            memory::SlidingWindowMemory, prebuilt::executor::BasicAgent, task::Task, ActorAgent,
            AgentBuilder, AgentDeriveT, AgentHooks, Context,
        },
        environment::Environment,
        protocol::{Event, SubmissionId},
        runtime::{SingleThreadedRuntime, TypedRuntime},
        tool::{shared_tools_to_boxes, ToolT},
    },
    llm::chat::StreamChunk,
};
use futures::StreamExt;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::ReceiverStream;

#[derive(Debug, Clone)]
pub struct ChainAgent {
    name: String,
    description: String,
    tools: Vec<Arc<dyn ToolT>>,
    next_topic: Option<String>,
    result_tx: Option<Arc<tokio::sync::Mutex<Option<mpsc::Sender<String>>>>>,
}

impl ChainAgent {
    pub fn new(
        name: String,
        description: String,
        tools: Vec<Arc<dyn ToolT>>,
        next_topic: Option<String>,
        result_tx: Option<Arc<tokio::sync::Mutex<Option<mpsc::Sender<String>>>>>,
    ) -> Self {
        Self {
            name,
            description,
            tools,
            next_topic,
            result_tx,
        }
    }
}

impl AgentDeriveT for ChainAgent {
    type Output = String;

    fn description(&self) -> &'static str {
        Box::leak(self.description.clone().into_boxed_str())
    }

    fn name(&self) -> &'static str {
        Box::leak(self.name.clone().into_boxed_str())
    }

    fn output_schema(&self) -> Option<serde_json::Value> {
        None
    }

    fn tools(&self) -> Vec<Box<dyn ToolT>> {
        shared_tools_to_boxes(&self.tools)
    }
}

#[async_trait]
impl AgentHooks for ChainAgent {
    async fn on_run_complete(&self, _task: &Task, result: &Self::Output, ctx: &Context) {
        if let Some(next_topic) = &self.next_topic {
            // Forward to next agent in chain
            let _ = ctx
                .publish(Topic::<Task>::new(next_topic), Task::new(result))
                .await;
        } else if let Some(tx_mutex) = &self.result_tx {
            // Last agent - send final result
            if let Some(tx) = tx_mutex.lock().await.as_ref() {
                let _ = tx.send(result.clone()).await;
            }
        }
    }
}

pub struct SequentialWorkflow {
    agent_configs: Vec<AgentConfig>,
    stream: bool,
}

impl SequentialWorkflow {
    pub fn new(agent_configs: Vec<AgentConfig>, stream: bool) -> Self {
        Self {
            agent_configs,
            stream,
        }
    }

    pub fn stream_enabled(&self) -> bool {
        self.stream
    }

    pub async fn run(
        &self,
        input: String,
        _model_cache: Option<&crate::workflow::ModelCache>,
        _memory_cache: Option<
            &std::sync::Arc<
                tokio::sync::RwLock<
                    std::collections::HashMap<String, Vec<autoagents::llm::chat::ChatMessage>>,
                >,
            >,
        >,
        _workflow_name: Option<&str>,
        _memory_persistence: bool,
    ) -> Result<String> {
        // Generate unique suffix for this workflow run to avoid actor name collisions
        let run_id = uuid::Uuid::new_v4().to_string()[..8].to_string();

        let runtime = SingleThreadedRuntime::new(None);
        let mut environment = Environment::new(None);
        let _ = environment.register_runtime(runtime.clone()).await;

        // Channel for getting final result
        let (result_tx, mut result_rx) = mpsc::channel::<String>(1);
        let result_tx = Arc::new(tokio::sync::Mutex::new(Some(result_tx)));

        // Build chain of agents with unique names
        for (idx, agent_config) in self.agent_configs.iter().enumerate() {
            let llm = LLMFactory::create_llm(&agent_config.model).await?;
            let tools = ToolRegistry::create_tools(&agent_config.tools)?;

            let topic_name = format!("agent_{}_{}", idx, run_id);
            let next_topic = if idx < self.agent_configs.len() - 1 {
                Some(format!("agent_{}_{}", idx + 1, run_id))
            } else {
                None
            };

            let tx_for_last = if idx == self.agent_configs.len() - 1 {
                Some(result_tx.clone())
            } else {
                None
            };

            let agent_name = format!("{}_{}", agent_config.name, run_id);
            let chain_agent = ChainAgent::new(
                agent_name.clone(),
                agent_config
                    .instructions
                    .as_ref()
                    .unwrap_or(&agent_config.description)
                    .clone(),
                tools,
                next_topic,
                tx_for_last,
            );

            let agent = BasicAgent::new(chain_agent);
            let topic = Topic::<Task>::new(&topic_name);

            let window_size = agent_config
                .memory
                .as_ref()
                .map(|m| m.get_window_size())
                .unwrap_or(10);
            let memory = Box::new(SlidingWindowMemory::new(window_size));

            AgentBuilder::<_, ActorAgent>::new(agent)
                .llm(llm)
                .runtime(runtime.clone())
                .subscribe(topic.clone())
                .memory(memory)
                .stream(self.stream && idx == self.agent_configs.len() - 1)
                .build()
                .await?;

            // Publish initial task to first agent
            if idx == 0 {
                runtime.publish(&topic, Task::new(input.clone())).await?;
            }
        }

        // Run environment in background
        let env_handle = tokio::spawn(async move {
            let _ = environment.run().await;
        });

        // Wait for result with timeout
        let result = tokio::time::timeout(tokio::time::Duration::from_secs(300), result_rx.recv())
            .await
            .map_err(|_| {
                crate::error::WorkflowError::ExecutionError("Workflow timeout".to_string())
            })?
            .ok_or_else(|| {
                crate::error::WorkflowError::ExecutionError("No result received".to_string())
            })?;

        // Cleanup
        env_handle.abort();

        Ok(result)
    }

    pub async fn run_stream(
        &self,
        input: String,
        _model_cache: Option<&crate::workflow::ModelCache>,
        _memory_cache: Option<
            &std::sync::Arc<
                tokio::sync::RwLock<
                    std::collections::HashMap<String, Vec<autoagents::llm::chat::ChatMessage>>,
                >,
            >,
        >,
        _workflow_name: Option<&str>,
        _memory_persistence: bool,
    ) -> Result<WorkflowStream> {
        if !self.stream {
            return Err(WorkflowError::ExecutionError(
                "Streaming is disabled for this workflow".to_string(),
            ));
        }

        let run_id = uuid::Uuid::new_v4().to_string()[..8].to_string();

        let runtime = SingleThreadedRuntime::new(None);
        let mut environment = Environment::new(None);
        let _ = environment.register_runtime(runtime.clone()).await;

        let mut event_stream = environment
            .take_event_receiver(None)
            .await
            .map_err(|e| WorkflowError::RuntimeError(e.to_string()))?;

        let (result_tx, _result_rx) = mpsc::channel::<String>(1);
        let result_tx = Arc::new(tokio::sync::Mutex::new(Some(result_tx)));

        let mut final_agent_name = String::new();

        for (idx, agent_config) in self.agent_configs.iter().enumerate() {
            let llm = LLMFactory::create_llm(&agent_config.model).await?;
            let tools = ToolRegistry::create_tools(&agent_config.tools)?;

            let topic_name = format!("agent_{}_{}", idx, run_id);
            let next_topic = if idx < self.agent_configs.len() - 1 {
                Some(format!("agent_{}_{}", idx + 1, run_id))
            } else {
                None
            };

            let tx_for_last = if idx == self.agent_configs.len() - 1 {
                Some(result_tx.clone())
            } else {
                None
            };

            let agent_name = format!("{}_{}", agent_config.name, run_id);
            if idx == self.agent_configs.len() - 1 {
                final_agent_name = agent_name.clone();
            }

            let chain_agent = ChainAgent::new(
                agent_name.clone(),
                agent_config
                    .instructions
                    .as_ref()
                    .unwrap_or(&agent_config.description)
                    .clone(),
                tools,
                next_topic,
                tx_for_last,
            );

            let agent = BasicAgent::new(chain_agent);
            let topic = Topic::<Task>::new(&topic_name);

            let window_size = agent_config
                .memory
                .as_ref()
                .map(|m| m.get_window_size())
                .unwrap_or(10);
            let memory = Box::new(SlidingWindowMemory::new(window_size));

            AgentBuilder::<_, ActorAgent>::new(agent)
                .llm(llm)
                .runtime(runtime.clone())
                .subscribe(topic.clone())
                .memory(memory)
                .stream(self.stream && idx == self.agent_configs.len() - 1)
                .build()
                .await?;

            if idx == 0 {
                runtime.publish(&topic, Task::new(input.clone())).await?;
            }
        }

        let mut env_runner = environment;
        let env_task = tokio::spawn(async move {
            let _ = env_runner.run().await;
        });
        let env_handle = Arc::new(Mutex::new(Some(env_task)));

        let (tx, rx) = mpsc::channel::<Result<WorkflowStreamEvent>>(64);
        let tx_events = tx.clone();
        let env_handle_clone = env_handle.clone();
        let final_agent_name_clone = final_agent_name.clone();

        tokio::spawn(async move {
            let mut target_sub_id: Option<SubmissionId> = None;

            while let Some(event) = event_stream.next().await {
                match event {
                    Event::TaskStarted {
                        sub_id, actor_name, ..
                    } => {
                        if actor_name == final_agent_name_clone {
                            target_sub_id = Some(sub_id);
                        }
                    }
                    Event::StreamChunk { sub_id, chunk } => {
                        if target_sub_id == Some(sub_id) {
                            if let StreamChunk::Text(content) = chunk {
                                if !content.is_empty()
                                    && tx_events
                                        .send(Ok(WorkflowStreamEvent::Chunk { content }))
                                        .await
                                        .is_err()
                                {
                                    break;
                                }
                            }
                        }
                    }
                    Event::StreamToolCall { sub_id, tool_call } => {
                        if target_sub_id == Some(sub_id) {
                            let tool_name = tool_call
                                .get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or_default()
                                .to_string();
                            if tx_events
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
                    }
                    Event::StreamComplete { sub_id } => {
                        if target_sub_id == Some(sub_id) {
                            let _ = tx_events.send(Ok(WorkflowStreamEvent::Complete)).await;
                            break;
                        }
                    }
                    Event::TaskError { sub_id, error, .. } => {
                        if target_sub_id == Some(sub_id) {
                            let _ = tx_events
                                .send(Err(WorkflowError::ExecutionError(error)))
                                .await;
                            break;
                        }
                    }
                    _ => {}
                }
            }

            if let Some(handle) = env_handle_clone.lock().await.take() {
                handle.abort();
            }
        });

        drop(tx);

        Ok(Box::pin(ReceiverStream::new(rx)))
    }
}
