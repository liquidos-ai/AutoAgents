use crate::{
    config::{AgentConfig, HandlerConfig},
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
pub struct RouterAgent {
    name: String,
    description: String,
    tools: Vec<Arc<dyn ToolT>>,
}

impl RouterAgent {
    pub fn new(name: String, description: String, tools: Vec<Arc<dyn ToolT>>) -> Self {
        Self {
            name,
            description,
            tools,
        }
    }
}

impl AgentDeriveT for RouterAgent {
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
impl AgentHooks for RouterAgent {
    async fn on_run_complete(&self, _task: &Task, result: &Self::Output, ctx: &Context) {
        // Router publishes its decision to a topic named after the condition
        let condition = result.trim().to_lowercase();
        let topic = Topic::<Task>::new(&format!("handler_{}", condition));

        // Forward the original task to the appropriate handler
        let _ = ctx.publish(topic, _task.clone()).await;
    }
}

#[derive(Debug, Clone)]
pub struct HandlerAgent {
    name: String,
    description: String,
    tools: Vec<Arc<dyn ToolT>>,
    result_tx: Arc<tokio::sync::Mutex<Option<mpsc::Sender<String>>>>,
}

impl HandlerAgent {
    pub fn new(
        name: String,
        description: String,
        tools: Vec<Arc<dyn ToolT>>,
        result_tx: Arc<tokio::sync::Mutex<Option<mpsc::Sender<String>>>>,
    ) -> Self {
        Self {
            name,
            description,
            tools,
            result_tx,
        }
    }
}

impl AgentDeriveT for HandlerAgent {
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
impl AgentHooks for HandlerAgent {
    async fn on_run_complete(&self, _task: &Task, result: &Self::Output, _ctx: &Context) {
        // Send final result back
        if let Some(tx) = self.result_tx.lock().await.as_ref() {
            let _ = tx.send(result.clone()).await;
        }
    }
}

pub struct RoutingWorkflow {
    router_config: AgentConfig,
    handlers: Vec<HandlerConfig>,
    stream: bool,
}

impl RoutingWorkflow {
    pub fn new(router_config: AgentConfig, handlers: Vec<HandlerConfig>, stream: bool) -> Self {
        Self {
            router_config,
            handlers,
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

        // Build router agent with unique name
        let router_llm = LLMFactory::create_llm(&self.router_config.model).await?;
        let router_tools = ToolRegistry::create_tools(&self.router_config.tools)?;

        let router_name = format!("{}_{}", self.router_config.name, run_id);
        let router_agent = RouterAgent::new(
            router_name.clone(),
            self.router_config
                .instructions
                .as_ref()
                .unwrap_or(&self.router_config.description)
                .clone(),
            router_tools,
        );

        let router = BasicAgent::new(router_agent);
        let router_topic = Topic::<Task>::new("router");

        let router_window_size = self
            .router_config
            .memory
            .as_ref()
            .map(|m| m.get_window_size())
            .unwrap_or(10);
        let router_memory = Box::new(SlidingWindowMemory::new(router_window_size));

        AgentBuilder::<_, ActorAgent>::new(router)
            .llm(router_llm)
            .runtime(runtime.clone())
            .subscribe(router_topic.clone())
            .memory(router_memory)
            .build()
            .await?;

        // Build handler agents with unique names
        for handler in &self.handlers {
            let handler_llm = LLMFactory::create_llm(&handler.agent.model).await?;
            let handler_tools = ToolRegistry::create_tools(&handler.agent.tools)?;

            let handler_name = format!("{}_{}", handler.agent.name, run_id);
            let handler_agent = HandlerAgent::new(
                handler_name.clone(),
                handler
                    .agent
                    .instructions
                    .as_ref()
                    .unwrap_or(&handler.agent.description)
                    .clone(),
                handler_tools,
                result_tx.clone(),
            );

            let agent = BasicAgent::new(handler_agent);
            let handler_topic = Topic::<Task>::new(&format!("handler_{}", handler.condition));

            let handler_window_size = handler
                .agent
                .memory
                .as_ref()
                .map(|m| m.get_window_size())
                .unwrap_or(10);
            let handler_memory = Box::new(SlidingWindowMemory::new(handler_window_size));

            AgentBuilder::<_, ActorAgent>::new(agent)
                .llm(handler_llm)
                .runtime(runtime.clone())
                .subscribe(handler_topic)
                .memory(handler_memory)
                .stream(self.stream)
                .build()
                .await?;
        }

        // Publish initial task to router
        runtime.publish(&router_topic, Task::new(input)).await?;

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

        let router_llm = LLMFactory::create_llm(&self.router_config.model).await?;
        let router_tools = ToolRegistry::create_tools(&self.router_config.tools)?;

        let router_name = format!("{}_{}", self.router_config.name, run_id);
        let router_agent = RouterAgent::new(
            router_name.clone(),
            self.router_config
                .instructions
                .as_ref()
                .unwrap_or(&self.router_config.description)
                .clone(),
            router_tools,
        );

        let router = BasicAgent::new(router_agent);
        let router_topic = Topic::<Task>::new("router");

        let router_window_size = self
            .router_config
            .memory
            .as_ref()
            .map(|m| m.get_window_size())
            .unwrap_or(10);
        let router_memory = Box::new(SlidingWindowMemory::new(router_window_size));

        AgentBuilder::<_, ActorAgent>::new(router)
            .llm(router_llm)
            .runtime(runtime.clone())
            .subscribe(router_topic.clone())
            .memory(router_memory)
            .build()
            .await?;

        let mut handler_names = Vec::new();

        for handler in &self.handlers {
            let handler_llm = LLMFactory::create_llm(&handler.agent.model).await?;
            let handler_tools = ToolRegistry::create_tools(&handler.agent.tools)?;

            let handler_name = format!("{}_{}", handler.agent.name, run_id);
            handler_names.push(handler_name.clone());

            let handler_agent = HandlerAgent::new(
                handler_name.clone(),
                handler
                    .agent
                    .instructions
                    .as_ref()
                    .unwrap_or(&handler.agent.description)
                    .clone(),
                handler_tools,
                result_tx.clone(),
            );

            let agent = BasicAgent::new(handler_agent);
            let handler_topic = Topic::<Task>::new(&format!("handler_{}", handler.condition));

            let handler_window_size = handler
                .agent
                .memory
                .as_ref()
                .map(|m| m.get_window_size())
                .unwrap_or(10);
            let handler_memory = Box::new(SlidingWindowMemory::new(handler_window_size));

            AgentBuilder::<_, ActorAgent>::new(agent)
                .llm(handler_llm)
                .runtime(runtime.clone())
                .subscribe(handler_topic)
                .memory(handler_memory)
                .stream(true)
                .build()
                .await?;
        }

        runtime
            .publish(&router_topic, Task::new(input.clone()))
            .await?;

        let mut env_runner = environment;
        let env_task = tokio::spawn(async move {
            let _ = env_runner.run().await;
        });
        let env_handle = Arc::new(Mutex::new(Some(env_task)));

        let (tx, rx) = mpsc::channel::<Result<WorkflowStreamEvent>>(64);
        let tx_events = tx.clone();
        let env_handle_clone = env_handle.clone();
        let handler_names_clone = handler_names.clone();

        tokio::spawn(async move {
            let mut target_sub_id: Option<SubmissionId> = None;

            while let Some(event) = event_stream.next().await {
                match event {
                    Event::TaskStarted {
                        sub_id, actor_name, ..
                    } => {
                        if handler_names_clone.contains(&actor_name) {
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
