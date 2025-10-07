use crate::{
    config::{AgentConfig, HandlerConfig},
    error::Result,
    tools::ToolRegistry,
    workflow::llm_factory::LLMFactory,
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
        runtime::{SingleThreadedRuntime, TypedRuntime},
        tool::{shared_tools_to_boxes, ToolT},
    },
};
use std::sync::Arc;
use tokio::sync::mpsc;

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
}

impl RoutingWorkflow {
    pub fn new(router_config: AgentConfig, handlers: Vec<HandlerConfig>) -> Self {
        Self {
            router_config,
            handlers,
        }
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
        environment.register_runtime(runtime.clone()).await;

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
                .build()
                .await?;
        }

        // Publish initial task to router
        runtime.publish(&router_topic, Task::new(input)).await?;

        // Run environment in background
        let env_handle = tokio::spawn(async move {
            environment.run().await;
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
}
