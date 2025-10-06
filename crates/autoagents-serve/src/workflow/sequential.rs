use crate::{
    config::AgentConfig, error::Result, tools::ToolRegistry, workflow::llm_factory::LLMFactory,
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
}

impl SequentialWorkflow {
    pub fn new(agent_configs: Vec<AgentConfig>) -> Self {
        Self { agent_configs }
    }

    pub async fn run(&self, input: String) -> Result<String> {
        // Generate unique suffix for this workflow run to avoid actor name collisions
        let run_id = uuid::Uuid::new_v4().to_string()[..8].to_string();

        let runtime = SingleThreadedRuntime::new(None);
        let mut environment = Environment::new(None);
        environment.register_runtime(runtime.clone()).await;

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
                .build()
                .await?;

            // Publish initial task to first agent
            if idx == 0 {
                runtime.publish(&topic, Task::new(input.clone())).await?;
            }
        }

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
