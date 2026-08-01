use crate::agent::config::AgentConfig;
use crate::agent::executor::event_helper::EventHelper;
use crate::agent::memory::MemoryProvider;
use crate::agent::skill::{
    SkillConfiguration, SkillLifecycle, SkillLifecycleDecision, SkillRefreshReport, SkillRuntime,
    SkillRuntimeIdentity, SkillSession, SkillSnapshot,
};
use crate::agent::task::Task;
use crate::agent::{AgentExecutor, Context, output::AgentOutputT};
use crate::tool::{ToolT, to_llm_tool};
use async_trait::async_trait;
use autoagents_llm::LLMProvider;
use autoagents_llm::chat::Tool;
use autoagents_protocol::{ActorID, Event, SkillEvent, SubmissionId};

use serde_json::Value;
use std::marker::PhantomData;
use std::{fmt::Debug, sync::Arc};

#[cfg(target_arch = "wasm32")]
pub use futures::lock::Mutex;
#[cfg(not(target_arch = "wasm32"))]
pub use tokio::sync::Mutex;

#[cfg(target_arch = "wasm32")]
use futures::channel::mpsc::Sender;

#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::mpsc::Sender;

use crate::agent::error::RunnableAgentError;
use crate::agent::hooks::AgentHooks;
use uuid::Uuid;

/// Core trait that defines agent metadata and behavior
/// This trait is implemented via the #[agent] macro
#[async_trait]
pub trait AgentDeriveT: Send + Sync + 'static + Debug {
    /// The output type this agent produces
    type Output: AgentOutputT;

    /// Get the agent's description
    fn description(&self) -> &str;

    // If you provide None then its taken as String output
    fn output_schema(&self) -> Option<Value>;

    /// Get the agent's name
    fn name(&self) -> &str;

    /// Get the tools available to this agent
    fn tools(&self) -> Vec<Box<dyn ToolT>>;
}

pub trait AgentType: 'static + Send + Sync {
    fn type_name() -> &'static str;
}

/// Base agent type that wraps an AgentDeriveT implementation with additional runtime components
#[derive(Clone)]
pub struct BaseAgent<T: AgentDeriveT + AgentExecutor + AgentHooks + Send + Sync, A: AgentType> {
    /// The inner agent implementation (from macro)
    pub(crate) inner: Arc<T>,
    /// LLM provider for this agent
    pub(crate) llm: Arc<dyn LLMProvider>,
    /// Agent ID
    pub id: ActorID,
    /// Optional memory provider
    pub(crate) memory: Option<Arc<Mutex<Box<dyn MemoryProvider>>>>,
    pub(crate) skills: Option<SkillConfiguration>,
    /// Cached serialized tool definitions
    pub(crate) serialized_tools: Option<Arc<Vec<Tool>>>,
    /// Tx sender
    pub(crate) tx: Option<Sender<Event>>,
    //Stream
    pub(crate) stream: bool,
    pub(crate) marker: PhantomData<A>,
}

impl<T: AgentDeriveT + AgentExecutor + AgentHooks, A: AgentType> Debug for BaseAgent<T, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("A: {} - T: {}", self.inner().name(), A::type_name()).as_str())
    }
}

impl<T: AgentDeriveT + AgentExecutor + AgentHooks, A: AgentType> BaseAgent<T, A> {
    /// Create a new BaseAgent wrapping an AgentDeriveT implementation
    pub async fn new(
        inner: T,
        llm: Arc<dyn LLMProvider>,
        memory: Option<Box<dyn MemoryProvider>>,
        skills: Option<SkillConfiguration>,
        tx: Sender<Event>,
        stream: bool,
    ) -> Result<Self, RunnableAgentError> {
        let tool_defs = inner.tools();
        if skills.is_some() && !inner.supports_agent_skills() {
            return Err(RunnableAgentError::InitializationError(format!(
                "executor '{}' does not support Agent Skills",
                std::any::type_name::<T>()
            )));
        }
        if skills.is_some() {
            SkillRuntime::validate_agent_tools(&tool_defs)
                .map_err(|error| RunnableAgentError::InitializationError(error.to_string()))?;
        }
        let serialized_tools = if tool_defs.is_empty() {
            None
        } else {
            Some(Arc::new(
                tool_defs.iter().map(to_llm_tool).collect::<Vec<_>>(),
            ))
        };
        let agent = Self {
            inner: Arc::new(inner),
            id: Uuid::new_v4(),
            llm,
            tx: Some(tx),
            memory: memory.map(|m| Arc::new(Mutex::new(m))),
            skills,
            serialized_tools,
            stream,
            marker: PhantomData,
        };

        //Run Hook
        agent.inner().on_agent_create().await;
        if let Some(runtime) = agent.skill_runtime(None) {
            runtime.initialize().await;
        }

        Ok(agent)
    }

    pub fn inner(&self) -> Arc<T> {
        self.inner.clone()
    }

    /// Get the agent's name
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    /// Get the agent's description
    pub fn description(&self) -> &str {
        self.inner.description()
    }

    /// Get the tools as Arc-wrapped references
    pub fn tools(&self) -> Vec<Box<dyn ToolT>> {
        self.inner.tools()
    }

    pub fn serialized_tools(&self) -> Option<Arc<Vec<Tool>>> {
        self.serialized_tools.clone()
    }

    pub fn stream(&self) -> bool {
        self.stream
    }

    pub(crate) fn create_context(&self, task: &Task) -> Arc<Context> {
        let tools = self.tools();
        let cached_tools = self
            .serialized_tools()
            .filter(|cached| Self::tools_match_cached(&tools, cached));
        let runtime = self.skill_runtime(Some(task.submission_id));
        Arc::new(
            Context::new(self.llm(), self.tx.clone())
                .with_memory(self.memory())
                .with_serialized_tools(cached_tools)
                .with_tools(tools)
                .with_config(self.agent_config())
                .with_skills(runtime)
                .with_stream(self.stream()),
        )
    }

    pub fn skill_session(&self) -> Option<Arc<SkillSession>> {
        self.skills.as_ref().map(SkillConfiguration::session)
    }

    pub fn skill_snapshot(&self) -> Option<Arc<SkillSnapshot>> {
        self.skills
            .as_ref()
            .map(SkillConfiguration::registry)
            .map(|registry| registry.snapshot())
    }

    pub async fn refresh_skills(&self) -> Option<SkillRefreshReport> {
        let runtime = self.skill_runtime(None)?;
        Some(runtime.refresh_now().await)
    }

    pub async fn reset_skill_session(&self) {
        if let Some(runtime) = self.skill_runtime(None) {
            runtime.reset_session().await;
        }
    }

    fn skill_runtime(&self, submission_id: Option<SubmissionId>) -> Option<Arc<SkillRuntime>> {
        let configuration = self.skills.clone()?;
        let lifecycle: Arc<dyn SkillLifecycle> = Arc::new(AgentSkillLifecycle {
            inner: self.inner(),
        });
        let identity = SkillRuntimeIdentity::new(self.id, submission_id, self.tx.clone());
        Some(SkillRuntime::new(configuration, identity, lifecycle))
    }

    fn tools_match_cached(tools: &[Box<dyn ToolT>], cached: &[Tool]) -> bool {
        if tools.len() != cached.len() {
            return false;
        }

        tools.iter().zip(cached.iter()).all(|(tool, cached_tool)| {
            cached_tool.tool_type == "function"
                && cached_tool.function.name == tool.name()
                && cached_tool.function.description == tool.description()
                && cached_tool.function.parameters == tool.args_schema()
        })
    }

    pub fn agent_config(&self) -> AgentConfig {
        let output_schema = self.inner().output_schema();
        let structured_schema =
            output_schema.and_then(|schema| serde_json::from_value(schema).ok());
        AgentConfig {
            name: self.name().into(),
            description: self.description().into(),
            id: self.id,
            output_schema: structured_schema,
        }
    }

    /// Get the LLM provider
    pub fn llm(&self) -> Arc<dyn LLMProvider> {
        self.llm.clone()
    }

    /// Get the memory provider if available
    pub fn memory(&self) -> Option<Arc<Mutex<Box<dyn MemoryProvider>>>> {
        self.memory.clone()
    }

    /// Clone handle-style fields without requiring `T: Clone`.
    pub(crate) fn clone_shallow(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            llm: self.llm.clone(),
            id: self.id,
            memory: self.memory.clone(),
            skills: self.skills.clone(),
            serialized_tools: self.serialized_tools.clone(),
            tx: self.tx.clone(),
            stream: self.stream,
            marker: PhantomData,
        }
    }

    /// Emit `TaskComplete`, run `on_run_complete`, and return the agent output.
    ///
    /// Uses `Value: From<AgentExecutor::Output>` so `TaskComplete` serialization matches
    /// the executor payload regardless of `AgentDeriveT::Output`.
    pub(crate) async fn finish_executor_run(
        &self,
        task: &Task,
        context: &Context,
        submission_id: SubmissionId,
        executor_out: <T as AgentExecutor>::Output,
    ) -> Result<<T as AgentDeriveT>::Output, RunnableAgentError>
    where
        Value: From<<T as AgentExecutor>::Output>,
        <T as AgentDeriveT>::Output: From<<T as AgentExecutor>::Output>,
        <T as AgentExecutor>::Output: Clone,
    {
        let tx_event = self.tx.clone();
        let value: Value = executor_out.clone().into();
        #[cfg(not(target_arch = "wasm32"))]
        if let Err(e) = EventHelper::send_task_completed_value(
            &tx_event,
            submission_id,
            self.id,
            self.name().to_string(),
            &value,
        )
        .await
        {
            let err = RunnableAgentError::ExecutorError(e.to_string());
            EventHelper::send_task_error(&tx_event, submission_id, self.id, err.to_string()).await;
            return Err(err);
        }

        let agent_out: <T as AgentDeriveT>::Output = executor_out.into();
        self.inner.on_run_complete(task, &agent_out, context).await;
        Ok(agent_out)
    }
}

#[derive(Debug)]
struct AgentSkillLifecycle<T: AgentDeriveT + AgentExecutor + AgentHooks> {
    inner: Arc<T>,
}

#[async_trait]
impl<T: AgentDeriveT + AgentExecutor + AgentHooks> SkillLifecycle for AgentSkillLifecycle<T> {
    async fn on_catalog_changed(&self, event: &SkillEvent) {
        self.inner.on_skill_catalog_changed(event).await;
    }

    async fn on_activation_requested(&self, event: &SkillEvent) -> SkillLifecycleDecision {
        match self.inner.on_skill_activation(event).await {
            crate::agent::HookOutcome::Continue => SkillLifecycleDecision::Continue,
            crate::agent::HookOutcome::Abort => SkillLifecycleDecision::Abort,
        }
    }

    async fn on_activated(&self, event: &SkillEvent) {
        self.inner.on_skill_activated(event).await;
    }

    async fn on_deactivated(&self, event: &SkillEvent) {
        self.inner.on_skill_deactivated(event).await;
    }

    async fn on_resource_access_requested(&self, event: &SkillEvent) -> SkillLifecycleDecision {
        match self.inner.on_skill_resource_access(event).await {
            crate::agent::HookOutcome::Continue => SkillLifecycleDecision::Continue,
            crate::agent::HookOutcome::Abort => SkillLifecycleDecision::Abort,
        }
    }

    async fn on_resource_accessed(&self, event: &SkillEvent) {
        self.inner.on_skill_resource_result(event).await;
    }

    async fn on_operation_failed(&self, event: &SkillEvent) {
        self.inner.on_skill_error(event).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::memory::SlidingWindowMemory;
    use crate::agent::{AgentConfig, DirectAgent};
    use crate::tests::{MockAgentImpl, MockLLMProvider};
    use autoagents_llm::chat::StructuredOutputFormat;
    use std::sync::Arc;
    use tokio::sync::mpsc::{Receiver, channel};
    use uuid::Uuid;

    #[test]
    fn test_agent_config_with_schema() {
        let schema = StructuredOutputFormat {
            name: "TestSchema".to_string(),
            description: Some("Test schema".to_string()),
            schema: Some(serde_json::json!({"type": "object"})),
            strict: Some(true),
        };

        let config = AgentConfig {
            name: "test_agent".to_string(),
            id: Uuid::new_v4(),
            description: "A test agent".to_string(),
            output_schema: Some(schema.clone()),
        };

        assert_eq!(config.name, "test_agent");
        assert_eq!(config.description, "A test agent");
        assert!(config.output_schema.is_some());
        assert_eq!(config.output_schema.unwrap().name, "TestSchema");
    }

    #[tokio::test]
    async fn test_base_agent_creation_with_memory_and_stream() {
        let mock_agent = MockAgentImpl::new("test", "test description");
        let llm = Arc::new(MockLLMProvider);
        let memory = Box::new(SlidingWindowMemory::new(5));
        let (tx, _): (Sender<Event>, Receiver<Event>) = channel(32);
        let base_agent =
            BaseAgent::<_, DirectAgent>::new(mock_agent, llm, Some(memory), None, tx, true)
                .await
                .unwrap();

        assert_eq!(base_agent.name(), "test");
        assert_eq!(base_agent.description(), "test description");
        assert!(base_agent.memory().is_some());
        assert!(base_agent.stream);
    }

    #[tokio::test]
    async fn test_base_agent_create_context_populates_config() {
        let mock_agent = MockAgentImpl::new("ctx_agent", "context agent");
        let llm = Arc::new(MockLLMProvider);
        let (tx, _): (Sender<Event>, Receiver<Event>) = channel(32);
        let base_agent = BaseAgent::<_, DirectAgent>::new(mock_agent, llm, None, None, tx, false)
            .await
            .unwrap();

        let context = base_agent.create_context(&Task::new("test"));
        let config = context.config();
        assert_eq!(config.name, "ctx_agent");
        assert_eq!(config.description, "context agent");
    }
}
