use crate::{
    config::{WorkflowConfig, WorkflowKind},
    error::{Result, WorkflowError},
    workflow::{
        direct::DirectWorkflow, parallel::ParallelWorkflow, routing::RoutingWorkflow,
        sequential::SequentialWorkflow,
    },
};
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum WorkflowOutput {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkflowStreamEvent {
    Chunk {
        content: String,
    },
    ToolCall {
        tool_name: String,
        payload: serde_json::Value,
    },
    ToolCallComplete {
        tool_name: String,
        result: serde_json::Value,
    },
    Complete,
}

pub type WorkflowStream =
    Pin<Box<dyn Stream<Item = crate::error::Result<WorkflowStreamEvent>> + Send>>;

pub enum Workflow {
    Direct(DirectWorkflow),
    Sequential(SequentialWorkflow),
    Parallel(ParallelWorkflow),
    Routing(RoutingWorkflow),
}

impl Workflow {
    pub fn from_config(config: WorkflowConfig) -> Result<Self> {
        let stream = config.stream;

        match config.kind {
            WorkflowKind::Direct => {
                let agent = config.workflow.agent.ok_or_else(|| {
                    crate::error::WorkflowError::MissingField("workflow.agent".to_string())
                })?;
                Ok(Workflow::Direct(DirectWorkflow::new(agent, stream)))
            }
            WorkflowKind::Sequential => {
                let agents = config.workflow.agents.ok_or_else(|| {
                    crate::error::WorkflowError::MissingField("workflow.agents".to_string())
                })?;
                Ok(Workflow::Sequential(SequentialWorkflow::new(
                    agents, stream,
                )))
            }
            WorkflowKind::Parallel => {
                let agents = config.workflow.agents.ok_or_else(|| {
                    crate::error::WorkflowError::MissingField("workflow.agents".to_string())
                })?;
                Ok(Workflow::Parallel(ParallelWorkflow::new(agents)))
            }
            WorkflowKind::Routing => {
                let router = config.workflow.router.ok_or_else(|| {
                    crate::error::WorkflowError::MissingField("workflow.router".to_string())
                })?;
                let handlers = config.workflow.handlers.ok_or_else(|| {
                    crate::error::WorkflowError::MissingField("workflow.handlers".to_string())
                })?;
                Ok(Workflow::Routing(RoutingWorkflow::new(
                    router, handlers, stream,
                )))
            }
        }
    }

    pub fn stream_enabled(&self) -> bool {
        match self {
            Workflow::Direct(w) => w.stream_enabled(),
            Workflow::Sequential(w) => w.stream_enabled(),
            Workflow::Parallel(_) => false,
            Workflow::Routing(w) => w.stream_enabled(),
        }
    }

    pub async fn execute(
        &self,
        input: String,
        model_cache: Option<
            &std::sync::Arc<
                tokio::sync::RwLock<
                    std::collections::HashMap<
                        String,
                        std::sync::Arc<dyn autoagents::llm::LLMProvider>,
                    >,
                >,
            >,
        >,
        memory_cache: Option<
            &std::sync::Arc<
                tokio::sync::RwLock<
                    std::collections::HashMap<String, Vec<autoagents::llm::chat::ChatMessage>>,
                >,
            >,
        >,
        workflow_name: Option<&str>,
        memory_persistence: bool,
    ) -> Result<WorkflowOutput> {
        match self {
            Workflow::Direct(w) => Ok(WorkflowOutput::Single(
                w.run(
                    input,
                    model_cache,
                    memory_cache,
                    workflow_name,
                    memory_persistence,
                )
                .await?,
            )),
            Workflow::Sequential(w) => Ok(WorkflowOutput::Single(
                w.run(
                    input,
                    model_cache,
                    memory_cache,
                    workflow_name,
                    memory_persistence,
                )
                .await?,
            )),
            Workflow::Parallel(w) => Ok(WorkflowOutput::Multiple(
                w.run(
                    input,
                    model_cache,
                    memory_cache,
                    workflow_name,
                    memory_persistence,
                )
                .await?,
            )),
            Workflow::Routing(w) => Ok(WorkflowOutput::Single(
                w.run(
                    input,
                    model_cache,
                    memory_cache,
                    workflow_name,
                    memory_persistence,
                )
                .await?,
            )),
        }
    }

    pub async fn execute_stream(
        &self,
        input: String,
        model_cache: Option<
            &std::sync::Arc<
                tokio::sync::RwLock<
                    std::collections::HashMap<
                        String,
                        std::sync::Arc<dyn autoagents::llm::LLMProvider>,
                    >,
                >,
            >,
        >,
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
        match self {
            Workflow::Direct(w) => {
                w.run_stream(
                    input,
                    model_cache,
                    memory_cache,
                    workflow_name,
                    memory_persistence,
                )
                .await
            }
            Workflow::Sequential(w) => {
                w.run_stream(
                    input,
                    model_cache,
                    memory_cache,
                    workflow_name,
                    memory_persistence,
                )
                .await
            }
            Workflow::Parallel(_) => Err(WorkflowError::ExecutionError(
                "Streaming is not supported for parallel workflows".to_string(),
            )),
            Workflow::Routing(w) => {
                w.run_stream(
                    input,
                    model_cache,
                    memory_cache,
                    workflow_name,
                    memory_persistence,
                )
                .await
            }
        }
    }
}
