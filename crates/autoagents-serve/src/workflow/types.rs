use crate::{
    config::{WorkflowConfig, WorkflowKind},
    error::Result,
    workflow::{
        direct::DirectWorkflow, parallel::ParallelWorkflow, routing::RoutingWorkflow,
        sequential::SequentialWorkflow,
    },
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum WorkflowOutput {
    Single(String),
    Multiple(Vec<String>),
}

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
                Ok(Workflow::Sequential(SequentialWorkflow::new(agents)))
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
                Ok(Workflow::Routing(RoutingWorkflow::new(router, handlers)))
            }
        }
    }

    pub async fn execute(&self, input: String) -> Result<WorkflowOutput> {
        match self {
            Workflow::Direct(w) => Ok(WorkflowOutput::Single(w.run(input).await?)),
            Workflow::Sequential(w) => Ok(WorkflowOutput::Single(w.run(input).await?)),
            Workflow::Parallel(w) => Ok(WorkflowOutput::Multiple(w.run(input).await?)),
            Workflow::Routing(w) => Ok(WorkflowOutput::Single(w.run(input).await?)),
        }
    }
}
