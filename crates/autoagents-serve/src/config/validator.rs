use super::schema::{WorkflowConfig, WorkflowKind};
use crate::error::{Result, WorkflowError};

pub fn validate_workflow(config: &WorkflowConfig) -> Result<()> {
    match config.kind {
        WorkflowKind::Direct => validate_direct_workflow(config)?,
        WorkflowKind::Sequential => validate_sequential_workflow(config)?,
        WorkflowKind::Parallel => validate_parallel_workflow(config)?,
        WorkflowKind::Routing => validate_routing_workflow(config)?,
    }
    Ok(())
}

fn validate_direct_workflow(config: &WorkflowConfig) -> Result<()> {
    if config.workflow.agent.is_none() {
        return Err(WorkflowError::MissingField(
            "workflow.agent is required for Direct workflow".to_string(),
        ));
    }
    Ok(())
}

fn validate_sequential_workflow(config: &WorkflowConfig) -> Result<()> {
    if config.workflow.agents.is_none() {
        return Err(WorkflowError::MissingField(
            "workflow.agents is required for Sequential workflow".to_string(),
        ));
    }
    let agents = config.workflow.agents.as_ref().unwrap();
    if agents.len() < 2 {
        return Err(WorkflowError::ValidationError(
            "Sequential workflow requires at least 2 agents".to_string(),
        ));
    }
    Ok(())
}

fn validate_parallel_workflow(config: &WorkflowConfig) -> Result<()> {
    if config.workflow.agents.is_none() {
        return Err(WorkflowError::MissingField(
            "workflow.agents is required for Parallel workflow".to_string(),
        ));
    }
    let agents = config.workflow.agents.as_ref().unwrap();
    if agents.len() < 2 {
        return Err(WorkflowError::ValidationError(
            "Parallel workflow requires at least 2 agents".to_string(),
        ));
    }
    Ok(())
}

fn validate_routing_workflow(config: &WorkflowConfig) -> Result<()> {
    if config.workflow.router.is_none() {
        return Err(WorkflowError::MissingField(
            "workflow.router is required for Routing workflow".to_string(),
        ));
    }
    if config.workflow.handlers.is_none() {
        return Err(WorkflowError::MissingField(
            "workflow.handlers is required for Routing workflow".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::*;

    #[test]
    fn test_validate_direct_workflow_missing_agent() {
        let config = WorkflowConfig {
            kind: WorkflowKind::Direct,
            name: None,
            description: None,
            version: None,
            stream: false,
            workflow: WorkflowSpec {
                agent: None,
                agents: None,
                router: None,
                handlers: None,
                output: None,
            },
            memory_persistence: None,
            environment: None,
            runtime: None,
        };
        assert!(validate_workflow(&config).is_err());
    }
}
