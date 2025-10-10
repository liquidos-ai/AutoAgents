use crate::config::ToolConfig;
use crate::error::{Result, WorkflowError};
use autoagents::core::tool::ToolT;
use std::sync::Arc;

#[cfg(feature = "search-tools")]
use autoagents_toolkit::tools::search::BraveSearch;

pub struct ToolRegistry;

impl ToolRegistry {
    pub fn create_tool(config: &ToolConfig) -> Result<Arc<dyn ToolT>> {
        match config.name.as_str() {
            #[cfg(feature = "search-tools")]
            "brave_search" => {
                let tool: Arc<dyn ToolT> = Arc::new(BraveSearch::new());
                Ok(tool)
            }
            _ => Err(WorkflowError::ToolNotFound(format!(
                "Tool '{}' not found or feature not enabled",
                config.name
            ))),
        }
    }

    pub fn create_tools(configs: &[ToolConfig]) -> Result<Vec<Arc<dyn ToolT>>> {
        configs
            .iter()
            .map(|config| Self::create_tool(config))
            .collect()
    }
}
