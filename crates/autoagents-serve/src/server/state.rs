use crate::builder::BuiltWorkflow;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct AppState {
    pub workflows: Arc<RwLock<HashMap<String, Arc<BuiltWorkflow>>>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            workflows: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn add_workflow(&self, name: String, workflow: BuiltWorkflow) {
        let mut workflows = self.workflows.write().await;
        workflows.insert(name, Arc::new(workflow));
    }

    pub async fn get_workflow(&self, name: &str) -> Option<Arc<BuiltWorkflow>> {
        let workflows = self.workflows.read().await;
        workflows.get(name).cloned()
    }

    pub async fn list_workflows(&self) -> Vec<String> {
        let workflows = self.workflows.read().await;
        workflows.keys().cloned().collect()
    }
}
