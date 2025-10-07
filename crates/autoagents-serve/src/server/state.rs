use crate::builder::BuiltWorkflow;
use autoagents::llm::LLMProvider;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct AppState {
    pub workflows: Arc<RwLock<HashMap<String, Arc<BuiltWorkflow>>>>,
    pub model_cache: Arc<RwLock<HashMap<String, Arc<dyn LLMProvider>>>>,
    pub memory_cache: Arc<RwLock<HashMap<String, Vec<autoagents::llm::chat::ChatMessage>>>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            workflows: Arc::new(RwLock::new(HashMap::new())),
            model_cache: Arc::new(RwLock::new(HashMap::new())),
            memory_cache: Arc::new(RwLock::new(HashMap::new())),
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

    pub async fn add_model(&self, key: String, model: Arc<dyn LLMProvider>) {
        let mut cache = self.model_cache.write().await;
        cache.insert(key, model);
    }

    pub async fn get_model(&self, key: &str) -> Option<Arc<dyn LLMProvider>> {
        let cache = self.model_cache.read().await;
        cache.get(key).cloned()
    }

    pub async fn add_memory(&self, key: String, messages: Vec<autoagents::llm::chat::ChatMessage>) {
        let mut cache = self.memory_cache.write().await;
        cache.insert(key, messages);
    }

    pub async fn get_memory(&self, key: &str) -> Option<Vec<autoagents::llm::chat::ChatMessage>> {
        let cache = self.memory_cache.read().await;
        cache.get(key).cloned()
    }

    pub async fn update_memory(
        &self,
        key: String,
        messages: Vec<autoagents::llm::chat::ChatMessage>,
    ) {
        let mut cache = self.memory_cache.write().await;
        cache.insert(key, messages);
    }
}
