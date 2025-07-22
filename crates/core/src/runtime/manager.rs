use super::{Runtime, RuntimeError};
use crate::protocol::RuntimeID;
use futures::future::try_join_all;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::RwLock;

#[derive(Default)]
pub struct RuntimeManager {
    runtimes: RwLock<HashMap<RuntimeID, Arc<dyn Runtime>>>,
}

impl RuntimeManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn register_runtime(&self, runtime: Arc<dyn Runtime>) -> Result<(), RuntimeError> {
        let mut runtimes = self.runtimes.write().await;
        runtimes.insert(runtime.id(), runtime.clone());
        Ok(())
    }

    pub async fn get_runtime(&self, runtime_id: &RuntimeID) -> Option<Arc<dyn Runtime>> {
        let runtimes = self.runtimes.read().await;
        runtimes.get(runtime_id).cloned()
    }

    pub async fn run(&self) -> Result<(), RuntimeError> {
        let runtimes = self.runtimes.read().await;
        let tasks = runtimes
            .values()
            .cloned()
            .map(|runtime| tokio::spawn(async move { runtime.run().await }))
            .collect::<Vec<_>>();
        // Await all in parallel and propagate the first error
        let _ = try_join_all(tasks).await.map_err(RuntimeError::from)?;
        Ok(())
    }

    pub async fn stop(&self) -> Result<(), RuntimeError> {
        let runtimes = self.runtimes.read().await;
        // Call `stop()` on all runtimes
        let tasks = runtimes
            .values()
            .cloned()
            .map(|runtime| tokio::spawn(async move { runtime.stop().await }))
            .collect::<Vec<_>>();

        // Wait for all to finish and propagate first error if any
        let _ = try_join_all(tasks).await.map_err(RuntimeError::from)?;
        Ok(())
    }
}
