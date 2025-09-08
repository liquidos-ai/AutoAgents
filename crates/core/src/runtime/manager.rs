use super::{Runtime, RuntimeError};
use crate::protocol::RuntimeID;
use futures::future::try_join_all;
use log::error;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::RwLock;

pub struct RuntimeManager {
    runtimes: RwLock<HashMap<RuntimeID, Arc<dyn Runtime>>>,
}

impl RuntimeManager {
    pub fn new() -> Self {
        let runtimes = RwLock::new(HashMap::new());
        RuntimeManager { runtimes }
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
        let _ = try_join_all(tasks).await.map_err(RuntimeError::JoinError)?;
        Ok(())
    }

    /// Spawn all runtimes and return immediately without waiting for completion
    pub async fn run_background(&self) -> Result<(), RuntimeError> {
        let runtimes = self.runtimes.read().await;
        let _ = runtimes.values().cloned().map(|runtime| {
            tokio::spawn(async move {
                if let Err(err) = runtime.run().await {
                    error!("Runtime {} failed: {:?}", runtime.id(), err);
                }
            });
        });
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
        let _ = try_join_all(tasks).await.map_err(RuntimeError::JoinError)?;
        Ok(())
    }
}
