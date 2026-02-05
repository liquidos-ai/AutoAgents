use crate::error::Error;
use crate::protocol::{Event, RuntimeID};
use crate::runtime::manager::RuntimeManager;
use crate::runtime::{Runtime, RuntimeError};
use crate::utils::BoxEventStream;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::task::JoinHandle;

/// Errors emitted when managing runtimes and consuming event receivers
#[derive(Debug, thiserror::Error)]
pub enum EnvironmentError {
    #[error("Runtime not found: {0}")]
    RuntimeNotFound(RuntimeID),

    #[error("Runtime error: {0}")]
    RuntimeError(#[from] RuntimeError),

    #[error("Error when consuming receiver")]
    EventError,
}

/// Configuration for the process environment that owns one or more runtimes.
#[derive(Clone)]
pub struct EnvironmentConfig {
    pub working_dir: PathBuf,
}

impl Default for EnvironmentConfig {
    fn default() -> Self {
        Self {
            working_dir: std::env::current_dir().unwrap_or_default(),
        }
    }
}

/// High-level container that owns one or more runtimes, exposes a unified
/// event receiver, and provides lifecycle helpers for running and shutting down
/// the underlying actor system.
pub struct Environment {
    config: EnvironmentConfig,
    runtime_manager: Arc<RuntimeManager>,
    default_runtime: Option<RuntimeID>,
    handle: Option<JoinHandle<Result<(), RuntimeError>>>,
}

impl Environment {
    /// Create a new environment with optional configuration.
    pub fn new(config: Option<EnvironmentConfig>) -> Self {
        let config = config.unwrap_or_default();
        let runtime_manager = Arc::new(RuntimeManager::new());

        Self {
            config,
            runtime_manager,
            default_runtime: None,
            handle: None,
        }
    }

    /// Register a runtime with this environment and make it the default if none
    /// is set yet.
    pub async fn register_runtime(&mut self, runtime: Arc<dyn Runtime>) -> Result<(), Error> {
        self.runtime_manager
            .register_runtime(runtime.clone())
            .await?;
        if self.default_runtime.is_none() {
            self.default_runtime = Some(runtime.id());
        }
        Ok(())
    }

    /// Access the environment configuration.
    pub fn config(&self) -> &EnvironmentConfig {
        &self.config
    }

    /// Get a runtime by its id, if present.
    pub async fn get_runtime(&self, runtime_id: &RuntimeID) -> Option<Arc<dyn Runtime>> {
        self.runtime_manager.get_runtime(runtime_id).await
    }

    /// Get the specified runtime or the default one when `None` is passed.
    pub async fn get_runtime_or_default(
        &self,
        runtime_id: Option<RuntimeID>,
    ) -> Result<Arc<dyn Runtime>, Error> {
        let rid = runtime_id.unwrap_or(self.default_runtime.unwrap());
        self.get_runtime(&rid)
            .await
            .ok_or_else(|| EnvironmentError::RuntimeNotFound(rid).into())
    }

    /// Start all registered runtimes and return a handle that resolves when
    /// they finish. This will typically run until shutdown is requested.
    pub fn run(&mut self) -> JoinHandle<Result<(), RuntimeError>> {
        let manager = self.runtime_manager.clone();
        // Spawn background task to run the runtimes. This will wait indefinitely
        tokio::spawn(async move { manager.run().await })
    }

    /// Start all registered runtimes and return immediately without waiting
    /// for completion.
    pub async fn run_background(&mut self) -> Result<(), RuntimeError> {
        let manager = self.runtime_manager.clone();
        // Spawn background task to run the runtimes.
        manager.run_background().await
    }

    /// Take the event receiver for a specific runtime (or the default one) so
    /// the caller can consume protocol events. This can only be taken once.
    pub async fn take_event_receiver(
        &mut self,
        runtime_id: Option<RuntimeID>,
    ) -> Result<BoxEventStream<Event>, EnvironmentError> {
        if let Ok(runtime) = self.get_runtime_or_default(runtime_id).await {
            runtime
                .take_event_receiver()
                .await
                .ok_or_else(|| EnvironmentError::EventError)
        } else {
            Err(EnvironmentError::RuntimeNotFound(runtime_id.unwrap()))
        }
    }

    /// Subscribe to runtime events without consuming the receiver.
    pub async fn subscribe_events(
        &self,
        runtime_id: Option<RuntimeID>,
    ) -> Result<BoxEventStream<Event>, EnvironmentError> {
        let runtime = self
            .get_runtime_or_default(runtime_id)
            .await
            .map_err(|err| match err {
                Error::EnvironmentError(env_err) => env_err,
                _ => EnvironmentError::EventError,
            })?;
        Ok(runtime.subscribe_events().await)
    }

    /// Request shutdown on all runtimes and await the run handle if present.
    pub async fn shutdown(&mut self) {
        let _ = self.runtime_manager.stop().await;

        if let Some(handle) = self.handle.take() {
            let _ = handle.await;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::runtime::SingleThreadedRuntime;

    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_environment_config_default() {
        let config = EnvironmentConfig::default();
        assert_eq!(
            config.working_dir,
            std::env::current_dir().unwrap_or_default()
        );
    }

    #[test]
    fn test_environment_config_custom() {
        let config = EnvironmentConfig {
            working_dir: std::path::PathBuf::from("/tmp"),
        };
        assert_eq!(config.working_dir, std::path::PathBuf::from("/tmp"));
    }

    #[tokio::test]
    async fn test_environment_get_runtime() {
        let mut env = Environment::new(None);
        let runtime = SingleThreadedRuntime::new(None);
        let runtime_id = runtime.id;
        env.register_runtime(runtime).await.unwrap();

        // Test getting default runtime
        let runtime = env.get_runtime(&runtime_id).await;

        assert!(runtime.is_some());

        // Test getting non-existent runtime
        let non_existent_id = Uuid::new_v4();
        let runtime = env.get_runtime(&non_existent_id).await;
        assert!(runtime.is_none());
    }

    #[tokio::test]
    async fn test_environment_take_event_receiver() {
        let mut env = Environment::new(None);
        let runtime = SingleThreadedRuntime::new(None);
        let _ = runtime.id;
        env.register_runtime(runtime).await.unwrap();
        let receiver = env.take_event_receiver(None).await;
        assert!(receiver.is_ok());

        // Second call should return None
        let receiver2 = env.take_event_receiver(None).await;
        assert!(receiver2.is_err());
    }

    #[tokio::test]
    async fn test_environment_shutdown() {
        let mut env = Environment::new(None);
        env.shutdown().await;
        // Should not panic
    }

    #[tokio::test]
    async fn test_environment_error_runtime_not_found() {
        let mut env = Environment::new(None);
        let runtime = SingleThreadedRuntime::new(None);
        let _ = runtime.id;
        env.register_runtime(runtime).await.unwrap();
        let non_existent_id = Uuid::new_v4();

        let result = env.get_runtime_or_default(Some(non_existent_id)).await;
        assert!(result.is_err());

        assert!(result.is_err());
        // Just test that it's an error, not the specific variant
        assert!(result.is_err());
    }

    #[test]
    fn test_environment_error_display() {
        let runtime_id = Uuid::new_v4();
        let error = EnvironmentError::RuntimeNotFound(runtime_id);
        assert!(error.to_string().contains("Runtime not found"));
        assert!(error.to_string().contains(&runtime_id.to_string()));
    }
}
