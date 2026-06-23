use crate::error::Error;
use crate::runtime::manager::RuntimeManager;
use crate::runtime::{Runtime, RuntimeError};
use crate::utils::BoxEventStream;
use autoagents_protocol::{Event, RuntimeID};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::task::JoinHandle;

/// Errors emitted when managing runtimes and consuming event receivers
#[derive(Debug, thiserror::Error)]
pub enum EnvironmentError {
    #[error("Runtime not found: {0}")]
    RuntimeNotFound(RuntimeID),

    #[error("No default runtime registered")]
    NoDefaultRuntime,

    #[error("Environment is already running")]
    AlreadyRunning,

    #[error("Runtime error: {0}")]
    RuntimeError(#[from] RuntimeError),

    #[error("Error when consuming receiver")]
    EventError,
}

/// Returned when [`Environment::run`] is called while a run task is already active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("Environment is already running")]
pub struct EnvironmentAlreadyRunning;

impl From<EnvironmentAlreadyRunning> for EnvironmentError {
    fn from(_: EnvironmentAlreadyRunning) -> Self {
        Self::AlreadyRunning
    }
}

impl From<EnvironmentAlreadyRunning> for Error {
    fn from(err: EnvironmentAlreadyRunning) -> Self {
        EnvironmentError::from(err).into()
    }
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
        let rid = match runtime_id {
            Some(id) => id,
            None => self
                .default_runtime
                .ok_or(EnvironmentError::NoDefaultRuntime)?,
        };
        self.get_runtime(&rid)
            .await
            .ok_or_else(|| EnvironmentError::RuntimeNotFound(rid).into())
    }

    /// Start all registered runtimes in the background.
    ///
    /// Stores the spawned task handle so [`shutdown`](Self::shutdown) can stop
    /// runtimes and await completion. Returns [`EnvironmentError::AlreadyRunning`]
    /// if a run task is already in progress.
    ///
    /// Use [`wait`](Self::wait) to await the background run task, or
    /// [`shutdown`](Self::shutdown) to stop runtimes and join the task.
    pub fn run(&mut self) -> Result<(), EnvironmentAlreadyRunning> {
        if let Some(handle) = &self.handle {
            if !handle.is_finished() {
                return Err(EnvironmentAlreadyRunning);
            }
            self.handle = None;
        }

        let manager = self.runtime_manager.clone();
        let handle = tokio::spawn(async move { manager.run().await });
        self.handle = Some(handle);
        Ok(())
    }

    /// Await the background task started by [`run`](Self::run).
    ///
    /// Returns `Ok(Ok(()))` when no run task has been started.
    pub async fn wait(&mut self) -> Result<Result<(), RuntimeError>, tokio::task::JoinError> {
        match self.handle.as_mut() {
            Some(handle) => handle.await,
            None => Ok(Ok(())),
        }
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
        let runtime = self
            .get_runtime_or_default(runtime_id)
            .await
            .map_err(|err| match err {
                Error::EnvironmentError(env_err) => env_err,
                _ => EnvironmentError::EventError,
            })?;

        runtime
            .take_event_receiver()
            .await
            .ok_or(EnvironmentError::EventError)
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

    /// Returns whether a background run task is currently in progress.
    pub fn is_running(&self) -> bool {
        self.handle
            .as_ref()
            .is_some_and(|handle| !handle.is_finished())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::SingleThreadedRuntime;
    use tempfile::tempdir;
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
        let dir = tempdir().expect("Unable to create temp dir");
        let config = EnvironmentConfig {
            working_dir: dir.path().to_path_buf(),
        };
        assert_eq!(config.working_dir, dir.path().to_path_buf());
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
    fn test_environment_error_already_running_display() {
        let error = EnvironmentAlreadyRunning;
        assert!(error.to_string().contains("already running"));
    }

    #[test]
    fn test_environment_error_display() {
        let runtime_id = Uuid::new_v4();
        let error = EnvironmentError::RuntimeNotFound(runtime_id);
        assert!(error.to_string().contains("Runtime not found"));
        assert!(error.to_string().contains(&runtime_id.to_string()));
    }

    #[test]
    fn test_environment_error_no_default_display() {
        let error = EnvironmentError::NoDefaultRuntime;
        assert!(error.to_string().contains("No default runtime registered"));
    }

    #[tokio::test]
    async fn test_get_runtime_or_default_no_default_runtime() {
        let env = Environment::new(None);

        let result = env.get_runtime_or_default(None).await;
        assert!(matches!(
            result,
            Err(Error::EnvironmentError(EnvironmentError::NoDefaultRuntime))
        ));
    }

    #[tokio::test]
    async fn test_take_event_receiver_no_default_runtime() {
        let mut env = Environment::new(None);

        let result = env.take_event_receiver(None).await;
        assert!(matches!(result, Err(EnvironmentError::NoDefaultRuntime)));
    }

    #[tokio::test]
    async fn test_subscribe_events_no_default_runtime() {
        let env = Environment::new(None);

        let result = env.subscribe_events(None).await;
        assert!(matches!(result, Err(EnvironmentError::NoDefaultRuntime)));
    }

    #[tokio::test]
    async fn test_environment_run_stores_handle() {
        let mut env = Environment::new(None);
        let runtime = SingleThreadedRuntime::new(None);
        env.register_runtime(runtime).await.unwrap();

        assert!(!env.is_running());
        env.run().expect("run should succeed");
        assert!(env.is_running());
        assert_eq!(
            env.run().expect_err("second run should fail while active"),
            EnvironmentAlreadyRunning
        );

        env.shutdown().await;
        assert!(!env.is_running());
    }

    #[tokio::test]
    async fn test_environment_run_can_restart_after_shutdown() {
        let mut env = Environment::new(None);
        let runtime = SingleThreadedRuntime::new(None);
        env.register_runtime(runtime).await.unwrap();

        env.run().expect("initial run should succeed");
        env.shutdown().await;
        env.run().expect("run after shutdown should succeed");
        env.shutdown().await;
    }

    #[tokio::test]
    async fn test_get_runtime_or_default_runtime_not_found_variant() {
        let mut env = Environment::new(None);
        let runtime = SingleThreadedRuntime::new(None);
        env.register_runtime(runtime).await.unwrap();
        let non_existent_id = Uuid::new_v4();

        let result = env.get_runtime_or_default(Some(non_existent_id)).await;
        assert!(matches!(
            result,
            Err(Error::EnvironmentError(EnvironmentError::RuntimeNotFound(id)))
            if id == non_existent_id
        ));
    }

    #[tokio::test]
    async fn test_take_event_receiver_runtime_not_found() {
        let mut env = Environment::new(None);
        let runtime = SingleThreadedRuntime::new(None);
        env.register_runtime(runtime).await.unwrap();
        let non_existent_id = Uuid::new_v4();

        let result = env.take_event_receiver(Some(non_existent_id)).await;
        assert!(matches!(
            result,
            Err(EnvironmentError::RuntimeNotFound(id)) if id == non_existent_id
        ));
    }
}
