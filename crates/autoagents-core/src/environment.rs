use crate::error::Error;
use crate::runtime::manager::RuntimeManager;
use crate::runtime::{Runtime, RuntimeError};
use crate::utils::BoxEventStream;
use autoagents_protocol::{Event, RuntimeID};
use futures_util::FutureExt;
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

    #[error("Run task join error: {0}")]
    JoinError(#[from] tokio::task::JoinError),

    #[error("Finished run task result was not yet available")]
    RunResultNotReady,
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
///
/// ## Run lifecycle state machine
///
/// An internal [`RuntimeLaunchState`] records how runtimes were last started:
///
/// - **`Idle`** — no active launch mode; safe to call `run()` or `run_background()`.
/// - **`Managed`** — [`run`](Self::run) spawned a join handle tracked by this environment.
///   Await it with [`wait`](Self::wait) or stop with [`shutdown`](Self::shutdown).
/// - **`Background`** — [`run_background`](Self::run_background) started runtimes without storing
///   a handle on the environment. Call [`shutdown`](Self::shutdown) before `run()`.
///
/// `run()` and `run_background()` both call [`reconcile_finished_managed_launch`](Self::reconcile_finished_managed_launch)
/// first. When a managed run task finished without `wait()` or `shutdown()`, that helper joins the
/// stale handle and surfaces any runtime or join error before a new launch proceeds.
///
/// [`wait`](Self::wait) temporarily takes the managed join handle. If the future is dropped early
/// (for example when another branch wins in `tokio::select!`), [`RestoreRunHandleOnDrop`] puts the
/// handle back so a later `shutdown()` can still stop runtimes and join the task. A successful
/// `wait()` clears the handle and returns the launch state to `Idle`.
pub struct Environment {
    config: EnvironmentConfig,
    runtime_manager: Arc<RuntimeManager>,
    default_runtime: Option<RuntimeID>,
    handle: Option<JoinHandle<Result<(), RuntimeError>>>,
    launch_state: RuntimeLaunchState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum RuntimeLaunchState {
    /// No launch in progress; `run()` and `run_background()` are allowed.
    #[default]
    Idle,
    /// [`Environment::run`] spawned a managed join handle stored on the environment.
    Managed,
    /// [`Environment::run_background`] started runtimes without a stored join handle.
    Background,
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
            launch_state: RuntimeLaunchState::Idle,
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
    /// If a previous managed run task finished without [`wait`](Self::wait) or
    /// [`shutdown`](Self::shutdown), its result is joined and returned before
    /// spawning a new run task.
    ///
    /// Use [`wait`](Self::wait) to await the background run task, or
    /// [`shutdown`](Self::shutdown) to stop runtimes and join the task.
    #[allow(clippy::result_large_err)] // Only `AlreadyRunning` is returned from this method.
    pub fn run(&mut self) -> Result<(), EnvironmentError> {
        self.reconcile_finished_managed_launch()?;

        if self.launch_state == RuntimeLaunchState::Background {
            return Err(EnvironmentError::AlreadyRunning);
        }

        if self.is_running() {
            return Err(EnvironmentError::AlreadyRunning);
        }

        let manager = self.runtime_manager.clone();
        let handle = tokio::spawn(async move { manager.run().await });
        self.handle = Some(handle);
        self.launch_state = RuntimeLaunchState::Managed;
        Ok(())
    }

    /// Await the background task started by [`run`](Self::run).
    ///
    /// Returns `Ok(Ok(()))` when no run task has been started. After the task
    /// completes, the stored handle is cleared so subsequent calls return
    /// immediately.
    ///
    /// If this future is dropped before completion (for example when a
    /// `tokio::select!` branch wins elsewhere), the join handle is restored on
    /// the environment so a later [`shutdown`](Self::shutdown) can still stop
    /// runtimes and join the run task.
    pub async fn wait(&mut self) -> Result<Result<(), RuntimeError>, tokio::task::JoinError> {
        let handle = match self.handle.take() {
            Some(handle) => handle,
            None => return Ok(Ok(())),
        };

        let mut guard = RestoreRunHandleOnDrop {
            environment: self,
            handle: Some(handle),
        };

        let join_result = guard.handle.as_mut().expect("handle was just set").await;

        guard.handle = None;
        guard.environment.launch_state = RuntimeLaunchState::Idle;

        join_result
    }

    /// Start all registered runtimes and return immediately without waiting
    /// for completion.
    ///
    /// Cannot be combined with [`run`](Self::run) on the same environment
    /// instance without calling [`shutdown`](Self::shutdown) first.
    pub async fn run_background(&mut self) -> Result<(), EnvironmentError> {
        self.reconcile_finished_managed_launch()?;

        if self.launch_state != RuntimeLaunchState::Idle || self.is_running() {
            return Err(EnvironmentError::AlreadyRunning);
        }

        let manager = self.runtime_manager.clone();
        manager
            .run_background()
            .await
            .map_err(EnvironmentError::RuntimeError)?;
        self.launch_state = RuntimeLaunchState::Background;
        Ok(())
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
    pub async fn shutdown(&mut self) -> Result<(), EnvironmentError> {
        let stop_result = self.runtime_manager.stop().await;

        let join_result = if let Some(handle) = self.handle.take() {
            Some(handle.await)
        } else {
            None
        };

        self.launch_state = RuntimeLaunchState::Idle;

        if let Err(e) = stop_result {
            return Err(EnvironmentError::RuntimeError(e));
        }

        match join_result {
            None | Some(Ok(Ok(()))) => Ok(()),
            Some(Ok(Err(e))) => Err(EnvironmentError::RuntimeError(e)),
            Some(Err(e)) => Err(EnvironmentError::JoinError(e)),
        }
    }

    /// Returns whether a background run task is currently in progress.
    pub fn is_running(&self) -> bool {
        self.handle
            .as_ref()
            .is_some_and(|handle| !handle.is_finished())
    }

    #[allow(clippy::result_large_err)]
    /// Join a managed run task that finished without `wait()` or `shutdown()`.
    ///
    /// No-op unless `launch_state` is [`RuntimeLaunchState::Managed`]. When the stored handle is
    /// still running, the handle is restored and this returns `Ok(())`. When the handle finished,
    /// it is joined via [`join_finished_handle`](Self::join_finished_handle) and `launch_state`
    /// becomes `Idle`.
    fn reconcile_finished_managed_launch(&mut self) -> Result<(), EnvironmentError> {
        if self.launch_state != RuntimeLaunchState::Managed {
            return Ok(());
        }

        let Some(handle) = self.handle.take() else {
            self.launch_state = RuntimeLaunchState::Idle;
            return Ok(());
        };

        if !handle.is_finished() {
            self.handle = Some(handle);
            return Ok(());
        }

        self.launch_state = RuntimeLaunchState::Idle;
        Self::join_finished_handle(handle)
    }

    #[allow(clippy::result_large_err)]
    fn join_finished_handle(
        handle: JoinHandle<Result<(), RuntimeError>>,
    ) -> Result<(), EnvironmentError> {
        debug_assert!(
            handle.is_finished(),
            "join_finished_handle requires a finished run task"
        );

        match handle.now_or_never() {
            Some(Ok(Ok(()))) => Ok(()),
            Some(Ok(Err(e))) => Err(EnvironmentError::RuntimeError(e)),
            Some(Err(e)) => Err(EnvironmentError::JoinError(e)),
            None => Err(EnvironmentError::RunResultNotReady),
        }
    }
}

/// Restores a managed run [`JoinHandle`] when a [`Environment::wait`] future is
/// cancelled before the join completes.
struct RestoreRunHandleOnDrop<'a> {
    environment: &'a mut Environment,
    handle: Option<JoinHandle<Result<(), RuntimeError>>>,
}

impl Drop for RestoreRunHandleOnDrop<'_> {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            self.environment.handle = Some(handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::SingleThreadedRuntime;
    use tempfile::tempdir;
    use tokio::sync::mpsc;
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
        env.shutdown()
            .await
            .expect("shutdown should succeed when idle");
    }

    #[tokio::test]
    async fn test_environment_error_runtime_not_found() {
        let mut env = Environment::new(None);
        let runtime = SingleThreadedRuntime::new(None);
        let _ = runtime.id;
        env.register_runtime(runtime).await.unwrap();
        let non_existent_id = Uuid::new_v4();

        let result = env.get_runtime_or_default(Some(non_existent_id)).await;
        assert!(matches!(
            result,
            Err(Error::EnvironmentError(EnvironmentError::RuntimeNotFound(id)))
            if id == non_existent_id
        ));
    }

    #[test]
    fn test_environment_error_already_running_display() {
        let error = EnvironmentError::AlreadyRunning;
        assert!(error.to_string().contains("already running"));
    }

    #[test]
    fn test_environment_error_run_result_not_ready_display() {
        let error = EnvironmentError::RunResultNotReady;
        assert!(error.to_string().contains("not yet available"));
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
        assert!(matches!(env.run(), Err(EnvironmentError::AlreadyRunning)));

        env.shutdown().await.expect("shutdown should succeed");
        assert!(!env.is_running());
    }

    #[tokio::test]
    async fn test_environment_run_can_restart_after_shutdown() {
        let mut env = Environment::new(None);
        let runtime = SingleThreadedRuntime::new(None);
        env.register_runtime(runtime).await.unwrap();

        env.run().expect("initial run should succeed");
        env.shutdown().await.expect("shutdown should succeed");

        // Environment allows spawning a new run task after shutdown. The same
        // SingleThreadedRuntime instance cannot re-enter its event loop, so the
        // second run task fails when joined.
        env.run().expect("run after shutdown should succeed");
        let run_result = env.wait().await.expect("wait should join run task");
        assert!(run_result.is_err());

        env.shutdown()
            .await
            .expect("shutdown should succeed when idle after failed run");
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

    #[tokio::test]
    async fn test_environment_wait_restores_handle_when_cancelled() {
        use tokio::time::{Duration, sleep};

        let mut env = Environment::new(None);
        let runtime = SingleThreadedRuntime::new(None);
        env.register_runtime(runtime).await.unwrap();

        env.run().expect("run should succeed");
        assert!(env.is_running());

        {
            let wait_fut = env.wait();
            tokio::pin!(wait_fut);

            tokio::select! {
                _ = &mut wait_fut => panic!("run task should not finish immediately"),
                _ = sleep(Duration::from_millis(10)) => {}
            }
        }

        assert!(
            env.is_running(),
            "cancelled wait should restore the join handle"
        );

        env.shutdown()
            .await
            .expect("shutdown should join the restored handle");
        assert!(!env.is_running());
    }

    #[tokio::test]
    async fn test_environment_wait_is_idempotent() {
        use tokio::time::{Duration, timeout};

        let mut env = Environment::new(None);
        let runtime = SingleThreadedRuntime::new(None);
        env.register_runtime(runtime).await.unwrap();

        env.run().expect("run should succeed");
        env.shutdown().await.expect("shutdown should succeed");

        for _ in 0..2 {
            let result = timeout(Duration::from_secs(1), env.wait())
                .await
                .expect("wait should not hang");
            assert!(result.is_ok());
        }
    }

    #[derive(Clone, Copy)]
    enum ImmediateRuntimeBehavior {
        Success,
        Error,
    }

    struct ImmediateRuntime {
        id: RuntimeID,
        behavior: ImmediateRuntimeBehavior,
        tx: mpsc::Sender<Event>,
    }

    #[async_trait::async_trait]
    impl Runtime for ImmediateRuntime {
        fn id(&self) -> RuntimeID {
            self.id
        }

        async fn subscribe_any(
            &self,
            _topic_name: &str,
            _topic_type: std::any::TypeId,
            _actor: Arc<dyn crate::actor::AnyActor>,
        ) -> Result<(), RuntimeError> {
            Ok(())
        }

        async fn publish_any(
            &self,
            _topic_name: &str,
            _topic_type: std::any::TypeId,
            _message: Arc<dyn std::any::Any + Send + Sync>,
        ) -> Result<(), RuntimeError> {
            Ok(())
        }

        fn tx(&self) -> mpsc::Sender<Event> {
            self.tx.clone()
        }

        async fn transport(&self) -> Arc<dyn crate::actor::Transport> {
            Arc::new(crate::actor::LocalTransport)
        }

        async fn take_event_receiver(&self) -> Option<BoxEventStream<Event>> {
            None
        }

        async fn subscribe_events(&self) -> BoxEventStream<Event> {
            Box::pin(futures::stream::empty())
        }

        async fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            match self.behavior {
                ImmediateRuntimeBehavior::Success => Ok(()),
                ImmediateRuntimeBehavior::Error => {
                    Err(std::io::Error::other("immediate runtime run failed").into())
                }
            }
        }

        async fn stop(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_environment_wait_awaits_completed_run_task() {
        let mut env = Environment::new(None);
        let (tx, _rx) = mpsc::channel(1);
        let runtime = Arc::new(ImmediateRuntime {
            id: RuntimeID::new_v4(),
            behavior: ImmediateRuntimeBehavior::Success,
            tx,
        }) as Arc<dyn Runtime>;
        env.register_runtime(runtime).await.unwrap();

        env.run().expect("run should succeed");
        let wait_result = env.wait().await.expect("wait join should succeed");
        assert!(wait_result.is_ok());
    }

    #[tokio::test]
    async fn test_environment_run_restarts_after_finished_handle() {
        use tokio::time::{Duration, sleep};

        let mut env = Environment::new(None);
        let (tx, _rx) = mpsc::channel(1);
        let runtime = Arc::new(ImmediateRuntime {
            id: RuntimeID::new_v4(),
            behavior: ImmediateRuntimeBehavior::Success,
            tx,
        }) as Arc<dyn Runtime>;
        env.register_runtime(runtime).await.unwrap();

        env.run().expect("initial run should succeed");
        sleep(Duration::from_millis(20)).await;
        assert!(!env.is_running());

        env.run().expect("run after finished handle should succeed");
        env.shutdown().await.expect("shutdown should succeed");
    }

    #[tokio::test]
    async fn test_environment_run_background_starts_runtimes() {
        let mut env = Environment::new(None);
        let (tx, _rx) = mpsc::channel(1);
        let runtime = Arc::new(ImmediateRuntime {
            id: RuntimeID::new_v4(),
            behavior: ImmediateRuntimeBehavior::Success,
            tx,
        }) as Arc<dyn Runtime>;
        env.register_runtime(runtime).await.unwrap();

        env.run_background()
            .await
            .expect("run_background should succeed");

        env.shutdown().await.expect("shutdown should succeed");
    }

    #[tokio::test]
    async fn test_environment_run_background_rejects_after_run() {
        let mut env = Environment::new(None);
        let runtime = SingleThreadedRuntime::new(None);
        env.register_runtime(runtime).await.unwrap();

        env.run().expect("run should succeed");
        assert!(matches!(
            env.run_background().await,
            Err(EnvironmentError::AlreadyRunning)
        ));

        env.shutdown().await.expect("shutdown should succeed");
    }

    #[tokio::test]
    async fn test_environment_run_rejects_after_run_background() {
        let mut env = Environment::new(None);
        let (tx, _rx) = mpsc::channel(1);
        let runtime = Arc::new(ImmediateRuntime {
            id: RuntimeID::new_v4(),
            behavior: ImmediateRuntimeBehavior::Success,
            tx,
        }) as Arc<dyn Runtime>;
        env.register_runtime(runtime).await.unwrap();

        env.run_background()
            .await
            .expect("run_background should succeed");
        assert!(matches!(env.run(), Err(EnvironmentError::AlreadyRunning)));

        env.shutdown().await.expect("shutdown should succeed");
    }

    #[tokio::test]
    async fn test_environment_wait_propagates_run_failure() {
        let mut env = Environment::new(None);
        let (tx, _rx) = mpsc::channel(1);
        let runtime = Arc::new(ImmediateRuntime {
            id: RuntimeID::new_v4(),
            behavior: ImmediateRuntimeBehavior::Error,
            tx,
        }) as Arc<dyn Runtime>;
        env.register_runtime(runtime).await.unwrap();

        env.run().expect("run should succeed");
        let wait_result = env.wait().await.expect("wait join should succeed");
        assert!(wait_result.is_err());
    }

    #[tokio::test]
    async fn test_environment_run_surfaces_prior_failure_without_wait() {
        use tokio::time::{Duration, sleep};

        let mut env = Environment::new(None);
        let (tx, _rx) = mpsc::channel(1);
        let runtime = Arc::new(ImmediateRuntime {
            id: RuntimeID::new_v4(),
            behavior: ImmediateRuntimeBehavior::Error,
            tx,
        }) as Arc<dyn Runtime>;
        env.register_runtime(runtime).await.unwrap();

        env.run().expect("initial run should succeed");
        sleep(Duration::from_millis(20)).await;
        assert!(!env.is_running());

        let err = env
            .run()
            .expect_err("restart should surface prior run failure");
        assert!(matches!(err, EnvironmentError::RuntimeError(_)));
    }

    #[tokio::test]
    async fn test_environment_run_background_surfaces_prior_failure_without_wait() {
        use tokio::time::{Duration, sleep};

        let mut env = Environment::new(None);
        let (tx, _rx) = mpsc::channel(1);
        let runtime = Arc::new(ImmediateRuntime {
            id: RuntimeID::new_v4(),
            behavior: ImmediateRuntimeBehavior::Error,
            tx,
        }) as Arc<dyn Runtime>;
        env.register_runtime(runtime).await.unwrap();

        env.run().expect("initial run should succeed");
        sleep(Duration::from_millis(20)).await;
        assert!(!env.is_running());

        let err = env
            .run_background()
            .await
            .expect_err("run_background should surface prior run failure");
        assert!(matches!(err, EnvironmentError::RuntimeError(_)));
    }

    #[test]
    fn test_environment_config_accessor() {
        let dir = tempdir().expect("Unable to create temp dir");
        let config = EnvironmentConfig {
            working_dir: dir.path().to_path_buf(),
        };
        let env = Environment::new(Some(config.clone()));
        assert_eq!(env.config().working_dir, config.working_dir);
    }

    #[tokio::test]
    async fn test_subscribe_events_with_default_runtime() {
        let mut env = Environment::new(None);
        let runtime = SingleThreadedRuntime::new(None);
        env.register_runtime(runtime).await.unwrap();

        let stream = env.subscribe_events(None).await;
        assert!(stream.is_ok());
    }

    #[tokio::test]
    async fn test_get_runtime_or_default_uses_default_runtime() {
        let mut env = Environment::new(None);
        let runtime = SingleThreadedRuntime::new(None);
        let runtime_id = runtime.id;
        env.register_runtime(runtime).await.unwrap();

        let resolved = env
            .get_runtime_or_default(None)
            .await
            .expect("default runtime should resolve");
        assert_eq!(resolved.id(), runtime_id);
    }
}
