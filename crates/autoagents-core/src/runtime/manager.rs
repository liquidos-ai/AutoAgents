use super::{Runtime, RuntimeError};
use autoagents_protocol::RuntimeID;
use futures::future::{join_all, try_join_all};
use log::error;
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::{sync::RwLock, task::JoinError};

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
            .map(|runtime| {
                let runtime = Arc::clone(runtime);
                tokio::spawn(async move { runtime.run().await })
            })
            .collect::<Vec<_>>();
        let results = try_join_all(tasks).await.map_err(RuntimeError::JoinError)?;
        for result in results {
            result.map_err(|err| RuntimeError::OperationFailed(err.to_string()))?;
        }
        Ok(())
    }

    /// Spawn all runtimes and return immediately without waiting for completion
    pub async fn run_background(&self) -> Result<(), RuntimeError> {
        let runtimes = self.runtimes.read().await;
        for runtime in runtimes.values() {
            let runtime = Arc::clone(runtime);
            tokio::spawn(async move {
                if let Err(err) = runtime.run().await {
                    error!("Runtime {} failed: {:?}", runtime.id(), err);
                }
            });
        }
        Ok(())
    }

    /// Request shutdown of all registered runtimes and wait for every
    /// [`Runtime::stop`] future to complete.
    ///
    /// This waits without a deadline. Use [`stop_with_timeout`](Self::stop_with_timeout)
    /// when an unresponsive runtime must not block shutdown indefinitely.
    pub async fn stop(&self) -> Result<(), RuntimeError> {
        let runtimes = self.runtimes.read().await;
        // Call `stop()` on all runtimes
        let tasks = runtimes
            .values()
            .map(|runtime| {
                let runtime = Arc::clone(runtime);
                tokio::spawn(async move { runtime.stop().await })
            })
            .collect::<Vec<_>>();

        let results = try_join_all(tasks).await.map_err(RuntimeError::JoinError)?;
        for result in results {
            result.map_err(|err| RuntimeError::OperationFailed(err.to_string()))?;
        }
        Ok(())
    }

    /// Request shutdown of all registered runtimes, waiting at most `timeout`
    /// for each [`Runtime::stop`] future to complete.
    ///
    /// Every runtime is stopped concurrently and the deadline applies to each
    /// one individually, so the call returns after roughly `timeout` even when
    /// several runtimes are unresponsive.
    ///
    /// Unlike [`stop`](Self::stop), this never short-circuits: every runtime is
    /// awaited (up to the deadline) so a single unresponsive runtime cannot
    /// prevent the others from being stopped.
    ///
    /// # Errors
    ///
    /// Outcomes are reported in a fixed precedence so the result is
    /// deterministic when runtimes fail in different ways:
    ///
    /// 1. [`RuntimeError::ShutdownTimeout`] when any runtime missed the
    ///    deadline, carrying the ids of all unresponsive runtimes.
    /// 2. [`RuntimeError::JoinError`] when any `stop()` future panicked.
    /// 3. [`RuntimeError::OperationFailed`] when any `stop()` future returned an
    ///    error.
    ///
    /// # Cancellation
    ///
    /// Exceeding the deadline stops the manager from waiting; it does not abort
    /// the underlying `stop()` future. A timed-out shutdown keeps running
    /// detached until it completes on its own, so callers should treat
    /// [`RuntimeError::ShutdownTimeout`] as "shutdown is still in progress"
    /// rather than "shutdown was cancelled".
    pub async fn stop_with_timeout(&self, timeout: Duration) -> Result<(), RuntimeError> {
        let runtimes = self.runtimes.read().await;
        // Call `stop()` on all runtimes, keeping each runtime id so unresponsive
        // runtimes can be named in the error.
        let tasks = runtimes
            .values()
            .map(|runtime| {
                let runtime = Arc::clone(runtime);
                let runtime_id = runtime.id();
                (
                    runtime_id,
                    tokio::spawn(async move { runtime.stop().await }),
                )
            })
            .collect::<Vec<_>>();

        let outcomes = join_all(tasks.into_iter().map(|(runtime_id, task)| async move {
            (runtime_id, tokio::time::timeout(timeout, task).await)
        }))
        .await;

        let mut timed_out: Vec<RuntimeID> = Vec::new();
        let mut join_error: Option<JoinError> = None;
        let mut operation_error: Option<String> = None;

        for (runtime_id, outcome) in outcomes {
            match outcome {
                Err(_elapsed) => {
                    error!("Runtime {runtime_id} did not stop within {timeout:?}");
                    timed_out.push(runtime_id);
                }
                Ok(Err(err)) => {
                    error!("Runtime {runtime_id} panicked while stopping: {err}");
                    if join_error.is_none() {
                        join_error = Some(err);
                    }
                }
                Ok(Ok(Err(err))) => {
                    error!("Runtime {runtime_id} failed to stop: {err}");
                    if operation_error.is_none() {
                        operation_error = Some(err.to_string());
                    }
                }
                Ok(Ok(Ok(()))) => {}
            }
        }

        if !timed_out.is_empty() {
            // Registration order is not observable through the `HashMap`, so sort
            // to keep the reported ids stable across calls.
            timed_out.sort_unstable();
            return Err(RuntimeError::ShutdownTimeout {
                runtime_ids: timed_out,
                timeout,
            });
        }

        if let Some(err) = join_error {
            return Err(RuntimeError::JoinError(err));
        }

        if let Some(err) = operation_error {
            return Err(RuntimeError::OperationFailed(err));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::{AnyActor, LocalTransport, Transport};
    use crate::utils::BoxEventStream;
    use async_trait::async_trait;
    use autoagents_protocol::Event;
    use futures::stream;
    use std::any::{Any, TypeId};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::{Notify, mpsc};
    use tokio::time::{Duration, timeout};

    #[derive(Clone, Copy)]
    enum Behavior {
        Success,
        Error,
        Panic,
        /// Never resolves, emulating a runtime that cannot complete shutdown.
        Hang,
    }

    struct TestRuntime {
        id: RuntimeID,
        run_behavior: Behavior,
        stop_behavior: Behavior,
        run_calls: Arc<AtomicUsize>,
        stop_calls: Arc<AtomicUsize>,
        stop_completions: Arc<AtomicUsize>,
        run_started: Option<Arc<Notify>>,
        stop_gate: Option<Arc<Notify>>,
        tx: mpsc::Sender<Event>,
    }

    impl TestRuntime {
        fn new(run_behavior: Behavior, stop_behavior: Behavior) -> Self {
            let (tx, _rx) = mpsc::channel(1);
            Self {
                id: RuntimeID::new_v4(),
                run_behavior,
                stop_behavior,
                run_calls: Arc::new(AtomicUsize::new(0)),
                stop_calls: Arc::new(AtomicUsize::new(0)),
                stop_completions: Arc::new(AtomicUsize::new(0)),
                run_started: None,
                stop_gate: None,
                tx,
            }
        }

        fn with_run_notify(mut self, run_started: Arc<Notify>) -> Self {
            self.run_started = Some(run_started);
            self
        }

        /// Block `stop()` until the returned gate is notified.
        fn with_stop_gate(mut self, stop_gate: Arc<Notify>) -> Self {
            self.stop_gate = Some(stop_gate);
            self
        }

        fn run_calls(&self) -> Arc<AtomicUsize> {
            Arc::clone(&self.run_calls)
        }

        fn stop_calls(&self) -> Arc<AtomicUsize> {
            Arc::clone(&self.stop_calls)
        }

        fn stop_completions(&self) -> Arc<AtomicUsize> {
            Arc::clone(&self.stop_completions)
        }
    }

    #[async_trait]
    impl Runtime for TestRuntime {
        fn id(&self) -> RuntimeID {
            self.id
        }

        async fn subscribe_any(
            &self,
            _topic_name: &str,
            _topic_type: TypeId,
            _actor: Arc<dyn AnyActor>,
        ) -> Result<(), RuntimeError> {
            Ok(())
        }

        async fn publish_any(
            &self,
            _topic_name: &str,
            _topic_type: TypeId,
            _message: Arc<dyn Any + Send + Sync>,
        ) -> Result<(), RuntimeError> {
            Ok(())
        }

        fn tx(&self) -> mpsc::Sender<Event> {
            self.tx.clone()
        }

        async fn transport(&self) -> Arc<dyn Transport> {
            Arc::new(LocalTransport)
        }

        async fn take_event_receiver(&self) -> Option<BoxEventStream<Event>> {
            None
        }

        async fn subscribe_events(&self) -> BoxEventStream<Event> {
            Box::pin(stream::empty())
        }

        async fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            self.run_calls.fetch_add(1, Ordering::SeqCst);
            if let Some(run_started) = &self.run_started {
                run_started.notify_waiters();
            }

            match self.run_behavior {
                Behavior::Success => Ok(()),
                Behavior::Error => Err(std::io::Error::other("run failed").into()),
                Behavior::Panic => panic!("runtime run panic"),
                Behavior::Hang => std::future::pending().await,
            }
        }

        async fn stop(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            self.stop_calls.fetch_add(1, Ordering::SeqCst);
            if let Some(stop_gate) = &self.stop_gate {
                stop_gate.notified().await;
            }

            let result: Result<(), Box<dyn std::error::Error + Send + Sync>> =
                match self.stop_behavior {
                    Behavior::Success => Ok(()),
                    Behavior::Error => Err(std::io::Error::other("stop failed").into()),
                    Behavior::Panic => panic!("runtime stop panic"),
                    Behavior::Hang => std::future::pending().await,
                };

            self.stop_completions.fetch_add(1, Ordering::SeqCst);
            result
        }
    }

    #[tokio::test]
    async fn register_runtime_allows_lookup_by_id() {
        let manager = RuntimeManager::new();
        let runtime: Arc<dyn Runtime> =
            Arc::new(TestRuntime::new(Behavior::Success, Behavior::Success));
        let runtime_id = runtime.id();

        manager
            .register_runtime(Arc::clone(&runtime))
            .await
            .expect("runtime registers");

        let fetched = manager
            .get_runtime(&runtime_id)
            .await
            .expect("runtime exists");
        assert_eq!(fetched.id(), runtime_id);
    }

    #[tokio::test]
    async fn run_executes_all_registered_runtimes() {
        let manager = RuntimeManager::new();
        let first = TestRuntime::new(Behavior::Success, Behavior::Success);
        let second = TestRuntime::new(Behavior::Success, Behavior::Success);
        let first_calls = first.run_calls();
        let second_calls = second.run_calls();

        manager
            .register_runtime(Arc::new(first))
            .await
            .expect("register first runtime");
        manager
            .register_runtime(Arc::new(second))
            .await
            .expect("register second runtime");

        manager.run().await.expect("all runtimes run");

        assert_eq!(first_calls.load(Ordering::SeqCst), 1);
        assert_eq!(second_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn run_returns_join_error_when_runtime_panics() {
        let manager = RuntimeManager::new();
        manager
            .register_runtime(Arc::new(TestRuntime::new(
                Behavior::Panic,
                Behavior::Success,
            )))
            .await
            .expect("register runtime");

        let err = manager
            .run()
            .await
            .expect_err("panic should surface as join error");
        assert!(matches!(err, RuntimeError::JoinError(_)));
    }

    #[tokio::test]
    async fn run_returns_runtime_error_when_runtime_fails() {
        let manager = RuntimeManager::new();
        manager
            .register_runtime(Arc::new(TestRuntime::new(
                Behavior::Error,
                Behavior::Success,
            )))
            .await
            .expect("register runtime");

        let err = manager
            .run()
            .await
            .expect_err("runtime error should be surfaced");
        assert!(matches!(err, RuntimeError::OperationFailed(_)));
        assert!(err.to_string().contains("run failed"));
    }

    #[tokio::test]
    async fn run_background_starts_runtimes_without_blocking() {
        let manager = RuntimeManager::new();
        let started = Arc::new(Notify::new());
        let runtime = TestRuntime::new(Behavior::Error, Behavior::Success)
            .with_run_notify(Arc::clone(&started));
        let run_calls = runtime.run_calls();

        manager
            .register_runtime(Arc::new(runtime))
            .await
            .expect("register runtime");

        manager
            .run_background()
            .await
            .expect("background execution starts");

        timeout(Duration::from_secs(1), started.notified())
            .await
            .expect("background task starts promptly");
        assert_eq!(run_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn stop_executes_all_registered_runtimes() {
        let manager = RuntimeManager::new();
        let first = TestRuntime::new(Behavior::Success, Behavior::Success);
        let second = TestRuntime::new(Behavior::Success, Behavior::Success);
        let first_calls = first.stop_calls();
        let second_calls = second.stop_calls();

        manager
            .register_runtime(Arc::new(first))
            .await
            .expect("register first runtime");
        manager
            .register_runtime(Arc::new(second))
            .await
            .expect("register second runtime");

        manager.stop().await.expect("all runtimes stop");

        assert_eq!(first_calls.load(Ordering::SeqCst), 1);
        assert_eq!(second_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn stop_returns_join_error_when_runtime_panics() {
        let manager = RuntimeManager::new();
        manager
            .register_runtime(Arc::new(TestRuntime::new(
                Behavior::Success,
                Behavior::Panic,
            )))
            .await
            .expect("register runtime");

        let err = manager
            .stop()
            .await
            .expect_err("panic should surface as join error");
        assert!(matches!(err, RuntimeError::JoinError(_)));
    }

    #[tokio::test]
    async fn stop_returns_runtime_error_when_runtime_fails() {
        let manager = RuntimeManager::new();
        manager
            .register_runtime(Arc::new(TestRuntime::new(
                Behavior::Success,
                Behavior::Error,
            )))
            .await
            .expect("register runtime");

        let err = manager
            .stop()
            .await
            .expect_err("runtime error should be surfaced");
        assert!(matches!(err, RuntimeError::OperationFailed(_)));
        assert!(err.to_string().contains("stop failed"));
    }

    /// Deadline handed to `stop_with_timeout`. Short so hanging runtimes are
    /// reported quickly.
    const STOP_TIMEOUT: Duration = Duration::from_millis(50);
    /// Upper bound for the whole call; generously above `STOP_TIMEOUT` so a slow
    /// machine cannot make the test flaky while still catching a hang.
    const WATCHDOG: Duration = Duration::from_secs(5);

    #[tokio::test]
    async fn stop_with_timeout_stops_all_registered_runtimes() {
        let manager = RuntimeManager::new();
        let first = TestRuntime::new(Behavior::Success, Behavior::Success);
        let second = TestRuntime::new(Behavior::Success, Behavior::Success);
        let first_calls = first.stop_calls();
        let second_calls = second.stop_calls();

        manager
            .register_runtime(Arc::new(first))
            .await
            .expect("register first runtime");
        manager
            .register_runtime(Arc::new(second))
            .await
            .expect("register second runtime");

        manager
            .stop_with_timeout(STOP_TIMEOUT)
            .await
            .expect("all runtimes stop within the deadline");

        assert_eq!(first_calls.load(Ordering::SeqCst), 1);
        assert_eq!(second_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn stop_with_timeout_succeeds_without_registered_runtimes() {
        let manager = RuntimeManager::new();

        manager
            .stop_with_timeout(STOP_TIMEOUT)
            .await
            .expect("stopping an empty manager succeeds");
    }

    #[tokio::test]
    async fn stop_with_timeout_reports_unresponsive_runtime() {
        let manager = RuntimeManager::new();
        let unresponsive = TestRuntime::new(Behavior::Success, Behavior::Hang);
        let unresponsive_id = unresponsive.id();
        let responsive = TestRuntime::new(Behavior::Success, Behavior::Success);
        let responsive_calls = responsive.stop_calls();

        manager
            .register_runtime(Arc::new(unresponsive))
            .await
            .expect("register unresponsive runtime");
        manager
            .register_runtime(Arc::new(responsive))
            .await
            .expect("register responsive runtime");

        let err = timeout(WATCHDOG, manager.stop_with_timeout(STOP_TIMEOUT))
            .await
            .expect("stop_with_timeout must not wait indefinitely")
            .expect_err("unresponsive runtime should be reported");

        assert!(matches!(
            &err,
            RuntimeError::ShutdownTimeout { runtime_ids, timeout }
            if runtime_ids.as_slice() == [unresponsive_id] && *timeout == STOP_TIMEOUT
        ));
        assert_eq!(
            responsive_calls.load(Ordering::SeqCst),
            1,
            "an unresponsive runtime must not prevent the others from stopping"
        );
    }

    #[tokio::test]
    async fn stop_with_timeout_reports_every_unresponsive_runtime() {
        let manager = RuntimeManager::new();
        let first = TestRuntime::new(Behavior::Success, Behavior::Hang);
        let second = TestRuntime::new(Behavior::Success, Behavior::Hang);
        let mut expected_ids = vec![first.id(), second.id()];
        expected_ids.sort_unstable();

        manager
            .register_runtime(Arc::new(first))
            .await
            .expect("register first runtime");
        manager
            .register_runtime(Arc::new(second))
            .await
            .expect("register second runtime");

        let err = timeout(WATCHDOG, manager.stop_with_timeout(STOP_TIMEOUT))
            .await
            .expect("stop_with_timeout must not wait indefinitely")
            .expect_err("unresponsive runtimes should be reported");

        match err {
            RuntimeError::ShutdownTimeout { runtime_ids, .. } => {
                assert_eq!(runtime_ids, expected_ids);
            }
            other => panic!("expected a shutdown timeout, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stop_with_timeout_prefers_timeout_over_stop_failure() {
        let manager = RuntimeManager::new();
        let unresponsive = TestRuntime::new(Behavior::Success, Behavior::Hang);
        let unresponsive_id = unresponsive.id();

        manager
            .register_runtime(Arc::new(unresponsive))
            .await
            .expect("register unresponsive runtime");
        manager
            .register_runtime(Arc::new(TestRuntime::new(
                Behavior::Success,
                Behavior::Error,
            )))
            .await
            .expect("register failing runtime");

        let err = timeout(WATCHDOG, manager.stop_with_timeout(STOP_TIMEOUT))
            .await
            .expect("stop_with_timeout must not wait indefinitely")
            .expect_err("shutdown should fail");

        assert!(matches!(
            &err,
            RuntimeError::ShutdownTimeout { runtime_ids, .. }
            if runtime_ids.as_slice() == [unresponsive_id]
        ));
    }

    #[tokio::test]
    async fn stop_with_timeout_surfaces_runtime_error() {
        let manager = RuntimeManager::new();
        manager
            .register_runtime(Arc::new(TestRuntime::new(
                Behavior::Success,
                Behavior::Error,
            )))
            .await
            .expect("register runtime");

        let err = manager
            .stop_with_timeout(STOP_TIMEOUT)
            .await
            .expect_err("runtime error should be surfaced");
        assert!(matches!(err, RuntimeError::OperationFailed(_)));
        assert!(err.to_string().contains("stop failed"));
    }

    #[tokio::test]
    async fn stop_with_timeout_returns_join_error_when_runtime_panics() {
        let manager = RuntimeManager::new();
        manager
            .register_runtime(Arc::new(TestRuntime::new(
                Behavior::Success,
                Behavior::Panic,
            )))
            .await
            .expect("register runtime");

        let err = manager
            .stop_with_timeout(STOP_TIMEOUT)
            .await
            .expect_err("panic should surface as join error");
        assert!(matches!(err, RuntimeError::JoinError(_)));
    }

    #[tokio::test]
    async fn stop_with_timeout_does_not_abort_a_late_runtime() {
        let manager = RuntimeManager::new();
        let gate = Arc::new(Notify::new());
        let runtime = TestRuntime::new(Behavior::Success, Behavior::Success)
            .with_stop_gate(Arc::clone(&gate));
        let stop_completions = runtime.stop_completions();

        manager
            .register_runtime(Arc::new(runtime))
            .await
            .expect("register runtime");

        timeout(WATCHDOG, manager.stop_with_timeout(STOP_TIMEOUT))
            .await
            .expect("stop_with_timeout must not wait indefinitely")
            .expect_err("gated runtime should miss the deadline");
        assert_eq!(stop_completions.load(Ordering::SeqCst), 0);

        // The deadline only stops the manager from waiting: the detached stop
        // operation still runs to completion once it is unblocked.
        gate.notify_one();
        timeout(WATCHDOG, async {
            while stop_completions.load(Ordering::SeqCst) == 0 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("the timed-out stop operation should still complete");
    }

    #[test]
    fn shutdown_timeout_error_names_the_unresponsive_runtimes() {
        let runtime_id = RuntimeID::new_v4();
        let err = RuntimeError::ShutdownTimeout {
            runtime_ids: vec![runtime_id],
            timeout: STOP_TIMEOUT,
        };

        let message = err.to_string();
        assert!(message.contains("did not complete"));
        assert!(message.contains(&runtime_id.to_string()));
    }
}
