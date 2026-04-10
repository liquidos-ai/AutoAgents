use super::{Runtime, RuntimeError};
use autoagents_protocol::RuntimeID;
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
    }

    struct TestRuntime {
        id: RuntimeID,
        run_behavior: Behavior,
        stop_behavior: Behavior,
        run_calls: Arc<AtomicUsize>,
        stop_calls: Arc<AtomicUsize>,
        run_started: Option<Arc<Notify>>,
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
                run_started: None,
                tx,
            }
        }

        fn with_run_notify(mut self, run_started: Arc<Notify>) -> Self {
            self.run_started = Some(run_started);
            self
        }

        fn run_calls(&self) -> Arc<AtomicUsize> {
            Arc::clone(&self.run_calls)
        }

        fn stop_calls(&self) -> Arc<AtomicUsize> {
            Arc::clone(&self.stop_calls)
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
            }
        }

        async fn stop(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            self.stop_calls.fetch_add(1, Ordering::SeqCst);

            match self.stop_behavior {
                Behavior::Success => Ok(()),
                Behavior::Error => Err(std::io::Error::other("stop failed").into()),
                Behavior::Panic => panic!("runtime stop panic"),
            }
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
}
