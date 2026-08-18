use super::{Runtime, RuntimeError};
use crate::agent::constants::DEFAULT_CHANNEL_BUFFER;
use crate::utils::{BoxEventStream, receiver_into_stream};
use crate::{
    actor::{AnyActor, Transport},
    error::Error,
};
use async_trait::async_trait;
use autoagents_protocol::{Event, InternalEvent, RuntimeID};
use futures_util::StreamExt;
use log::{debug, error, info, warn};
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU8, Ordering},
    },
};
use tokio::sync::{Mutex, Notify, RwLock, broadcast, mpsc, watch};
use tokio_stream::wrappers::{BroadcastStream, errors::BroadcastStreamRecvError};
use uuid::Uuid;

const DEFAULT_INTERNAL_BUFFER: usize = 1000;

/// Lifecycle states used to make the `run()`/`stop()` startup decision with a
/// single atomic operation, so "who happened first" is never resolved by a
/// racy check-then-act. Shutdown *completion* is tracked separately by
/// `shutdown_complete_tx`; these states only describe the startup transition.
mod lifecycle {
    /// `run()` has not started and no `stop()` has claimed the runtime yet.
    pub(super) const NOT_STARTED: u8 = 0;
    /// `run()` has started; an event loop exists to acknowledge shutdown.
    pub(super) const RUNNING: u8 = 1;
    /// `stop()` won the startup race before `run()`: the Shutdown request is
    /// queued for a future `run()`, and there is no loop to wait on.
    pub(super) const STOP_BEFORE_RUN: u8 = 2;
}

/// Publishes shutdown completion when dropped, so a `stop()` caller waiting on
/// the completion channel is released however the runtime task stops running:
/// normal exit, an early `?` error, a panic unwind, or task cancellation (the
/// future being dropped). It is created by `run()` in the same synchronous step
/// that publishes `RUNNING`, so it covers cancellation even before the event
/// loop starts. Without this, an aborted or panicking runtime task would skip
/// the completion signal, and because the `watch::Sender` lives on in the
/// shared runtime, `stop()` could neither observe completion nor see the
/// channel close — it would wait forever.
struct CompletionGuard<'a> {
    tx: &'a watch::Sender<bool>,
}

impl Drop for CompletionGuard<'_> {
    fn drop(&mut self) {
        // `send_replace` never fails and works even with no receivers, so it is
        // safe to run while unwinding or during cancellation.
        self.tx.send_replace(true);
    }
}

/// Topic subscription entry storing type information and actor references
#[derive(Debug)]
struct Subscription {
    topic_type: TypeId,
    actors: Vec<Arc<dyn AnyActor>>,
}

#[derive(Debug)]
/// Single-threaded runtime implementation with internal event routing
pub struct SingleThreadedRuntime {
    pub id: RuntimeID,
    // External event channel for application consumption
    external_tx: mpsc::Sender<Event>,
    external_rx: Mutex<Option<mpsc::Receiver<Event>>>,
    // Broadcast event channel for multi-subscriber consumption
    broadcast_tx: broadcast::Sender<Event>,
    // Internal event channel for runtime processing
    internal_tx: mpsc::Sender<InternalEvent>,
    internal_rx: Mutex<Option<mpsc::Receiver<InternalEvent>>>,
    // Subscriptions map: topic_name -> Subscription
    subscriptions: Arc<RwLock<HashMap<String, Subscription>>>,
    // Transport layer for message delivery
    transport: Arc<dyn Transport>,
    // Runtime state
    // Startup lifecycle state (see `lifecycle`); drives the race-free
    // decision between `run()` and `stop()`.
    lifecycle: Arc<AtomicU8>,
    shutdown_flag: Arc<AtomicBool>,
    shutdown_notify: Arc<Notify>,
    // Published once, after the event loop has drained and exited, so that
    // `stop()` can await *actual* completion instead of an arbitrary sleep.
    shutdown_complete_tx: watch::Sender<bool>,
}

impl SingleThreadedRuntime {
    pub fn new(channel_buffer: Option<usize>) -> Arc<Self> {
        Self::with_transport(channel_buffer, Arc::new(crate::actor::LocalTransport))
    }

    pub fn with_transport(
        channel_buffer: Option<usize>,
        transport: Arc<dyn Transport>,
    ) -> Arc<Self> {
        let id = Uuid::new_v4();
        let buffer_size = channel_buffer.unwrap_or(DEFAULT_CHANNEL_BUFFER);

        // Create channels
        let (external_tx, external_rx) = mpsc::channel(buffer_size);
        let (internal_tx, internal_rx) = mpsc::channel(DEFAULT_INTERNAL_BUFFER);
        let (broadcast_tx, _) = broadcast::channel(buffer_size);
        let (shutdown_complete_tx, _) = watch::channel(false);

        Arc::new(Self {
            id,
            external_tx,
            external_rx: Mutex::new(Some(external_rx)),
            broadcast_tx,
            internal_tx,
            internal_rx: Mutex::new(Some(internal_rx)),
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
            transport,
            lifecycle: Arc::new(AtomicU8::new(lifecycle::NOT_STARTED)),
            shutdown_flag: Arc::new(AtomicBool::new(false)),
            shutdown_notify: Arc::new(Notify::new()),
            shutdown_complete_tx,
        })
    }

    /// Process internal events in the runtime
    async fn process_internal_event(&self, event: InternalEvent) -> Result<(), Error> {
        debug!("Received internal event: {event:?}");
        match event {
            InternalEvent::ProtocolEvent(event) => {
                self.process_protocol_event(*event).await?;
            }
            InternalEvent::Shutdown => {
                self.shutdown_flag.store(true, Ordering::SeqCst);
                self.shutdown_notify.notify_waiters();
            }
        }
        Ok(())
    }

    /// Forward protocol events to external channel
    async fn process_protocol_event(&self, event: Event) -> Result<(), Error> {
        if let Event::PublishMessage {
            topic_type,
            topic_name,
            message,
        } = event
        {
            self.handle_publish_message(&topic_name, topic_type, message)
                .await?;
        } else {
            //Other protocol events are sent to external
            let _ = self.broadcast_tx.send(event.clone());
            self.external_tx
                .send(event)
                .await
                .map_err(|e| RuntimeError::EventError(Box::new(e)))?;
        }
        Ok(())
    }

    /// Handle message publishing to topic subscribers
    async fn handle_publish_message(
        &self,
        topic_name: &str,
        topic_type: TypeId,
        message: Arc<dyn Any + Send + Sync>,
    ) -> Result<(), RuntimeError> {
        debug!("Handling publish event: {topic_name}");

        let subscriptions = self.subscriptions.read().await;

        if let Some(subscription) = subscriptions.get(topic_name) {
            // Verify type safety
            if subscription.topic_type != topic_type {
                error!(
                    "Type mismatch for topic '{}': expected {:?}, got {:?}",
                    topic_name, subscription.topic_type, topic_type
                );
                return Err(RuntimeError::TopicTypeMismatch(
                    topic_name.to_owned(),
                    topic_type,
                ));
            }

            // Send to all subscribed actors sequentially to maintain strict ordering
            for actor in &subscription.actors {
                if let Err(e) = self
                    .transport
                    .send(actor.as_ref(), Arc::clone(&message))
                    .await
                {
                    error!("Failed to send message to subscriber: {e}");
                }
            }
        } else {
            debug!("No subscribers for topic: {topic_name}");
        }

        Ok(())
    }

    /// Handle actor subscription to a topic
    async fn handle_subscribe(
        &self,
        topic_name: &str,
        topic_type: TypeId,
        actor: Arc<dyn AnyActor>,
    ) -> Result<(), RuntimeError> {
        info!("Actor subscribing to topic: {topic_name}");

        let mut subscriptions = self.subscriptions.write().await;

        match subscriptions.get_mut(topic_name) {
            Some(subscription) => {
                // Verify type consistency
                if subscription.topic_type != topic_type {
                    return Err(RuntimeError::TopicTypeMismatch(
                        topic_name.to_string(),
                        subscription.topic_type,
                    ));
                }
                subscription.actors.push(actor);
            }
            None => {
                // Create new subscription
                subscriptions.insert(
                    topic_name.to_string(),
                    Subscription {
                        topic_type,
                        actors: vec![actor],
                    },
                );
            }
        }

        Ok(())
    }

    /// Start the internal event processing loop
    async fn event_loop(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut internal_rx = self
            .internal_rx
            .lock()
            .await
            .take()
            .ok_or("Internal receiver already taken")?;

        info!("Runtime event loop starting");

        // Records the first error that terminates normal event processing. It is
        // preserved across the drain/cleanup below and returned once the loop
        // has finished, so a real processing failure is never masked by the
        // `Ok(())` epilogue. A graceful `Shutdown` leaves this `None`.
        let mut terminal_error: Option<Error> = None;

        loop {
            tokio::select! {
                // Process internal events
                Some(event) = internal_rx.recv() => {
                    debug!("Processing internal event");

                    // Check for shutdown event first
                    if matches!(event, InternalEvent::Shutdown) {
                        info!("Received shutdown event");
                        self.process_internal_event(event).await?;
                        break;
                    }

                    if let Err(e) = self.process_internal_event(event).await {
                        error!("Error processing internal event: {e}");
                        terminal_error = Some(e);
                        break;
                    }
                }
                // Check for shutdown notification
                _ = self.shutdown_notify.notified() => {
                    if self.shutdown_flag.load(Ordering::SeqCst) {
                        info!("Runtime received shutdown notification");
                        break;
                    }
                }
                // Handle channel closure
                else => {
                    warn!("Internal event channel closed");
                    break;
                }
            }
        }

        // Drain remaining events regardless of why the loop exited. A drain
        // failure is only logged, never returned: the first terminal error above
        // must win, and a graceful shutdown must not be turned into an error by a
        // late drain failure.
        info!("Draining remaining events before shutdown");
        while let Ok(event) = internal_rx.try_recv() {
            if let Err(e) = self.process_internal_event(event).await {
                error!("Error processing event during shutdown: {e}");
            }
        }

        // Completion is published by the `run()` completion guard once this
        // returns, i.e. after the drain above has finished.
        info!("Runtime event loop stopped");
        if let Some(error) = terminal_error {
            return Err(error.into());
        }
        Ok(())
    }
}

#[async_trait]
impl Runtime for SingleThreadedRuntime {
    fn id(&self) -> RuntimeID {
        self.id
    }

    async fn subscribe_any(
        &self,
        topic_name: &str,
        topic_type: TypeId,
        actor: Arc<dyn AnyActor>,
    ) -> Result<(), RuntimeError> {
        self.handle_subscribe(topic_name, topic_type, actor).await
    }

    async fn publish_any(
        &self,
        topic_name: &str,
        topic_type: TypeId,
        message: Arc<dyn Any + Send + Sync>,
    ) -> Result<(), RuntimeError> {
        self.handle_publish_message(topic_name, topic_type, message)
            .await
    }

    fn tx(&self) -> mpsc::Sender<Event> {
        // Create an intercepting sender that routes events through internal processing
        let internal_tx = self.internal_tx.clone();
        let (interceptor_tx, mut interceptor_rx) = mpsc::channel::<Event>(DEFAULT_CHANNEL_BUFFER);

        tokio::spawn(async move {
            while let Some(event) = interceptor_rx.recv().await {
                if let Err(e) = internal_tx
                    .send(InternalEvent::ProtocolEvent(Box::new(event)))
                    .await
                {
                    error!("Failed to forward event to internal channel: {e}");
                    break;
                }
            }
        });

        interceptor_tx
    }

    async fn transport(&self) -> Arc<dyn Transport> {
        Arc::clone(&self.transport)
    }

    async fn take_event_receiver(&self) -> Option<BoxEventStream<Event>> {
        let mut guard = self.external_rx.lock().await;
        guard.take().map(receiver_into_stream)
    }

    async fn subscribe_events(&self) -> BoxEventStream<Event> {
        let rx = self.broadcast_tx.subscribe();
        let stream = BroadcastStream::new(rx)
            .filter_map(|item: Result<Event, BroadcastStreamRecvError>| async move { item.ok() });
        Box::pin(stream)
    }

    async fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Publish `RUNNING` and install the completion guard as one synchronous
        // step: there is no `.await` between the swap and the guard binding, so
        // this task cannot be cancelled in between. That guarantees any `stop()`
        // which observes `RUNNING` is matched by a guard that will publish
        // completion on every exit path — including cancellation before the
        // event loop starts. `swap` also gates re-entry: a second `run()` finds
        // the state already `RUNNING`, claims nothing, and returns without a
        // guard, so it can never falsely signal completion for the live loop.
        //
        // If a `stop()` already queued a Shutdown before the loop existed
        // (`STOP_BEFORE_RUN`), that request stays buffered on the internal
        // channel and the loop below consumes it and exits immediately.
        if self.lifecycle.swap(lifecycle::RUNNING, Ordering::SeqCst) == lifecycle::RUNNING {
            return Err("Runtime event loop is already running".into());
        }
        let _completion = CompletionGuard {
            tx: &self.shutdown_complete_tx,
        };

        info!("Starting SingleThreadedRuntime {}", self.id);
        self.event_loop().await
    }

    async fn stop(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut shutdown_complete_rx = self.shutdown_complete_tx.subscribe();

        // Shutdown has already completed.
        if *shutdown_complete_rx.borrow() {
            return Ok(());
        }

        info!("Initiating runtime shutdown for {}", self.id);

        // Send the shutdown request.
        if let Err(e) = self.internal_tx.send(InternalEvent::Shutdown).await {
            // The runtime may have completed between the initial check
            // and sending the shutdown request.
            if *shutdown_complete_rx.borrow() {
                return Ok(());
            }

            return Err(format!("Failed to send shutdown signal: {e}").into());
        }

        // Decide atomically whether an event loop exists to acknowledge
        // completion. Winning this compare-exchange means `run()` has not
        // started, so there is no loop to wait on: the queued Shutdown stays
        // buffered for a future `run()`. Losing it means `run()` is live and
        // will publish completion, so fall through and wait for it. Making this
        // a single atomic operation removes the check-then-act race where
        // `run()` could store its state between a plain load and this branch.
        match self.lifecycle.compare_exchange(
            lifecycle::NOT_STARTED,
            lifecycle::STOP_BEFORE_RUN,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            // We won before `run()` started, or another `stop()` already claimed
            // the pre-run state: either way there is no event loop to wait on.
            Ok(_) | Err(lifecycle::STOP_BEFORE_RUN) => return Ok(()),
            // `RUNNING`: the event loop is live and will publish completion.
            Err(_) => {}
        }

        // Wait until the event loop reports actual shutdown completion.
        loop {
            if *shutdown_complete_rx.borrow_and_update() {
                return Ok(());
            }

            shutdown_complete_rx
                .changed()
                .await
                .map_err(|e| format!("Shutdown completion channel closed: {e}"))?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::{CloneableMessage, Topic};
    use crate::runtime::{RuntimeConfig, TypedRuntime};
    use ractor::{Actor, ActorProcessingErr, ActorRef};
    use tokio::time::{Duration, sleep};

    // Test message types
    #[derive(Clone, Debug)]
    struct TestMessage {
        content: String,
    }

    impl crate::actor::ActorMessage for TestMessage {}
    impl CloneableMessage for TestMessage {}

    // Test actor
    struct TestActor {
        received: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl Actor for TestActor {
        type Msg = TestMessage;
        type State = ();
        type Arguments = Arc<Mutex<Vec<String>>>;

        async fn pre_start(
            &self,
            _myself: ActorRef<Self::Msg>,
            _args: Self::Arguments,
        ) -> Result<Self::State, ActorProcessingErr> {
            Ok(())
        }

        async fn handle(
            &self,
            _myself: ActorRef<Self::Msg>,
            message: Self::Msg,
            _state: &mut Self::State,
        ) -> Result<(), ActorProcessingErr> {
            let mut received = self.received.lock().await;
            received.push(message.content);
            Ok(())
        }
    }

    /// Transport that only counts dispatches, so lifecycle tests can observe
    /// that the event loop processed an event without depending on asynchronous
    /// actor delivery.
    #[derive(Debug)]
    struct CountingTransport {
        delivered: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait]
    impl Transport for CountingTransport {
        async fn send(
            &self,
            _actor: &dyn AnyActor,
            _msg: Arc<dyn Any + Send + Sync>,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            self.delivered.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    /// Build a queued publish event for `topic_name` carrying `content`.
    fn publish_event(topic_name: &str, content: &str) -> InternalEvent {
        InternalEvent::ProtocolEvent(Box::new(Event::PublishMessage {
            topic_name: topic_name.to_string(),
            topic_type: TypeId::of::<TestMessage>(),
            message: Arc::new(TestMessage {
                content: content.to_string(),
            }) as Arc<dyn Any + Send + Sync>,
        }))
    }

    #[tokio::test]
    async fn test_runtime_creation() {
        let runtime = SingleThreadedRuntime::new(None);
        assert_ne!(runtime.id(), Uuid::nil());
    }

    #[tokio::test]
    async fn test_publish_subscribe_cloneable() {
        let runtime = SingleThreadedRuntime::new(Some(10));
        let runtime_handle = runtime.clone();

        // Start runtime in background
        let runtime_task = tokio::spawn(async move { runtime_handle.run().await });

        // Create test actor
        let received = Arc::new(Mutex::new(Vec::new()));
        let (actor_ref, _actor_handle) = Actor::spawn(
            None,
            TestActor {
                received: received.clone(),
            },
            received.clone(),
        )
        .await
        .unwrap();

        // Subscribe to topic
        let topic = Topic::<TestMessage>::new("test_topic");
        runtime.subscribe(&topic, actor_ref).await.unwrap();

        // Publish messages
        runtime
            .publish(
                &topic,
                TestMessage {
                    content: "Hello".to_string(),
                },
            )
            .await
            .unwrap();

        runtime
            .publish(
                &topic,
                TestMessage {
                    content: "World".to_string(),
                },
            )
            .await
            .unwrap();

        // Wait for messages to be processed
        sleep(Duration::from_millis(100)).await;

        // Verify messages were received
        let received_msgs = received.lock().await;
        assert_eq!(received_msgs.len(), 2);
        assert_eq!(received_msgs[0], "Hello");
        assert_eq!(received_msgs[1], "World");

        // Shutdown
        runtime.stop().await.unwrap();
        runtime_task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_type_safety() {
        let runtime = SingleThreadedRuntime::new(None);
        let runtime_handle = runtime.clone();

        // Start runtime in background
        let runtime_task = tokio::spawn(async move { runtime_handle.run().await });

        // Create topic and subscribe with one type
        let topic_name = "typed_topic";
        let topic1 = Topic::<TestMessage>::new(topic_name);

        let received = Arc::new(Mutex::new(Vec::new()));
        let (actor_ref, _) = Actor::spawn(
            None,
            TestActor {
                received: received.clone(),
            },
            received.clone(),
        )
        .await
        .unwrap();

        runtime.subscribe(&topic1, actor_ref).await.unwrap();

        // Wait for subscription to be processed
        sleep(Duration::from_millis(50)).await;

        // Try to subscribe with different type to same topic name - should fail
        #[derive(Clone)]
        struct OtherMessage;
        impl crate::actor::ActorMessage for OtherMessage {}
        impl CloneableMessage for OtherMessage {}

        let topic2 = Topic::<OtherMessage>::new(topic_name);

        struct OtherActor;
        #[async_trait]
        impl Actor for OtherActor {
            type Msg = OtherMessage;
            type State = ();
            type Arguments = ();

            async fn pre_start(
                &self,
                _myself: ActorRef<Self::Msg>,
                _args: Self::Arguments,
            ) -> Result<Self::State, ActorProcessingErr> {
                Ok(())
            }

            async fn handle(
                &self,
                _myself: ActorRef<Self::Msg>,
                _message: Self::Msg,
                _state: &mut Self::State,
            ) -> Result<(), ActorProcessingErr> {
                Ok(())
            }
        }

        let (other_ref, _) = Actor::spawn(None, OtherActor, ()).await.unwrap();

        // This should fail due to type mismatch
        let result = runtime.subscribe(&topic2, other_ref).await;

        // The subscribe method should return an error for type mismatch
        assert!(result.is_err());

        // Verify it's the correct error type
        if let Err(RuntimeError::TopicTypeMismatch(topic, _)) = result {
            assert_eq!(topic, topic_name);
        } else {
            panic!("Expected TopicTypeMismatch error");
        }

        // Shutdown
        runtime.stop().await.unwrap();
        runtime_task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_message_ordering() {
        let runtime = SingleThreadedRuntime::new(Some(10));
        let runtime_handle = runtime.clone();

        // Start runtime in background
        let runtime_task = tokio::spawn(async move { runtime_handle.run().await });

        // Create test actor that tracks message order
        let received = Arc::new(Mutex::new(Vec::new()));
        let (actor_ref, _actor_handle) = Actor::spawn(
            None,
            TestActor {
                received: received.clone(),
            },
            received.clone(),
        )
        .await
        .unwrap();

        // Subscribe to topic
        let topic = Topic::<TestMessage>::new("order_test");
        runtime.subscribe(&topic, actor_ref).await.unwrap();

        // Publish multiple messages rapidly
        for i in 0..10 {
            runtime
                .publish(
                    &topic,
                    TestMessage {
                        content: format!("Message {i}"),
                    },
                )
                .await
                .unwrap();
        }

        // Wait for all messages to be processed
        sleep(Duration::from_millis(200)).await;

        // Verify messages were received in order
        let received_msgs = received.lock().await;
        assert_eq!(received_msgs.len(), 10);

        for (i, msg) in received_msgs.iter().enumerate() {
            assert_eq!(msg, &format!("Message {i}"));
        }

        // Shutdown
        runtime.stop().await.unwrap();
        runtime_task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_runtime_multiple_topics() {
        let runtime = SingleThreadedRuntime::new(Some(10));
        let runtime_handle = runtime.clone();

        // Start runtime in background
        let runtime_task = tokio::spawn(async move { runtime_handle.run().await });

        // Create multiple topics
        let topic1 = Topic::<TestMessage>::new("topic1");
        let topic2 = Topic::<TestMessage>::new("topic2");

        let received1 = Arc::new(Mutex::new(Vec::new()));
        let received2 = Arc::new(Mutex::new(Vec::new()));

        let (actor_ref1, _) = Actor::spawn(
            None,
            TestActor {
                received: received1.clone(),
            },
            received1.clone(),
        )
        .await
        .unwrap();

        let (actor_ref2, _) = Actor::spawn(
            None,
            TestActor {
                received: received2.clone(),
            },
            received2.clone(),
        )
        .await
        .unwrap();

        // Subscribe to different topics
        runtime.subscribe(&topic1, actor_ref1).await.unwrap();
        runtime.subscribe(&topic2, actor_ref2).await.unwrap();
        sleep(Duration::from_millis(50)).await;

        // Publish to topic1
        let message1 = TestMessage {
            content: "topic1_message".to_string(),
        };
        runtime.publish(&topic1, message1).await.unwrap();
        sleep(Duration::from_millis(50)).await;

        // Publish to topic2
        let message2 = TestMessage {
            content: "topic2_message".to_string(),
        };
        runtime.publish(&topic2, message2).await.unwrap();
        sleep(Duration::from_millis(50)).await;

        // Verify messages
        let received_msgs1 = received1.lock().await;
        let received_msgs2 = received2.lock().await;

        assert_eq!(received_msgs1.len(), 1);
        assert_eq!(received_msgs1[0], "topic1_message");

        assert_eq!(received_msgs2.len(), 1);
        assert_eq!(received_msgs2[0], "topic2_message");

        // Shutdown
        runtime.stop().await.unwrap();
        runtime_task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_runtime_subscribe_multiple_actors_same_topic() {
        let runtime = SingleThreadedRuntime::new(Some(10));
        let runtime_handle = runtime.clone();

        // Start runtime in background
        let runtime_task = tokio::spawn(async move { runtime_handle.run().await });

        let topic = Topic::<TestMessage>::new("shared_topic");

        let received1 = Arc::new(Mutex::new(Vec::new()));
        let received2 = Arc::new(Mutex::new(Vec::new()));

        let (actor_ref1, _) = Actor::spawn(
            None,
            TestActor {
                received: received1.clone(),
            },
            received1.clone(),
        )
        .await
        .unwrap();

        let (actor_ref2, _) = Actor::spawn(
            None,
            TestActor {
                received: received2.clone(),
            },
            received2.clone(),
        )
        .await
        .unwrap();

        // Subscribe both actors to same topic
        runtime.subscribe(&topic, actor_ref1).await.unwrap();
        runtime.subscribe(&topic, actor_ref2).await.unwrap();
        sleep(Duration::from_millis(50)).await;

        // Publish message
        let message = TestMessage {
            content: "broadcast_message".to_string(),
        };
        runtime.publish(&topic, message).await.unwrap();
        sleep(Duration::from_millis(100)).await;

        // Both actors should receive the message
        let received_msgs1 = received1.lock().await;
        let received_msgs2 = received2.lock().await;

        assert_eq!(received_msgs1.len(), 1);
        assert_eq!(received_msgs1[0], "broadcast_message");

        assert_eq!(received_msgs2.len(), 1);
        assert_eq!(received_msgs2[0], "broadcast_message");

        // Shutdown
        runtime.stop().await.unwrap();
        runtime_task.await.unwrap().unwrap();
    }

    #[test]
    fn test_runtime_config_creation() {
        let config = RuntimeConfig {
            queue_size: Some(100),
        };
        assert_eq!(config.queue_size, Some(100));
    }

    #[test]
    fn test_runtime_id_generation() {
        let runtime1 = SingleThreadedRuntime::new(None);
        let runtime2 = SingleThreadedRuntime::new(None);

        assert_ne!(runtime1.id(), runtime2.id());
    }

    #[tokio::test]
    async fn test_stop_waits_for_shutdown_completion() {
        let runtime = SingleThreadedRuntime::new(None);
        let runtime_handle = runtime.clone();

        let runtime_task = tokio::spawn(async move { runtime_handle.run().await });

        // Ensure run() has started before requesting shutdown.
        while runtime.lifecycle.load(Ordering::SeqCst) != lifecycle::RUNNING {
            tokio::task::yield_now().await;
        }

        runtime.stop().await.unwrap();

        assert!(
            *runtime.shutdown_complete_tx.borrow(),
            "stop() should only return after shutdown completion is acknowledged"
        );

        runtime_task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_stop_before_run_does_not_hang() {
        let runtime = SingleThreadedRuntime::new(None);

        tokio::time::timeout(Duration::from_secs(1), runtime.stop())
            .await
            .expect("stop() should not hang before run()")
            .expect("stop() should succeed");

        let runtime_handle = runtime.clone();

        let runtime_task = tokio::spawn(async move { runtime_handle.run().await });

        tokio::time::timeout(Duration::from_secs(1), runtime_task)
            .await
            .expect("runtime should process the queued shutdown request")
            .expect("runtime task should not panic")
            .expect("runtime should shut down successfully");

        assert!(
            *runtime.shutdown_complete_tx.borrow(),
            "shutdown should be completed after the runtime processes the queued request"
        );
    }

    #[tokio::test]
    async fn test_repeated_stop_is_safe() {
        let runtime = SingleThreadedRuntime::new(None);
        let runtime_handle = runtime.clone();

        let runtime_task = tokio::spawn(async move { runtime_handle.run().await });

        while runtime.lifecycle.load(Ordering::SeqCst) != lifecycle::RUNNING {
            tokio::task::yield_now().await;
        }

        runtime.stop().await.unwrap();
        runtime.stop().await.unwrap();

        runtime_task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_concurrent_stop_is_safe() {
        let runtime = SingleThreadedRuntime::new(None);
        let runtime_handle = runtime.clone();

        let runtime_task = tokio::spawn(async move { runtime_handle.run().await });

        while runtime.lifecycle.load(Ordering::SeqCst) != lifecycle::RUNNING {
            tokio::task::yield_now().await;
        }

        let (first, second, third) = tokio::join!(runtime.stop(), runtime.stop(), runtime.stop(),);

        first.unwrap();
        second.unwrap();
        third.unwrap();

        runtime_task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_run_stop_startup_race() {
        // Regression test for the run()/stop() startup race. `run()` and `stop()`
        // are started concurrently and neither waits for the other, so the
        // atomic "who happened first" decision is exercised under real
        // contention. Repeat to widen scheduling coverage while staying
        // deterministic; an outer timeout turns any lost shutdown or deadlock
        // into a failure instead of a hung suite.
        for _ in 0..100 {
            let runtime = SingleThreadedRuntime::new(None);
            let run_handle = runtime.clone();
            let stop_handle = runtime.clone();

            let run_task = tokio::spawn(async move { run_handle.run().await });
            let stop_task = tokio::spawn(async move { stop_handle.stop().await });

            tokio::time::timeout(Duration::from_secs(5), stop_task)
                .await
                .expect("stop() must not hang during the startup race")
                .expect("stop() task must not panic")
                .expect("stop() should succeed");

            // Whoever won the race, the shutdown request is never lost: run()
            // consumes it and exits on its own rather than being left running.
            tokio::time::timeout(Duration::from_secs(5), run_task)
                .await
                .expect("run() must not be left running after stop()")
                .expect("run() task must not panic")
                .expect("run() should exit cleanly");

            assert!(
                *runtime.shutdown_complete_tx.borrow(),
                "shutdown completion must be published once the race resolves"
            );
        }
    }

    #[tokio::test]
    async fn test_completion_published_after_drain() {
        let delivered = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let runtime = SingleThreadedRuntime::with_transport(
            None,
            Arc::new(CountingTransport {
                delivered: delivered.clone(),
            }),
        );

        // Subscribe an actor so the topic has a delivery target.
        let received = Arc::new(Mutex::new(Vec::new()));
        let (actor_ref, _handle) = Actor::spawn(
            None,
            TestActor {
                received: received.clone(),
            },
            received.clone(),
        )
        .await
        .unwrap();
        let topic = Topic::<TestMessage>::new("drain_topic");
        runtime.subscribe(&topic, actor_ref).await.unwrap();

        // Queue an event, then Shutdown, then two more events. The loop processes
        // the first event and breaks on Shutdown; the trailing two can only be
        // handled by the post-shutdown drain path.
        runtime
            .internal_tx
            .send(publish_event("drain_topic", "first"))
            .await
            .unwrap();
        runtime
            .internal_tx
            .send(InternalEvent::Shutdown)
            .await
            .unwrap();
        runtime
            .internal_tx
            .send(publish_event("drain_topic", "second"))
            .await
            .unwrap();
        runtime
            .internal_tx
            .send(publish_event("drain_topic", "third"))
            .await
            .unwrap();

        runtime.run().await.unwrap();

        // run() only returns after the drain loop, and completion is published
        // last, so both facts must hold together: every queued event was
        // dispatched, and only then was completion signalled.
        assert_eq!(
            delivered.load(Ordering::SeqCst),
            3,
            "all queued events, including those drained after shutdown, must be processed"
        );
        assert!(
            *runtime.shutdown_complete_tx.borrow(),
            "completion must be published once draining has finished"
        );
    }

    #[tokio::test]
    async fn test_processing_error_propagates_from_run() {
        // Regression test for the bug where a terminal event-processing failure
        // was swallowed and `run()` returned `Ok(())`. Dropping the external
        // receiver closes the external channel, so forwarding a non-publish
        // protocol event fails inside `process_protocol_event`, which is the
        // real production path that terminates the event loop.
        let runtime = SingleThreadedRuntime::new(None);
        drop(runtime.take_event_receiver().await);

        runtime
            .internal_tx
            .send(InternalEvent::ProtocolEvent(Box::new(Event::SendMessage {
                message: "boom".to_string(),
                actor_id: Uuid::new_v4(),
            })))
            .await
            .expect("queuing the terminal event should succeed");

        let result = runtime.run().await;
        assert!(
            result.is_err(),
            "run() must surface a terminal event-processing failure"
        );
        let message = result.unwrap_err().to_string();
        assert!(
            message.contains("Event error"),
            "expected the forwarding failure to be preserved, got: {message}"
        );

        // The completion guard must still publish terminal state on the error
        // path so a waiting `stop()` is released.
        assert!(
            *runtime.shutdown_complete_tx.borrow(),
            "completion must be published even when run() returns an error"
        );
    }

    #[tokio::test]
    async fn test_graceful_shutdown_returns_ok() {
        // A normal shutdown must remain a success: it must not be turned into an
        // error by the new terminal-error propagation.
        let runtime = SingleThreadedRuntime::new(None);
        let runtime_handle = runtime.clone();
        let runtime_task = tokio::spawn(async move { runtime_handle.run().await });

        while runtime.lifecycle.load(Ordering::SeqCst) != lifecycle::RUNNING {
            tokio::task::yield_now().await;
        }

        runtime.stop().await.expect("graceful stop should succeed");
        runtime_task
            .await
            .expect("runtime task should not panic")
            .expect("graceful shutdown must return Ok(())");
    }

    #[tokio::test]
    async fn test_drain_failure_does_not_replace_original_error() {
        // First terminal error wins: a failure while draining must not overwrite
        // the error that originally terminated normal processing.
        let runtime = SingleThreadedRuntime::new(None);

        // Register a subscription so a type-mismatched publish yields a
        // deterministic TopicTypeMismatch as the first terminal error.
        let received = Arc::new(Mutex::new(Vec::new()));
        let (actor_ref, _handle) = Actor::spawn(
            None,
            TestActor {
                received: received.clone(),
            },
            received.clone(),
        )
        .await
        .expect("spawning the test actor should succeed");
        let topic = Topic::<TestMessage>::new("drain_precedence");
        runtime
            .subscribe(&topic, actor_ref)
            .await
            .expect("subscribing the test actor should succeed");

        // Close the external channel so the drained event fails too.
        drop(runtime.take_event_receiver().await);

        // First event: type-mismatched publish -> TopicTypeMismatch (terminal).
        runtime
            .internal_tx
            .send(InternalEvent::ProtocolEvent(Box::new(
                Event::PublishMessage {
                    topic_name: "drain_precedence".to_string(),
                    topic_type: TypeId::of::<u8>(),
                    message: Arc::new(0u8) as Arc<dyn Any + Send + Sync>,
                },
            )))
            .await
            .expect("queuing the type-mismatched event should succeed");
        // Second event: routed to the closed external channel -> EventError,
        // encountered only during the post-break drain.
        runtime
            .internal_tx
            .send(InternalEvent::ProtocolEvent(Box::new(Event::SendMessage {
                message: "drain".to_string(),
                actor_id: Uuid::new_v4(),
            })))
            .await
            .expect("queuing the drain event should succeed");

        let err = runtime
            .run()
            .await
            .expect_err("the terminal processing failure must propagate");
        let message = err.to_string();
        assert!(
            message.contains("TopicTypeMismatch"),
            "the first terminal error must win over a later drain failure, got: {message}"
        );
    }

    #[tokio::test]
    async fn test_completion_published_when_run_task_cancelled() {
        // Regression test: once run() records RUNNING, a stop() caller waits on
        // the completion channel. If the runtime task is cancelled (or panics)
        // the event loop's normal epilogue never runs, yet the completion guard
        // must still fire on drop so the waiter is released instead of hanging
        // forever on a `watch::Sender` that the shared runtime keeps alive.
        let delivered = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let runtime = SingleThreadedRuntime::with_transport(
            None,
            Arc::new(CountingTransport {
                delivered: delivered.clone(),
            }),
        );

        let received = Arc::new(Mutex::new(Vec::new()));
        let (actor_ref, _handle) = Actor::spawn(
            None,
            TestActor {
                received: received.clone(),
            },
            received.clone(),
        )
        .await
        .unwrap();
        let topic = Topic::<TestMessage>::new("cancel_topic");
        runtime.subscribe(&topic, actor_ref).await.unwrap();

        let run_handle = runtime.clone();
        let runtime_task = tokio::spawn(async move { run_handle.run().await });

        // Drive one event through so the event loop is provably running with its
        // completion guard installed (the counter only advances inside the loop).
        runtime
            .internal_tx
            .send(publish_event("cancel_topic", "warmup"))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while delivered.load(Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the event loop should process the warmup event");

        // Cancel the task without requesting shutdown: only the completion guard
        // (run on drop) can release a waiter now.
        runtime_task.abort();
        let join = runtime_task.await;
        assert!(
            join.is_err(),
            "the aborted run task should report cancellation"
        );

        // Completion was published on drop, so stop() returns promptly instead of
        // waiting forever.
        tokio::time::timeout(Duration::from_secs(5), runtime.stop())
            .await
            .expect("stop() must not hang after the run task was cancelled")
            .expect("stop() should succeed");
        assert!(
            *runtime.shutdown_complete_tx.borrow(),
            "completion must be published even when the run task is cancelled"
        );
    }
}
