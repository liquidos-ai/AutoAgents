use super::{Runtime, RuntimeError};
use crate::protocol::InternalEvent;
use crate::{
    actor::{AnyActor, Transport},
    error::Error,
    protocol::{Event, RuntimeID},
};
use async_trait::async_trait;
use log::{debug, error, info, warn};
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
use tokio::sync::{mpsc, Mutex, Notify, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

const DEFAULT_CHANNEL_BUFFER: usize = 100;
const DEFAULT_INTERNAL_BUFFER: usize = 1000;

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
    // Internal event channel for runtime processing
    internal_tx: mpsc::Sender<InternalEvent>,
    internal_rx: Mutex<Option<mpsc::Receiver<InternalEvent>>>,
    // Subscriptions map: topic_name -> Subscription
    subscriptions: Arc<RwLock<HashMap<String, Subscription>>>,
    // Transport layer for message delivery
    transport: Arc<dyn Transport>,
    // Runtime state
    shutdown_flag: Arc<AtomicBool>,
    shutdown_notify: Arc<Notify>,
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

        Arc::new(Self {
            id,
            external_tx,
            external_rx: Mutex::new(Some(external_rx)),
            internal_tx,
            internal_rx: Mutex::new(Some(internal_rx)),
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
            transport,
            shutdown_flag: Arc::new(AtomicBool::new(false)),
            shutdown_notify: Arc::new(Notify::new()),
        })
    }

    /// Process internal events in the runtime
    async fn process_internal_event(&self, event: InternalEvent) -> Result<(), Error> {
        debug!("Received internal event: {event:?}");
        match event {
            InternalEvent::ProtocolEvent(event) => {
                self.process_protocol_event(event).await?;
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
        match event {
            Event::PublishMessage {
                topic_type,
                topic_name,
                message,
            } => {
                self.handle_publish_message(&topic_name, topic_type, message)
                    .await?;
            }
            _ => {
                //Other protocol events are sent to external
                self.external_tx
                    .send(event)
                    .await
                    .map_err(RuntimeError::EventError)?;
            }
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

        // Drain remaining events
        info!("Draining remaining events before shutdown");
        while let Ok(event) = internal_rx.try_recv() {
            if let Err(e) = self.process_internal_event(event).await {
                error!("Error processing event during shutdown: {e}");
            }
        }

        info!("Runtime event loop stopped");
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

    async fn tx(&self) -> mpsc::Sender<Event> {
        // Create an intercepting sender that routes events through internal processing
        let internal_tx = self.internal_tx.clone();
        let (interceptor_tx, mut interceptor_rx) = mpsc::channel::<Event>(DEFAULT_CHANNEL_BUFFER);

        tokio::spawn(async move {
            while let Some(event) = interceptor_rx.recv().await {
                if let Err(e) = internal_tx.send(InternalEvent::ProtocolEvent(event)).await {
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

    async fn take_event_receiver(&self) -> Option<ReceiverStream<Event>> {
        self.external_rx
            .lock()
            .await
            .take()
            .map(ReceiverStream::new)
    }

    async fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Starting SingleThreadedRuntime {}", self.id);
        self.event_loop().await
    }

    async fn stop(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Initiating runtime shutdown for {}", self.id);

        // Send shutdown signal
        self.internal_tx
            .send(InternalEvent::Shutdown)
            .await
            .map_err(|e| format!("Failed to send shutdown signal: {e}"))?;

        // Wait a brief moment for shutdown to complete
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::{CloneableMessage, SharedMessage, Topic};
    use crate::runtime::{RuntimeConfig, TypedRuntime};
    use ractor::{Actor, ActorProcessingErr, ActorRef};
    use tokio::time::{sleep, Duration};

    // Test message types
    #[derive(Clone, Debug)]
    #[cfg_attr(feature = "cluster", derive(serde::Serialize, serde::Deserialize))]
    struct TestMessage {
        content: String,
    }

    #[cfg(feature = "cluster")]
    impl ractor::BytesConvertable for TestMessage {
        fn into_bytes(self) -> Vec<u8> {
            serde_json::to_vec(&self).expect("Failed to serialize TestMessage")
        }

        fn from_bytes(data: Vec<u8>) -> Self {
            serde_json::from_slice(&data).expect("Failed to deserialize TestMessage")
        }
    }

    impl crate::actor::ActorMessage for TestMessage {}
    impl CloneableMessage for TestMessage {}

    #[derive(Debug)]
    #[cfg_attr(feature = "cluster", derive(serde::Serialize, serde::Deserialize))]
    struct SharedTestMessage {
        content: String,
    }

    #[cfg(feature = "cluster")]
    impl ractor::BytesConvertable for SharedTestMessage {
        fn into_bytes(self) -> Vec<u8> {
            serde_json::to_vec(&self).expect("Failed to serialize SharedTestMessage")
        }

        fn from_bytes(data: Vec<u8>) -> Self {
            serde_json::from_slice(&data).expect("Failed to deserialize SharedTestMessage")
        }
    }

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
        runtime_task.abort();
    }

    #[tokio::test]
    async fn test_publish_subscribe_shared() {
        let runtime = SingleThreadedRuntime::new(Some(10));
        let runtime_handle = runtime.clone();

        // Start runtime in background
        let runtime_task = tokio::spawn(async move { runtime_handle.run().await });

        // Create test actor for shared messages
        struct SharedActor {
            received: Arc<Mutex<Vec<String>>>,
        }

        #[async_trait]
        impl Actor for SharedActor {
            type Msg = SharedMessage<SharedTestMessage>;
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
                received.push(message.inner().content.clone());
                Ok(())
            }
        }

        let received = Arc::new(Mutex::new(Vec::new()));
        let (actor_ref, _actor_handle) = Actor::spawn(
            None,
            SharedActor {
                received: received.clone(),
            },
            received.clone(),
        )
        .await
        .unwrap();

        // Subscribe to shared topic
        let topic = Topic::<SharedMessage<SharedTestMessage>>::new("shared_topic");
        runtime.subscribe_shared(&topic, actor_ref).await.unwrap();

        // Publish shared messages
        runtime
            .publish_shared(
                &topic,
                SharedTestMessage {
                    content: "Shared1".to_string(),
                },
            )
            .await
            .unwrap();

        runtime
            .publish_shared(
                &topic,
                SharedTestMessage {
                    content: "Shared2".to_string(),
                },
            )
            .await
            .unwrap();

        // Wait for messages to be processed
        sleep(Duration::from_millis(100)).await;

        // Verify messages were received
        let received_msgs = received.lock().await;
        assert_eq!(received_msgs.len(), 2);
        assert_eq!(received_msgs[0], "Shared1");
        assert_eq!(received_msgs[1], "Shared2");

        // Shutdown
        runtime.stop().await.unwrap();
        runtime_task.abort();
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
        #[cfg_attr(feature = "cluster", derive(serde::Serialize, serde::Deserialize))]
        struct OtherMessage;

        #[cfg(feature = "cluster")]
        impl ractor::BytesConvertable for OtherMessage {
            fn into_bytes(self) -> Vec<u8> {
                Vec::new() // Empty message
            }

            fn from_bytes(_data: Vec<u8>) -> Self {
                OtherMessage
            }
        }

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
        runtime_task.abort();
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
        runtime_task.abort();
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
        runtime_task.abort();
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
        runtime_task.abort();
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
}
