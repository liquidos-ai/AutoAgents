use crate::actor::AnyActor;
use async_trait::async_trait;
use std::any::Any;
use std::sync::Arc;

#[async_trait]
pub trait Transport: Send + Sync {
    async fn send(
        &self,
        actor: &dyn AnyActor,
        msg: Arc<dyn Any + Send + Sync>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

#[derive(Debug)]
pub struct LocalTransport;

#[async_trait]
impl Transport for LocalTransport {
    async fn send(
        &self,
        actor: &dyn AnyActor,
        msg: Arc<dyn Any + Send + Sync>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        actor.send_any(msg).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::{ActorMessage, CloneableMessage};
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};

    // Test message types
    #[derive(Debug, Clone, PartialEq)]
    struct TestMessage {
        content: String,
    }
    impl ActorMessage for TestMessage {}
    impl CloneableMessage for TestMessage {}

    // Mock actor for testing
    struct MockActor {
        received_messages: Arc<Mutex<Vec<String>>>,
        should_fail: bool,
    }

    impl MockActor {
        fn new() -> Self {
            Self {
                received_messages: Arc::new(Mutex::new(Vec::new())),
                should_fail: false,
            }
        }

        fn with_failure() -> Self {
            Self {
                received_messages: Arc::new(Mutex::new(Vec::new())),
                should_fail: true,
            }
        }

        fn received_messages(&self) -> Vec<String> {
            self.received_messages.lock().unwrap().clone()
        }
    }

    impl std::fmt::Debug for MockActor {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MockActor")
                .field("should_fail", &self.should_fail)
                .finish()
        }
    }

    #[async_trait]
    impl AnyActor for MockActor {
        async fn send_any(
            &self,
            msg: Arc<dyn Any + Send + Sync>,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            if self.should_fail {
                return Err("Mock actor failure".into());
            }

            if let Some(test_msg) = msg.downcast_ref::<TestMessage>() {
                self.received_messages
                    .lock()
                    .unwrap()
                    .push(test_msg.content.clone());
                Ok(())
            } else {
                Err("Type mismatch in mock actor".into())
            }
        }
    }

    #[tokio::test]
    async fn test_local_transport_successful_send() {
        let transport = LocalTransport;
        let actor = MockActor::new();
        let msg = TestMessage {
            content: "test_message".to_string(),
        };
        let arc_msg: Arc<dyn Any + Send + Sync> = Arc::new(msg);

        let result = transport.send(&actor, arc_msg).await;

        assert!(result.is_ok());
        assert_eq!(actor.received_messages(), vec!["test_message"]);
    }

    #[tokio::test]
    async fn test_local_transport_failed_send() {
        let transport = LocalTransport;
        let actor = MockActor::with_failure();
        let msg = TestMessage {
            content: "failing_message".to_string(),
        };
        let arc_msg: Arc<dyn Any + Send + Sync> = Arc::new(msg);

        let result = transport.send(&actor, arc_msg).await;

        assert!(result.is_err());
        assert_eq!(actor.received_messages().len(), 0);
    }

    #[tokio::test]
    async fn test_local_transport_multiple_sends() {
        let transport = LocalTransport;
        let actor = MockActor::new();

        let msg1 = TestMessage {
            content: "message_1".to_string(),
        };
        let msg2 = TestMessage {
            content: "message_2".to_string(),
        };
        let msg3 = TestMessage {
            content: "message_3".to_string(),
        };

        let arc_msg1: Arc<dyn Any + Send + Sync> = Arc::new(msg1);
        let arc_msg2: Arc<dyn Any + Send + Sync> = Arc::new(msg2);
        let arc_msg3: Arc<dyn Any + Send + Sync> = Arc::new(msg3);

        let result1 = transport.send(&actor, arc_msg1).await;
        let result2 = transport.send(&actor, arc_msg2).await;
        let result3 = transport.send(&actor, arc_msg3).await;

        assert!(result1.is_ok());
        assert!(result2.is_ok());
        assert!(result3.is_ok());

        let received = actor.received_messages();
        assert_eq!(received.len(), 3);
        assert_eq!(received[0], "message_1");
        assert_eq!(received[1], "message_2");
        assert_eq!(received[2], "message_3");
    }

    #[tokio::test]
    async fn test_transport_trait_object() {
        let transport: Box<dyn Transport> = Box::new(LocalTransport);
        let actor = MockActor::new();
        let msg = TestMessage {
            content: "trait_object_test".to_string(),
        };
        let arc_msg: Arc<dyn Any + Send + Sync> = Arc::new(msg);

        let result = transport.send(&actor, arc_msg).await;

        assert!(result.is_ok());
        assert_eq!(actor.received_messages(), vec!["trait_object_test"]);
    }

    #[tokio::test]
    async fn test_transport_concurrent_sends() {
        let transport = Arc::new(LocalTransport);
        let actor = Arc::new(MockActor::new());

        let handles = (0..10)
            .map(|i| {
                let transport = Arc::clone(&transport);
                let actor = Arc::clone(&actor);

                tokio::spawn(async move {
                    let msg = TestMessage {
                        content: format!("concurrent_message_{i}"),
                    };
                    let arc_msg: Arc<dyn Any + Send + Sync> = Arc::new(msg);
                    transport.send(actor.as_ref(), arc_msg).await
                })
            })
            .collect::<Vec<_>>();

        let results = futures::future::join_all(handles).await;

        // All sends should succeed
        for result in results {
            assert!(result.is_ok());
            assert!(result.unwrap().is_ok());
        }

        // Should have received 10 messages
        assert_eq!(actor.received_messages().len(), 10);
    }

    #[test]
    fn test_local_transport_creation() {
        let _transport = LocalTransport;
        // LocalTransport is a unit struct, so creation should be trivial
    }

    #[test]
    fn test_local_transport_debug() {
        let transport = LocalTransport;

        // Should be able to debug print (though it's a unit struct)
        let _debug_str = format!("{transport:?}");
    }

    #[test]
    fn test_local_transport_clone() {
        let transport1 = LocalTransport;
        let transport2 = LocalTransport;

        // Unit structs should be equal
        // Note: We can't directly compare because LocalTransport doesn't implement PartialEq
        let _t1_debug = format!("{transport1:?}");
        let _t2_debug = format!("{transport2:?}");
    }

    #[tokio::test]
    async fn test_transport_with_wrong_message_type() {
        let transport = LocalTransport;
        let actor = MockActor::new();

        // Send a different type of message
        #[allow(dead_code)]
        #[derive(Debug)]
        struct WrongMessage {
            data: i32,
        }

        let msg = WrongMessage { data: 42 };
        let arc_msg: Arc<dyn Any + Send + Sync> = Arc::new(msg);

        let result = transport.send(&actor, arc_msg).await;

        // Should fail due to type mismatch in mock actor
        assert!(result.is_err());
        assert_eq!(actor.received_messages().len(), 0);
    }
}
