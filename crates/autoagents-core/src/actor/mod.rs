mod messaging;
mod subscriber;
mod topic;
mod transport;

use async_trait::async_trait;
pub use messaging::{ActorMessage, CloneableMessage, SharedMessage};
use ractor::ActorRef;
use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;
pub use subscriber::{SharedSubscriber, TypedSubscriber};
pub use topic::Topic;
pub use transport::{LocalTransport, Transport};

#[async_trait]
pub trait AnyActor: Send + Sync + Debug {
    async fn send_any(
        &self,
        msg: Arc<dyn Any + Send + Sync>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

// For actors that receive cloneable messages
#[async_trait]
impl<M: CloneableMessage + 'static> AnyActor for ActorRef<M> {
    async fn send_any(
        &self,
        msg: Arc<dyn Any + Send + Sync>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let msg = msg.downcast_ref::<M>().ok_or("Message type mismatch")?;

        self.cast(msg.clone()).map_err(|e| e.into())
    }
}

// Special implementation for SharedMessage actors
// This doesn't conflict because SharedMessage<M> doesn't implement CloneableMessage
#[async_trait]
impl<M: Send + Sync + 'static> AnyActor for ActorRef<SharedMessage<M>> {
    async fn send_any(
        &self,
        msg: Arc<dyn Any + Send + Sync>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Downcast to SharedMessage<M>
        let shared_msg = msg
            .downcast_ref::<SharedMessage<M>>()
            .ok_or("Message type mismatch")?;

        // Clone the SharedMessage (which clones the Arc, not M)
        self.cast(shared_msg.clone()).map_err(|e| e.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    // Test message types
    #[derive(Debug, Clone, PartialEq)]
    struct TestCloneableMessage {
        content: String,
    }
    impl ActorMessage for TestCloneableMessage {}
    impl CloneableMessage for TestCloneableMessage {}

    #[derive(Debug, PartialEq)]
    struct TestNonCloneableMessage {
        data: String,
    }
    impl ActorMessage for TestNonCloneableMessage {}

    // Mock actor for testing
    #[derive(Debug)]
    struct MockActor {
        received_messages: Arc<Mutex<Vec<String>>>,
    }

    impl MockActor {
        fn new() -> Self {
            Self {
                received_messages: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn received_messages(&self) -> Vec<String> {
            self.received_messages.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl AnyActor for MockActor {
        async fn send_any(
            &self,
            msg: Arc<dyn Any + Send + Sync>,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            if let Some(cloneable_msg) = msg.downcast_ref::<TestCloneableMessage>() {
                self.received_messages
                    .lock()
                    .unwrap()
                    .push(format!("cloneable:{}", cloneable_msg.content));
                Ok(())
            } else if let Some(shared_msg) =
                msg.downcast_ref::<SharedMessage<TestNonCloneableMessage>>()
            {
                self.received_messages
                    .lock()
                    .unwrap()
                    .push(format!("shared:{}", shared_msg.inner().data));
                Ok(())
            } else {
                Err("Unknown message type".into())
            }
        }
    }

    #[test]
    fn test_any_actor_trait_object_creation() {
        let actor = MockActor::new();
        let _trait_obj: Box<dyn AnyActor> = Box::new(actor);
    }

    #[tokio::test]
    async fn test_any_actor_send_cloneable_message() {
        let actor = MockActor::new();
        let msg = TestCloneableMessage {
            content: "test_message".to_string(),
        };
        let arc_msg: Arc<dyn Any + Send + Sync> = Arc::new(msg);

        let result = actor.send_any(arc_msg).await;

        assert!(result.is_ok());
        assert_eq!(actor.received_messages(), vec!["cloneable:test_message"]);
    }

    #[tokio::test]
    async fn test_any_actor_send_shared_message() {
        let actor = MockActor::new();
        let inner_msg = TestNonCloneableMessage {
            data: "shared_data".to_string(),
        };
        let shared_msg = SharedMessage::new(inner_msg);
        let arc_msg: Arc<dyn Any + Send + Sync> = Arc::new(shared_msg);

        let result = actor.send_any(arc_msg).await;

        assert!(result.is_ok());
        assert_eq!(actor.received_messages(), vec!["shared:shared_data"]);
    }

    #[tokio::test]
    async fn test_any_actor_send_unknown_message() {
        let actor = MockActor::new();

        #[allow(dead_code)]
        #[derive(Debug)]
        struct UnknownMessage {
            value: i32,
        }

        let msg = UnknownMessage { value: 42 };
        let arc_msg: Arc<dyn Any + Send + Sync> = Arc::new(msg);

        let result = actor.send_any(arc_msg).await;

        assert!(result.is_err());
        assert_eq!(actor.received_messages().len(), 0);
    }

    #[test]
    fn test_actor_message_trait_bounds() {
        // Test that our message types satisfy the trait bounds
        fn assert_actor_message<T: ActorMessage>() {}
        fn assert_cloneable_message<T: CloneableMessage>() {}

        assert_actor_message::<TestCloneableMessage>();
        assert_actor_message::<TestNonCloneableMessage>();
        assert_cloneable_message::<TestCloneableMessage>();

        // TestNonCloneableMessage should not satisfy CloneableMessage
        // assert_cloneable_message::<TestNonCloneableMessage>(); // This would fail to compile
    }

    #[test]
    fn test_shared_message_actor_message_trait() {
        fn assert_actor_message<T: ActorMessage>() {}

        assert_actor_message::<SharedMessage<TestNonCloneableMessage>>();

        // SharedMessage should NOT satisfy CloneableMessage to avoid conflicts
        // fn assert_cloneable_message<T: CloneableMessage>() {}
        // assert_cloneable_message::<SharedMessage<TestNonCloneableMessage>>(); // Would fail
    }

    #[tokio::test]
    async fn test_multiple_any_actors() {
        let actor1 = MockActor::new();
        let actor2 = MockActor::new();

        let actors: Vec<Box<dyn AnyActor>> = vec![Box::new(actor1), Box::new(actor2)];

        let msg = TestCloneableMessage {
            content: "broadcast".to_string(),
        };
        let arc_msg: Arc<dyn Any + Send + Sync> = Arc::new(msg);

        for actor in &actors {
            let result = actor.send_any(arc_msg.clone()).await;
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_actor_debug_trait() {
        let actor = MockActor::new();
        let debug_str = format!("{actor:?}");
        assert!(debug_str.contains("MockActor"));
    }

    #[tokio::test]
    async fn test_concurrent_any_actor_sends() {
        let actor = Arc::new(MockActor::new());

        let handles = (0..5)
            .map(|i| {
                let actor = Arc::clone(&actor);
                tokio::spawn(async move {
                    let msg = TestCloneableMessage {
                        content: format!("concurrent_{i}"),
                    };
                    let arc_msg: Arc<dyn Any + Send + Sync> = Arc::new(msg);
                    actor.send_any(arc_msg).await
                })
            })
            .collect::<Vec<_>>();

        let results = futures::future::join_all(handles).await;

        for result in results {
            assert!(result.is_ok());
            assert!(result.unwrap().is_ok());
        }

        // Should have received 5 messages
        assert_eq!(actor.received_messages().len(), 5);
    }
}
