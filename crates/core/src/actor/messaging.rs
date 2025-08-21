use std::sync::Arc;

/// Generic trait for messages that can be sent between actors
pub trait ActorMessage: Send + Sync + 'static {}

// For messages that can be cloned
pub trait CloneableMessage: ActorMessage + Clone {}

pub struct SharedMessage<M> {
    inner: Arc<M>,
}

// Manually implement Clone without requiring M: Clone
impl<M> Clone for SharedMessage<M> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<M> SharedMessage<M> {
    pub fn new(msg: M) -> Self {
        Self {
            inner: Arc::new(msg),
        }
    }

    pub fn inner(&self) -> &M {
        &self.inner
    }

    pub fn into_inner(self) -> Arc<M> {
        self.inner
    }
}

// SharedMessage<M> is always a PubSubMessage (but NOT CloneableMessage to avoid conflicts)
impl<M: Send + Sync + 'static> ActorMessage for SharedMessage<M> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // Test message types
    #[derive(Debug, Clone, PartialEq)]
    struct TestMessage {
        content: String,
    }

    impl ActorMessage for TestMessage {}
    impl CloneableMessage for TestMessage {}

    #[derive(Debug, PartialEq)]
    struct NonCloneableMessage {
        data: String,
    }

    impl ActorMessage for NonCloneableMessage {}

    #[test]
    fn test_shared_message_creation() {
        let msg = TestMessage {
            content: "test".to_string(),
        };
        let shared = SharedMessage::new(msg);

        assert_eq!(shared.inner().content, "test");
    }

    #[test]
    fn test_shared_message_clone() {
        let msg = TestMessage {
            content: "original".to_string(),
        };
        let shared1 = SharedMessage::new(msg);
        let shared2 = shared1.clone();

        // Both should reference the same inner data
        assert_eq!(shared1.inner().content, shared2.inner().content);
        assert_eq!(shared1.inner().content, "original");

        // Arc reference count should be 2
        assert_eq!(Arc::strong_count(&shared1.inner), 2);
        assert_eq!(Arc::strong_count(&shared2.inner), 2);
    }

    #[test]
    fn test_shared_message_into_inner() {
        let msg = TestMessage {
            content: "into_inner_test".to_string(),
        };
        let shared = SharedMessage::new(msg);
        let arc = shared.into_inner();

        assert_eq!(arc.content, "into_inner_test");
        assert_eq!(Arc::strong_count(&arc), 1);
    }

    #[test]
    fn test_shared_message_inner_reference() {
        let msg = TestMessage {
            content: "reference_test".to_string(),
        };
        let shared = SharedMessage::new(msg);
        let inner_ref = shared.inner();

        assert_eq!(inner_ref.content, "reference_test");
    }

    #[test]
    fn test_shared_message_with_non_cloneable() {
        let msg = NonCloneableMessage {
            data: "non_cloneable".to_string(),
        };
        let shared1 = SharedMessage::new(msg);
        let shared2 = shared1.clone();

        // Should work even with non-cloneable messages
        assert_eq!(shared1.inner().data, "non_cloneable");
        assert_eq!(shared2.inner().data, "non_cloneable");
    }

    #[test]
    fn test_actor_message_trait_implemented() {
        let msg = TestMessage {
            content: "trait_test".to_string(),
        };
        let shared = SharedMessage::new(msg);

        // Should be able to treat as ActorMessage
        fn accepts_actor_message<T: ActorMessage>(_: T) {}
        accepts_actor_message(shared);
    }

    #[test]
    fn test_cloneable_message_trait() {
        let msg = TestMessage {
            content: "cloneable_test".to_string(),
        };

        // Should implement both ActorMessage and CloneableMessage
        fn accepts_cloneable_message<T: CloneableMessage>(_: T) {}
        accepts_cloneable_message(msg);
    }

    #[test]
    fn test_shared_message_debug() {
        let msg = TestMessage {
            content: "debug_test".to_string(),
        };
        let shared = SharedMessage::new(msg);

        // Debug should work through inner message's Debug impl
        let debug_str = format!("{:?}", shared.inner());
        assert!(debug_str.contains("debug_test"));
    }

    #[test]
    fn test_multiple_shared_message_clones() {
        let msg = TestMessage {
            content: "multi_clone_test".to_string(),
        };
        let shared1 = SharedMessage::new(msg);
        let shared2 = shared1.clone();
        let shared3 = shared2.clone();
        let shared4 = shared1.clone();

        // All should reference the same data
        assert_eq!(shared1.inner().content, "multi_clone_test");
        assert_eq!(shared2.inner().content, "multi_clone_test");
        assert_eq!(shared3.inner().content, "multi_clone_test");
        assert_eq!(shared4.inner().content, "multi_clone_test");

        // Arc reference count should be 4
        assert_eq!(Arc::strong_count(&shared1.inner), 4);
    }

    #[test]
    fn test_shared_message_arc_cleanup() {
        let msg = TestMessage {
            content: "cleanup_test".to_string(),
        };
        let shared1 = SharedMessage::new(msg);
        let shared2 = shared1.clone();

        assert_eq!(Arc::strong_count(&shared1.inner), 2);

        drop(shared2);
        assert_eq!(Arc::strong_count(&shared1.inner), 1);

        drop(shared1);
        // shared1 is dropped, so we can't check the count anymore
    }
}
