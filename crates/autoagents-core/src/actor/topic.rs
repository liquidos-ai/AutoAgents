use crate::actor::messaging::ActorMessage;
use std::any::TypeId;
use std::fmt::Debug;
use std::marker::PhantomData;
use uuid::Uuid;

// Generic topic that is type-safe at compile time
#[derive(Clone)]
pub struct Topic<M: ActorMessage> {
    name: String,
    id: Uuid,
    _phantom: PhantomData<M>,
}
impl<M: ActorMessage> Topic<M> {
    pub fn new(name: impl Into<String>) -> Self {
        Topic {
            name: name.into(),
            id: Uuid::new_v4(),
            _phantom: PhantomData,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn id(&self) -> Uuid {
        self.id
    }

    pub fn type_id(&self) -> TypeId {
        TypeId::of::<M>()
    }
}

impl<M: ActorMessage> Debug for Topic<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Topic")
            .field("name", &self.name)
            .field("id", &self.id)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::TypeId;

    // Test message types
    #[derive(Debug, Clone)]
    struct TestMessage {
        _content: String,
    }
    impl ActorMessage for TestMessage {}

    #[derive(Debug)]
    struct AnotherTestMessage {
        _data: i32,
    }
    impl ActorMessage for AnotherTestMessage {}

    #[test]
    fn test_topic_creation_with_string() {
        let topic = Topic::<TestMessage>::new("test_topic");

        assert_eq!(topic.name(), "test_topic");
        assert!(!topic.id().is_nil());
    }

    #[test]
    fn test_topic_creation_with_str() {
        let topic = Topic::<TestMessage>::new("str_topic");

        assert_eq!(topic.name(), "str_topic");
        assert!(!topic.id().is_nil());
    }

    #[test]
    fn test_topic_unique_ids() {
        let topic1 = Topic::<TestMessage>::new("topic1");
        let topic2 = Topic::<TestMessage>::new("topic2");

        assert_ne!(topic1.id(), topic2.id());
    }

    #[test]
    fn test_topic_same_name_different_ids() {
        let topic1 = Topic::<TestMessage>::new("same_name");
        let topic2 = Topic::<TestMessage>::new("same_name");

        assert_eq!(topic1.name(), topic2.name());
        assert_ne!(topic1.id(), topic2.id());
    }

    #[test]
    fn test_topic_type_id_same_type() {
        let topic1 = Topic::<TestMessage>::new("topic1");
        let topic2 = Topic::<TestMessage>::new("topic2");

        assert_eq!(topic1.type_id(), topic2.type_id());
        assert_eq!(topic1.type_id(), TypeId::of::<TestMessage>());
    }

    #[test]
    fn test_topic_type_id_different_types() {
        let topic1 = Topic::<TestMessage>::new("topic1");
        let topic2 = Topic::<AnotherTestMessage>::new("topic2");

        assert_ne!(topic1.type_id(), topic2.type_id());
        assert_eq!(topic1.type_id(), TypeId::of::<TestMessage>());
        assert_eq!(topic2.type_id(), TypeId::of::<AnotherTestMessage>());
    }

    #[test]
    fn test_topic_clone() {
        let original = Topic::<TestMessage>::new("original_topic");
        let cloned = original.clone();

        assert_eq!(original.name(), cloned.name());
        assert_eq!(original.id(), cloned.id());
        assert_eq!(original.type_id(), cloned.type_id());
    }

    #[test]
    fn test_topic_debug() {
        let topic = Topic::<TestMessage>::new("debug_topic");
        let debug_str = format!("{topic:?}");

        assert!(debug_str.contains("Topic"));
        assert!(debug_str.contains("debug_topic"));
        assert!(debug_str.contains("name"));
        assert!(debug_str.contains("id"));
    }

    #[test]
    fn test_topic_name_accessor() {
        let topic = Topic::<TestMessage>::new("accessor_test");

        assert_eq!(topic.name(), "accessor_test");
    }

    #[test]
    fn test_topic_id_accessor() {
        let topic = Topic::<TestMessage>::new("id_test");
        let id = topic.id();

        assert!(!id.is_nil());
        assert_eq!(topic.id(), id); // Should return the same ID
    }

    #[test]
    fn test_topic_type_id_accessor() {
        let topic = Topic::<TestMessage>::new("type_id_test");
        let type_id = topic.type_id();

        assert_eq!(type_id, TypeId::of::<TestMessage>());
        assert_eq!(topic.type_id(), type_id); // Should return the same TypeId
    }

    #[test]
    fn test_topic_with_empty_name() {
        let topic = Topic::<TestMessage>::new("");

        assert_eq!(topic.name(), "");
        assert!(!topic.id().is_nil());
    }

    #[test]
    fn test_topic_with_unicode_name() {
        let topic = Topic::<TestMessage>::new("テスト_トピック_🚀");

        assert_eq!(topic.name(), "テスト_トピック_🚀");
        assert!(!topic.id().is_nil());
    }

    #[test]
    fn test_topic_with_long_name() {
        let long_name = "a".repeat(1000);
        let topic = Topic::<TestMessage>::new(long_name.clone());

        assert_eq!(topic.name(), long_name);
        assert!(!topic.id().is_nil());
    }

    #[test]
    fn test_topic_type_safety() {
        // This test ensures compile-time type safety
        let _topic1: Topic<TestMessage> = Topic::new("test");
        let _topic2: Topic<AnotherTestMessage> = Topic::new("test");

        // Different types should not be assignable
        // This won't compile: let _topic3: Topic<TestMessage> = topic2;
    }

    #[test]
    fn test_topic_phantom_data() {
        // Ensure PhantomData doesn't affect size significantly
        let topic = Topic::<TestMessage>::new("phantom_test");

        // PhantomData should be zero-sized
        assert_eq!(std::mem::size_of_val(&topic._phantom), 0);
    }
}
