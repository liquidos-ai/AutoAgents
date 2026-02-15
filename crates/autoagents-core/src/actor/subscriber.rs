use crate::actor::{AnyActor, CloneableMessage, SharedMessage};
use ractor::ActorRef;
use std::any::Any;
use std::marker::PhantomData;
use std::sync::Arc;

pub struct TypedSubscriber<M: CloneableMessage> {
    actors: Vec<Box<dyn AnyActor>>,
    _marker: PhantomData<M>,
}

impl<M: CloneableMessage + 'static> Default for TypedSubscriber<M> {
    fn default() -> Self {
        Self {
            actors: Vec::new(),
            _marker: PhantomData,
        }
    }
}

impl<M: CloneableMessage + 'static> TypedSubscriber<M> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, actor: ActorRef<M>) {
        self.actors.push(Box::new(actor) as Box<dyn AnyActor>);
    }

    pub async fn publish(&self, message: M) {
        let arc_msg: Arc<dyn Any + Send + Sync> = Arc::new(message);
        for actor in &self.actors {
            let _ = actor.send_any(arc_msg.clone()).await;
        }
    }
}

// Subscriber for shared messages
pub struct SharedSubscriber<M: Send + Sync + 'static> {
    actors: Vec<Box<dyn AnyActor>>,
    _marker: PhantomData<M>,
}

impl<M: Send + Sync + 'static> Default for SharedSubscriber<M> {
    fn default() -> Self {
        Self {
            actors: Vec::new(),
            _marker: PhantomData,
        }
    }
}

impl<M: Send + Sync + 'static> SharedSubscriber<M> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, actor: ActorRef<SharedMessage<M>>) {
        self.actors.push(Box::new(actor) as Box<dyn AnyActor>);
    }

    pub async fn publish(&self, message: SharedMessage<M>) {
        let arc_msg: Arc<dyn Any + Send + Sync> = Arc::new(message);
        for actor in &self.actors {
            let _ = actor.send_any(arc_msg.clone()).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::{ActorMessage, CloneableMessage, SharedMessage};
    use async_trait::async_trait;
    use ractor::{Actor, ActorProcessingErr, ActorRef};
    use tokio::sync::Mutex;
    use tokio::time::{Duration, sleep};

    #[derive(Clone, Debug)]
    struct TestMessage {
        content: String,
    }

    impl ActorMessage for TestMessage {}
    impl CloneableMessage for TestMessage {}

    struct CollectActor {
        received: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl Actor for CollectActor {
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
    async fn test_typed_subscriber_publish() {
        let received = Arc::new(Mutex::new(Vec::new()));
        let (actor_ref, _handle) = Actor::spawn(
            None,
            CollectActor {
                received: received.clone(),
            },
            received.clone(),
        )
        .await
        .unwrap();

        let mut subscriber = TypedSubscriber::<TestMessage>::new();
        subscriber.add(actor_ref);
        subscriber
            .publish(TestMessage {
                content: "hello".to_string(),
            })
            .await;

        sleep(Duration::from_millis(10)).await;
        let items = received.lock().await.clone();
        assert_eq!(items, vec!["hello"]);
    }

    #[derive(Debug)]
    struct SharedPayload {
        value: String,
    }

    impl ActorMessage for SharedPayload {}

    struct SharedActor {
        received: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl Actor for SharedActor {
        type Msg = SharedMessage<SharedPayload>;
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
            received.push(message.inner().value.clone());
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_shared_subscriber_publish() {
        let received = Arc::new(Mutex::new(Vec::new()));
        let (actor_ref, _handle) = Actor::spawn(
            None,
            SharedActor {
                received: received.clone(),
            },
            received.clone(),
        )
        .await
        .unwrap();

        let mut subscriber = SharedSubscriber::<SharedPayload>::new();
        subscriber.add(actor_ref);
        subscriber
            .publish(SharedMessage::new(SharedPayload {
                value: "shared".to_string(),
            }))
            .await;

        sleep(Duration::from_millis(10)).await;
        let items = received.lock().await.clone();
        assert_eq!(items, vec!["shared"]);
    }
}
