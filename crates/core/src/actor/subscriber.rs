use crate::actor::{AnyActor, CloneableMessage, SharedMessage};
use ractor::ActorRef;
use std::any::Any;
use std::marker::PhantomData;
use std::sync::Arc;

struct TypedSubscriber<M: CloneableMessage> {
    actors: Vec<Box<dyn AnyActor>>,
    _marker: PhantomData<M>,
}

impl<M: CloneableMessage + 'static> TypedSubscriber<M> {
    fn new() -> Self {
        Self { actors: Vec::new(), _marker: PhantomData }
    }

    fn add(&mut self, actor: ActorRef<M>) {
        self.actors.push(Box::new(actor) as Box<dyn AnyActor>);
    }

    async fn publish(&self, message: M) {
        let arc_msg: Arc<dyn Any + Send + Sync> = Arc::new(message);
        for actor in &self.actors {
            let _ = actor.send_any(arc_msg.clone()).await;
        }
    }
}

// Subscriber for shared messages
struct SharedSubscriber<M: Send + Sync + 'static> {
    actors: Vec<Box<dyn AnyActor>>,
    _marker: PhantomData<M>,
}

impl<M: Send + Sync + 'static> SharedSubscriber<M> {
    fn new() -> Self {
        Self { actors: Vec::new(), _marker: PhantomData }
    }

    fn add(&mut self, actor: ActorRef<SharedMessage<M>>) {
        self.actors.push(Box::new(actor) as Box<dyn AnyActor>);
    }

    async fn publish(&self, message: SharedMessage<M>) {
        let arc_msg: Arc<dyn Any + Send + Sync> = Arc::new(message);
        for actor in &self.actors {
            let _ = actor.send_any(arc_msg.clone()).await;
        }
    }
}