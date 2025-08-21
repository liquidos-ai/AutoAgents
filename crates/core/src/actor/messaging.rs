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
        Self { inner: Arc::new(msg) }
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