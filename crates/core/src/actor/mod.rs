mod messaging;
mod topic;
mod transport;

use async_trait::async_trait;
pub use messaging::{ActorMessage, CloneableMessage, SharedMessage};
use ractor::ActorRef;
use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;
pub use topic::Topic;
pub use transport::{LocalTransport, Transport};

#[async_trait]
pub trait AnyActor: Send + Sync + Debug {
    async fn send_any(&self, msg: Arc<dyn Any + Send + Sync>) -> Result<(), Box<dyn std::error::Error>>;
}

// For actors that receive cloneable messages
#[async_trait]
impl<M: CloneableMessage + 'static> AnyActor for ActorRef<M> {
    async fn send_any(&self, msg: Arc<dyn Any + Send + Sync>) -> Result<(), Box<dyn std::error::Error>> {
        let msg = msg
            .downcast_ref::<M>()
            .ok_or("Message type mismatch")?;

        self.cast(msg.clone()).map_err(|e| e.into())
    }
}

// Special implementation for SharedMessage actors
// This doesn't conflict because SharedMessage<M> doesn't implement CloneableMessage
#[async_trait]
impl<M: Send + Sync + 'static> AnyActor for ActorRef<SharedMessage<M>> {
    async fn send_any(&self, msg: Arc<dyn Any + Send + Sync>) -> Result<(), Box<dyn std::error::Error>> {
        // Downcast to SharedMessage<M>
        let shared_msg = msg
            .downcast_ref::<SharedMessage<M>>()
            .ok_or("Message type mismatch")?;

        // Clone the SharedMessage (which clones the Arc, not M)
        self.cast(shared_msg.clone()).map_err(|e| e.into())
    }
}