use crate::actor::{AnyActor, CloneableMessage, SharedMessage, Transport};
use crate::protocol::{Event, RuntimeID};
use async_trait::async_trait;
use ractor::ActorRef;
use std::any::{Any, TypeId};
use std::fmt::Debug;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::SendError;
use tokio::task::JoinError;
use tokio_stream::wrappers::ReceiverStream;

pub(crate) mod manager;
mod single_threaded;
use crate::actor::Topic;
pub use single_threaded::SingleThreadedRuntime;

#[cfg(feature = "cluster")]
mod cluster_runtime;

#[cfg(feature = "cluster")]
pub use cluster_runtime::ClusterMessage;
#[cfg(feature = "cluster")]
pub use cluster_runtime::{ClusterRuntime, ClusterHostRuntime, ClusterClientRuntime};

/// Configuration for runtime instances
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub queue_size: Option<usize>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            queue_size: Some(100),
        }
    }
}

/// Error types for Session operations
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("Send Message Error: {0}")]
    SendMessage(String),

    #[error("TopicTypeMismatch")]
    TopicTypeMismatch(String, TypeId),

    #[error("Join Error: {0}")]
    JoinError(JoinError),

    #[error("Event error: {0}")]
    EventError(#[from] SendError<Event>),
}

#[async_trait]
pub trait Runtime: Send + Sync {
    fn id(&self) -> RuntimeID;

    async fn subscribe_any(
        &self,
        topic_name: &str,
        topic_type: TypeId,
        actor: Arc<dyn AnyActor>,
    ) -> Result<(), RuntimeError>;

    async fn publish_any(
        &self,
        topic_name: &str,
        topic_type: TypeId,
        message: Arc<dyn Any + Send + Sync>,
    ) -> Result<(), RuntimeError>;

    //Local event processing event handler, This is passed to the agent handler and agents emit events using this, The runtime is responsible to move it to the parent environment
    async fn tx(&self) -> mpsc::Sender<Event>;
    async fn transport(&self) -> Arc<dyn Transport>;
    async fn take_event_receiver(&self) -> Option<ReceiverStream<Event>>;
    async fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    async fn stop(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Connect to a remote cluster node (cluster feature only)
    #[cfg(feature = "cluster")]
    async fn connect_to_remote(
        &self,
        remote_addr: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let _ = remote_addr;
        Err("Cluster connection not supported in this runtime".into())
    }
}

#[async_trait]
pub trait TypedRuntime: Runtime {
    async fn subscribe<M>(&self, topic: &Topic<M>, actor: ActorRef<M>) -> Result<(), RuntimeError>
    where
        M: CloneableMessage + 'static,
    {
        let arc_actor = Arc::new(actor) as Arc<dyn AnyActor>;
        self.subscribe_any(topic.name(), TypeId::of::<M>(), arc_actor)
            .await
    }

    async fn publish<M>(&self, topic: &Topic<M>, message: M) -> Result<(), RuntimeError>
    where
        M: CloneableMessage + 'static,
    {
        let arc_msg = Arc::new(message) as Arc<dyn Any + Send + Sync>;
        self.publish_any(topic.name(), TypeId::of::<M>(), arc_msg)
            .await
    }

    async fn send_message<M: CloneableMessage + 'static>(
        &self,
        message: M,
        addr: ActorRef<M>,
    ) -> Result<(), RuntimeError> {
        addr.cast(message)
            .map_err(|e| RuntimeError::SendMessage(e.to_string()))
    }

    async fn subscribe_shared<M>(
        &self,
        topic: &Topic<SharedMessage<M>>,
        actor: ActorRef<SharedMessage<M>>,
    ) -> Result<(), RuntimeError>
    where
        M: Send + Sync + 'static,
    {
        let arc_actor = Arc::new(actor) as Arc<dyn AnyActor>;
        self.subscribe_any(topic.name(), TypeId::of::<SharedMessage<M>>(), arc_actor)
            .await
    }

    async fn publish_shared<M>(
        &self,
        topic: &Topic<SharedMessage<M>>,
        message: M,
    ) -> Result<(), RuntimeError>
    where
        M: Send + Sync + 'static,
    {
        let shared_msg = SharedMessage::new(message);
        let arc_msg = Arc::new(shared_msg) as Arc<dyn Any + Send + Sync>;
        self.publish_any(topic.name(), TypeId::of::<SharedMessage<M>>(), arc_msg)
            .await
    }
}

// Auto-implement TypedRuntime for all Runtime implementations
impl<T: Runtime + ?Sized> TypedRuntime for T {}
