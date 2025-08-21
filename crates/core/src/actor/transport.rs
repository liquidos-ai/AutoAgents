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
    ) -> Result<(), Box<dyn std::error::Error>>;
}

pub struct LocalTransport;

#[async_trait]
impl Transport for LocalTransport {
    async fn send(
        &self,
        actor: &dyn AnyActor,
        msg: Arc<dyn Any + Send + Sync>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        actor.send_any(msg).await
    }
}