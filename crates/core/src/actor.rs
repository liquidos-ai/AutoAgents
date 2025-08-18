use std::any::Any;
use std::fmt::Debug;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::Sender;
use crate::protocol::Event;

pub trait ActorTask: Send + 'static + Debug {
    fn as_any(&self) -> &dyn Any;
}

#[derive(Debug)]
pub struct ActorMessage {
    pub task: Box<dyn ActorTask>,
    #[cfg(feature = "single_threaded")]
    pub tx: Sender<Event>,
}