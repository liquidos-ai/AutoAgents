use crate::actor::{ActorMessage, CloneableMessage};
use crate::protocol::SubmissionId;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub prompt: String,
    pub submission_id: SubmissionId,
    pub completed: bool,
    pub result: Option<Value>,
}

impl Task {
    pub fn new<T: Into<String>>(task: T) -> Self {
        Self {
            prompt: task.into(),
            submission_id: Uuid::new_v4(),
            completed: false,
            result: None,
        }
    }
}

impl ActorMessage for Task {}
impl CloneableMessage for Task {}