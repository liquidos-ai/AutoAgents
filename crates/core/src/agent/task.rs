use std::any::Any;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;
use crate::actor::ActorTask;
use crate::protocol::{ActorID, SubmissionId};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub prompt: String,
    pub submission_id: SubmissionId,
    pub completed: bool,
    pub result: Option<Value>,
    agent_id: Option<ActorID>,
}

impl Task {
    pub fn new<T: Into<String>>(task: T, agent_id: Option<ActorID>) -> Self {
        Self {
            prompt: task.into(),
            submission_id: Uuid::new_v4(),
            completed: false,
            result: None,
            agent_id,
        }
    }
}

impl ActorTask for Task {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
