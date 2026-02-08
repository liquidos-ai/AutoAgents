use crate::SubmissionId;
use crate::llm::ImageMime;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// A unit of work submitted to an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub prompt: String,
    pub image: Option<(ImageMime, Vec<u8>)>,
    #[serde(default)]
    pub system_prompt: Option<String>,
    pub submission_id: SubmissionId,
    pub completed: bool,
    pub result: Option<Value>,
}

impl Task {
    /// Create a new text-only task with a fresh submission id.
    pub fn new<T: Into<String>>(task: T) -> Self {
        Self {
            prompt: task.into(),
            image: None,
            system_prompt: None,
            submission_id: Uuid::new_v4(),
            completed: false,
            result: None,
        }
    }

    /// Create a new task with an image payload and a fresh submission id.
    pub fn new_with_image<T: Into<String>>(
        task: T,
        image_mime: ImageMime,
        image_data: Vec<u8>,
    ) -> Self {
        Self {
            prompt: task.into(),
            image: Some((image_mime, image_data)),
            system_prompt: None,
            submission_id: Uuid::new_v4(),
            completed: false,
            result: None,
        }
    }

    pub fn with_system_prompt<T: Into<String>>(mut self, prompt: T) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }
}
