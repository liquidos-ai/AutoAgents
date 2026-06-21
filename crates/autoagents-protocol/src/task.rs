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
    /// Arbitrary application-provided metadata (session/chat isolation, app context, anything the app threads through).
    #[serde(default)]
    pub app_meta: Option<Value>,
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
            app_meta: None,
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
            app_meta: None,
        }
    }

    pub fn with_system_prompt<T: Into<String>>(mut self, prompt: T) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Attach arbitrary application metadata (a JSON value, typically an object).
    pub fn with_app_meta(mut self, meta: Value) -> Self {
        self.app_meta = Some(meta);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_meta_roundtrip() {
        let mut task = Task::new("hello");
        task.app_meta = Some(serde_json::json!({
            "session_id": "s1",
            "chat_id": "c1",
            "extra": { "nested": [1, 2, 3] },
        }));
        let back: Task = serde_json::from_str(&serde_json::to_string(&task).unwrap()).unwrap();
        let meta = back
            .app_meta
            .expect("app_meta preserved across serde roundtrip");
        assert_eq!(meta.get("session_id").and_then(|v| v.as_str()), Some("s1"));
        assert_eq!(meta.get("chat_id").and_then(|v| v.as_str()), Some("c1"));
        assert!(
            meta.get("extra").is_some(),
            "arbitrary nested app data preserved"
        );
    }

    #[test]
    fn app_meta_defaults_to_none_when_absent() {
        // Back-compat: a payload serialized before app_meta existed must still deserialize.
        let mut v = serde_json::to_value(Task::new("legacy")).unwrap();
        v.as_object_mut().unwrap().remove("app_meta");
        let back: Task = serde_json::from_value(v).unwrap();
        assert!(back.app_meta.is_none());
    }

    #[test]
    fn with_app_meta_builder_sets_field() {
        let task = Task::new("hi")
            .with_app_meta(serde_json::json!({"session_id": "s1", "chat_id": "c1"}));
        let meta = task.app_meta.expect("with_app_meta should set app_meta");
        assert_eq!(meta.get("session_id").and_then(|v| v.as_str()), Some("s1"));
        assert_eq!(meta.get("chat_id").and_then(|v| v.as_str()), Some("c1"));
    }
}
