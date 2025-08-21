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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_task_creation() {
        let task = Task::new("Test task");

        assert_eq!(task.prompt, "Test task");
        assert!(!task.completed);
        assert!(task.result.is_none());
        assert!(!task.submission_id.is_nil());
    }

    #[test]
    fn test_task_creation_with_string() {
        let task_str = "Another test task".to_string();
        let task = Task::new(task_str);

        assert_eq!(task.prompt, "Another test task");
    }

    #[test]
    fn test_task_serialization() {
        let task = Task::new("Serialize me");

        // Test serialization
        let serialized = serde_json::to_string(&task).unwrap();
        assert!(serialized.contains("Serialize me"));
        assert!(serialized.contains("submission_id"));
        assert!(serialized.contains("completed"));

        // Test deserialization
        let deserialized: Task = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.prompt, task.prompt);
        assert_eq!(deserialized.submission_id, task.submission_id);
        assert_eq!(deserialized.completed, task.completed);
    }

    #[test]
    fn test_task_with_result() {
        let mut task = Task::new("Task with result");
        let result_value = json!({"output": "success", "value": 42});
        task.result = Some(result_value.clone());
        task.completed = true;

        assert!(task.completed);
        assert_eq!(task.result, Some(result_value));
    }

    #[test]
    fn test_task_unique_submission_ids() {
        let task1 = Task::new("Task 1");
        let task2 = Task::new("Task 2");

        assert_ne!(task1.submission_id, task2.submission_id);
    }

    #[test]
    fn test_task_clone() {
        let original = Task::new("Original task");
        let cloned = original.clone();

        assert_eq!(original.prompt, cloned.prompt);
        assert_eq!(original.submission_id, cloned.submission_id);
        assert_eq!(original.completed, cloned.completed);
        assert_eq!(original.result, cloned.result);
    }

    #[test]
    fn test_task_debug() {
        let task = Task::new("Debug test");
        let debug_str = format!("{task:?}");

        assert!(debug_str.contains("Task"));
        assert!(debug_str.contains("Debug test"));
    }
}
