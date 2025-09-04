use crate::agent::task::Task;
use crate::tool::ToolCallResult;
use autoagents_llm::chat::StreamChoice;
use serde::{Deserialize, Serialize};
use std::any::{Any, TypeId};
use std::fmt::Debug;
use std::sync::Arc;
use uuid::Uuid;

/// Submission IDs are used to track actor tasks
pub type SubmissionId = Uuid;

/// Agent IDs are used to identify actors
pub type ActorID = Uuid;

/// Session IDs are used to identify sessions
pub type RuntimeID = Uuid;

/// Event IDs are used to correlate events with their responses
pub type EventId = Uuid;

/// Protocol events represent the various events that can occur during actor execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Event {
    // /// A new task has been submitted to an agent
    NewTask {
        actor_id: ActorID,
        task: Task,
    },

    /// A task has started execution
    TaskStarted {
        sub_id: SubmissionId,
        actor_id: ActorID,
        task_description: String,
    },

    /// A task has been completed
    TaskComplete {
        sub_id: SubmissionId,
        result: String,
    },

    /// A task encountered an error
    TaskError {
        sub_id: SubmissionId,
        error: String,
    },

    #[serde(skip)]
    PublishMessage {
        topic_name: String,
        topic_type: TypeId,
        message: Arc<dyn Any + Send + Sync>,
    },

    SendMessage {
        message: String,
        actor_id: ActorID,
    },

    /// Tool call requested (with ID)
    ToolCallRequested {
        id: String,
        tool_name: String,
        arguments: String,
    },

    /// Tool call completed (with ID and result)
    ToolCallCompleted {
        id: String,
        tool_name: String,
        result: serde_json::Value,
    },

    /// Tool call has failed
    ToolCallFailed {
        id: String,
        tool_name: String,
        error: String,
    },

    /// A turn has started
    TurnStarted {
        turn_number: usize,
        max_turns: usize,
    },

    /// A turn has completed
    TurnCompleted {
        turn_number: usize,
        final_turn: bool,
    },

    /// Streaming chunk from agent
    StreamChunk {
        sub_id: SubmissionId,
        chunk: StreamChoice,
    },

    /// Streaming tool call chunk
    StreamToolCall {
        sub_id: SubmissionId,
        tool_call: serde_json::Value,
    },

    /// Streaming completed
    StreamComplete {
        sub_id: SubmissionId,
    },
}

/// Internal events that are processed within the runtime
#[derive(Debug)]
pub enum InternalEvent {
    /// Process a protocol event
    ProtocolEvent(Event),
    /// Shutdown signal
    Shutdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamingTurnResult {
    Complete(String),
    ToolCallsProcessed(Vec<ToolCallResult>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_event_serialization_new_task() {
        let _ = Uuid::new_v4();
        let event = Event::NewTask {
            actor_id: Default::default(),
            task: Task::new(String::from("test")),
        };

        //Check if serialization and deserilization works properly
        let serialized = serde_json::to_string(&event).unwrap();
        let deserialized: Event = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            Event::NewTask { task, .. } => {
                assert_eq!(task.prompt, "test");
            }
            _ => panic!("Expected NewTask variant"),
        }
    }

    #[test]
    fn test_event_serialization_task_started() {
        let event = Event::TaskStarted {
            sub_id: Uuid::new_v4(),
            actor_id: Default::default(),
            task_description: "Started task".to_string(),
        };

        let serialized = serde_json::to_string(&event).unwrap();
        let deserialized: Event = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            Event::TaskStarted {
                task_description, ..
            } => {
                assert_eq!(task_description, "Started task");
            }
            _ => panic!("Expected TaskStarted variant"),
        }
    }

    #[test]
    fn test_event_serialization_tool_calls() {
        let tool_call_requested = Event::ToolCallRequested {
            id: "call_123".to_string(),
            tool_name: "test_tool".to_string(),
            arguments: "{\"param\": \"value\"}".to_string(),
        };

        let serialized = serde_json::to_string(&tool_call_requested).unwrap();
        let deserialized: Event = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            Event::ToolCallRequested {
                id,
                tool_name,
                arguments,
            } => {
                assert_eq!(id, "call_123");
                assert_eq!(tool_name, "test_tool");
                assert_eq!(arguments, "{\"param\": \"value\"}");
            }
            _ => panic!("Expected ToolCallRequested variant"),
        }
    }

    #[test]
    fn test_event_serialization_tool_call_completed() {
        let result = json!({"output": "tool result"});
        let event = Event::ToolCallCompleted {
            id: "call_456".to_string(),
            tool_name: "completed_tool".to_string(),
            result: result.clone(),
        };

        let serialized = serde_json::to_string(&event).unwrap();
        let deserialized: Event = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            Event::ToolCallCompleted {
                id,
                tool_name,
                result: res,
            } => {
                assert_eq!(id, "call_456");
                assert_eq!(tool_name, "completed_tool");
                assert_eq!(res, result);
            }
            _ => panic!("Expected ToolCallCompleted variant"),
        }
    }

    #[test]
    fn test_event_serialization_tool_call_failed() {
        let event = Event::ToolCallFailed {
            id: "call_789".to_string(),
            tool_name: "failed_tool".to_string(),
            error: "Tool execution failed".to_string(),
        };

        let serialized = serde_json::to_string(&event).unwrap();
        let deserialized: Event = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            Event::ToolCallFailed {
                id,
                tool_name,
                error,
            } => {
                assert_eq!(id, "call_789");
                assert_eq!(tool_name, "failed_tool");
                assert_eq!(error, "Tool execution failed");
            }
            _ => panic!("Expected ToolCallFailed variant"),
        }
    }

    #[test]
    fn test_event_serialization_turn_events() {
        let turn_started = Event::TurnStarted {
            turn_number: 1,
            max_turns: 10,
        };

        let serialized = serde_json::to_string(&turn_started).unwrap();
        let deserialized: Event = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            Event::TurnStarted {
                turn_number,
                max_turns,
            } => {
                assert_eq!(turn_number, 1);
                assert_eq!(max_turns, 10);
            }
            _ => panic!("Expected TurnStarted variant"),
        }

        let turn_completed = Event::TurnCompleted {
            turn_number: 1,
            final_turn: false,
        };

        let serialized = serde_json::to_string(&turn_completed).unwrap();
        let deserialized: Event = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            Event::TurnCompleted {
                turn_number,
                final_turn,
            } => {
                assert_eq!(turn_number, 1);
                assert!(!final_turn);
            }
            _ => panic!("Expected TurnCompleted variant"),
        }
    }

    #[test]
    fn test_uuid_types() {
        let submission_id: SubmissionId = Uuid::new_v4();
        let agent_id: ActorID = Uuid::new_v4();
        let runtime_id: RuntimeID = Uuid::new_v4();
        let event_id: EventId = Uuid::new_v4();

        // Test that all UUID types can be used interchangeably
        assert_ne!(submission_id, agent_id);
        assert_ne!(runtime_id, event_id);
    }
}
