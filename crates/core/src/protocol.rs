use autoagents_llm::chat::ChatMessage;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Submission IDs are used to track agent tasks
pub type SubmissionId = Uuid;

/// Agent IDs are used to identify agents
pub type AgentID = Uuid;

/// Session IDs are used to identify sessions
pub type SessionId = Uuid;

/// Event IDs are used to correlate events with their responses
pub type EventId = Uuid;

/// Protocol events represent the various events that can occur during agent execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Event {
    /// A new task has been submitted to an agent
    NewTask {
        sub_id: SubmissionId,
        agent_id: Option<AgentID>,
        prompt: String,
    },

    /// A task has started execution
    TaskStarted {
        sub_id: SubmissionId,
        agent_id: AgentID,
        task_description: String,
    },

    /// A task has been completed
    TaskComplete {
        sub_id: SubmissionId,
        result: TaskResult,
    },

    /// A task encountered an error
    TaskError { sub_id: SubmissionId, error: String },

    /// An agent message to the user
    AgentMessage {
        sub_id: SubmissionId,
        message: AgentMessage,
    },

    /// Tool call has started
    ToolCallStarted { tool_name: String, args: String },

    /// Tool call has completed (simple)
    ToolCallCompletedSimple { tool_name: String, success: bool },

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

    /// A warning message
    Warning { message: String },

    /// An error occurred during task execution
    Error { sub_id: SubmissionId, error: String },
}

/// Results from a completed task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskResult {
    /// The task was completed successfully
    Success(String),

    /// The task was completed with a value
    Value(serde_json::Value),

    /// The task failed
    Failure(String),

    /// The task was aborted
    Aborted,
}

/// Messages from the agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    /// The content of the message
    pub content: String,

    /// Optional chat messages for a full conversation history
    pub chat_messages: Option<Vec<ChatMessage>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_event_serialization_new_task() {
        let event = Event::NewTask {
            sub_id: Uuid::new_v4(),
            agent_id: Some(Uuid::new_v4()),
            prompt: "Test task".to_string(),
        };

        let serialized = serde_json::to_string(&event).unwrap();
        let deserialized: Event = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            Event::NewTask { prompt, .. } => {
                assert_eq!(prompt, "Test task");
            }
            _ => panic!("Expected NewTask variant"),
        }
    }

    #[test]
    fn test_event_serialization_task_started() {
        let event = Event::TaskStarted {
            sub_id: Uuid::new_v4(),
            agent_id: Uuid::new_v4(),
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
    fn test_event_serialization_task_complete() {
        let event = Event::TaskComplete {
            sub_id: Uuid::new_v4(),
            result: TaskResult::Success("Task completed".to_string()),
        };

        let serialized = serde_json::to_string(&event).unwrap();
        let deserialized: Event = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            Event::TaskComplete { result, .. } => match result {
                TaskResult::Success(msg) => assert_eq!(msg, "Task completed"),
                _ => panic!("Expected Success result"),
            },
            _ => panic!("Expected TaskComplete variant"),
        }
    }

    #[test]
    fn test_event_serialization_task_error() {
        let event = Event::TaskError {
            sub_id: Uuid::new_v4(),
            error: "Task failed".to_string(),
        };

        let serialized = serde_json::to_string(&event).unwrap();
        let deserialized: Event = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            Event::TaskError { error, .. } => {
                assert_eq!(error, "Task failed");
            }
            _ => panic!("Expected TaskError variant"),
        }
    }

    #[test]
    fn test_event_serialization_agent_message() {
        let agent_message = AgentMessage {
            content: "Agent response".to_string(),
            chat_messages: None,
        };

        let event = Event::AgentMessage {
            sub_id: Uuid::new_v4(),
            message: agent_message,
        };

        let serialized = serde_json::to_string(&event).unwrap();
        let deserialized: Event = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            Event::AgentMessage { message, .. } => {
                assert_eq!(message.content, "Agent response");
                assert!(message.chat_messages.is_none());
            }
            _ => panic!("Expected AgentMessage variant"),
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
                assert_eq!(final_turn, false);
            }
            _ => panic!("Expected TurnCompleted variant"),
        }
    }

    #[test]
    fn test_event_serialization_warning() {
        let event = Event::Warning {
            message: "This is a warning".to_string(),
        };

        let serialized = serde_json::to_string(&event).unwrap();
        let deserialized: Event = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            Event::Warning { message } => {
                assert_eq!(message, "This is a warning");
            }
            _ => panic!("Expected Warning variant"),
        }
    }

    #[test]
    fn test_event_serialization_error() {
        let event = Event::Error {
            sub_id: Uuid::new_v4(),
            error: "An error occurred".to_string(),
        };

        let serialized = serde_json::to_string(&event).unwrap();
        let deserialized: Event = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            Event::Error { error, .. } => {
                assert_eq!(error, "An error occurred");
            }
            _ => panic!("Expected Error variant"),
        }
    }

    #[test]
    fn test_task_result_serialization() {
        let success_result = TaskResult::Success("Task successful".to_string());
        let serialized = serde_json::to_string(&success_result).unwrap();
        let deserialized: TaskResult = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            TaskResult::Success(msg) => assert_eq!(msg, "Task successful"),
            _ => panic!("Expected Success result"),
        }

        let value_result = TaskResult::Value(json!({"key": "value"}));
        let serialized = serde_json::to_string(&value_result).unwrap();
        let deserialized: TaskResult = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            TaskResult::Value(val) => {
                assert_eq!(val, json!({"key": "value"}));
            }
            _ => panic!("Expected Value result"),
        }

        let failure_result = TaskResult::Failure("Task failed".to_string());
        let serialized = serde_json::to_string(&failure_result).unwrap();
        let deserialized: TaskResult = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            TaskResult::Failure(msg) => assert_eq!(msg, "Task failed"),
            _ => panic!("Expected Failure result"),
        }

        let aborted_result = TaskResult::Aborted;
        let serialized = serde_json::to_string(&aborted_result).unwrap();
        let deserialized: TaskResult = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            TaskResult::Aborted => {}
            _ => panic!("Expected Aborted result"),
        }
    }

    #[test]
    fn test_agent_message_serialization() {
        let message = AgentMessage {
            content: "Test message".to_string(),
            chat_messages: None,
        };

        let serialized = serde_json::to_string(&message).unwrap();
        let deserialized: AgentMessage = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.content, "Test message");
        assert!(deserialized.chat_messages.is_none());
    }

    #[test]
    fn test_agent_message_with_chat_messages() {
        use autoagents_llm::chat::{ChatMessage, ChatRole, MessageType};

        let chat_msg = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "Hello".to_string(),
        };

        let message = AgentMessage {
            content: "Agent response".to_string(),
            chat_messages: Some(vec![chat_msg]),
        };

        let serialized = serde_json::to_string(&message).unwrap();
        let deserialized: AgentMessage = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.content, "Agent response");
        assert!(deserialized.chat_messages.is_some());
        assert_eq!(deserialized.chat_messages.unwrap().len(), 1);
    }

    #[test]
    fn test_event_debug_format() {
        let event = Event::NewTask {
            sub_id: Uuid::new_v4(),
            agent_id: Some(Uuid::new_v4()),
            prompt: "Debug test".to_string(),
        };

        let debug_str = format!("{:?}", event);
        assert!(debug_str.contains("NewTask"));
        assert!(debug_str.contains("Debug test"));
    }

    #[test]
    fn test_task_result_debug_format() {
        let result = TaskResult::Success("Debug success".to_string());
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("Success"));
        assert!(debug_str.contains("Debug success"));
    }

    #[test]
    fn test_agent_message_debug_format() {
        let message = AgentMessage {
            content: "Debug message".to_string(),
            chat_messages: None,
        };

        let debug_str = format!("{:?}", message);
        assert!(debug_str.contains("AgentMessage"));
        assert!(debug_str.contains("Debug message"));
    }

    #[test]
    fn test_event_clone() {
        let event = Event::Warning {
            message: "Test warning".to_string(),
        };

        let cloned = event.clone();
        match (event, cloned) {
            (Event::Warning { message: msg1 }, Event::Warning { message: msg2 }) => {
                assert_eq!(msg1, msg2);
            }
            _ => panic!("Clone failed"),
        }
    }

    #[test]
    fn test_task_result_clone() {
        let result = TaskResult::Success("Test success".to_string());
        let cloned = result.clone();

        match (result, cloned) {
            (TaskResult::Success(msg1), TaskResult::Success(msg2)) => {
                assert_eq!(msg1, msg2);
            }
            _ => panic!("Clone failed"),
        }
    }

    #[test]
    fn test_agent_message_clone() {
        let message = AgentMessage {
            content: "Test content".to_string(),
            chat_messages: None,
        };

        let cloned = message.clone();
        assert_eq!(message.content, cloned.content);
        assert_eq!(
            message.chat_messages.is_none(),
            cloned.chat_messages.is_none()
        );
    }

    #[test]
    fn test_uuid_types() {
        let submission_id: SubmissionId = Uuid::new_v4();
        let agent_id: AgentID = Uuid::new_v4();
        let session_id: SessionId = Uuid::new_v4();
        let event_id: EventId = Uuid::new_v4();

        // Test that all UUID types can be used interchangeably
        assert_ne!(submission_id, agent_id);
        assert_ne!(session_id, event_id);
    }

    #[test]
    fn test_event_with_all_uuid_types() {
        let sub_id: SubmissionId = Uuid::new_v4();
        let agent_id: AgentID = Uuid::new_v4();

        let event = Event::NewTask {
            sub_id,
            agent_id: Some(agent_id),
            prompt: "UUID test".to_string(),
        };

        // Should serialize and deserialize properly
        let serialized = serde_json::to_string(&event).unwrap();
        let _deserialized: Event = serde_json::from_str(&serialized).unwrap();
    }

    #[test]
    fn test_large_event_serialization() {
        let large_prompt = "x".repeat(10000);
        let event = Event::NewTask {
            sub_id: Uuid::new_v4(),
            agent_id: Some(Uuid::new_v4()),
            prompt: large_prompt.clone(),
        };

        let serialized = serde_json::to_string(&event).unwrap();
        let deserialized: Event = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            Event::NewTask { prompt, .. } => {
                assert_eq!(prompt, large_prompt);
            }
            _ => panic!("Expected NewTask variant"),
        }
    }

    #[test]
    fn test_event_with_empty_strings() {
        let event = Event::Warning {
            message: String::new(),
        };

        let serialized = serde_json::to_string(&event).unwrap();
        let deserialized: Event = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            Event::Warning { message } => {
                assert!(message.is_empty());
            }
            _ => panic!("Expected Warning variant"),
        }
    }
}
