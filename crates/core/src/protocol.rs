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

    /// A token has been streamed from the LLM
    Token(String),
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
