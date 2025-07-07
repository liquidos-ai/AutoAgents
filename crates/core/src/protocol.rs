use autoagents_llm::chat::ChatMessage;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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

    /// A task has been completed
    TaskComplete {
        sub_id: SubmissionId,
        result: TaskResult,
    },

    /// A message from the agent to be presented to the user
    AgentMessage {
        sub_id: SubmissionId,
        message: AgentMessage,
    },

    /// A tool request from the agent
    ToolRequest {
        sub_id: SubmissionId,
        event_id: EventId,
        request: ToolRequest,
    },

    /// A response to a tool request
    ToolResponse {
        sub_id: SubmissionId,
        event_id: EventId,
        response: ToolResponse,
    },

    /// An input request for the user
    InputRequest {
        sub_id: SubmissionId,
        event_id: EventId,
        prompt: String,
    },

    /// A response to an input request
    InputResponse {
        sub_id: SubmissionId,
        event_id: EventId,
        input: String,
    },

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

/// A request to use a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolRequest {
    /// The name of the tool to use
    pub tool_name: String,

    /// The arguments to the tool
    pub arguments: serde_json::Value,

    /// Whether the tool needs approval
    pub needs_approval: bool,
}

/// A response from a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolResponse {
    /// The tool was approved and executed
    Success(serde_json::Value),

    /// The tool was denied
    Denied,

    /// The tool execution failed
    Failure(String),
}

/// Protocol commands for controlling agent execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Command {
    /// Submit a new task to an agent
    Submit {
        agent_id: Uuid,
        prompt: String,
        context: Option<HashMap<String, String>>,
    },

    /// Submit a new task with a specific ID
    SubmitWithId {
        sub_id: SubmissionId,
        agent_id: Uuid,
        prompt: String,
        context: Option<HashMap<String, String>>,
    },

    /// Approve a tool request
    ApproveTool {
        sub_id: SubmissionId,
        event_id: EventId,
    },

    /// Deny a tool request
    DenyTool {
        sub_id: SubmissionId,
        event_id: EventId,
    },

    /// Provide input for an input request
    ProvideInput {
        sub_id: SubmissionId,
        event_id: EventId,
        input: String,
    },

    /// Abort a task
    Abort { sub_id: SubmissionId },
}
