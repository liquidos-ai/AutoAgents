use serde::{Deserialize, Serialize};
use autoagents_llm::chat::ChatMessage;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentProtocol {
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
}


/// Messages from the agent - used for A2A communication
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AgentMessage {
    /// The content of the message
    pub content: String,

    /// Optional chat messages for a full conversation history
    pub chat_messages: Option<Vec<ChatMessage>>,
}