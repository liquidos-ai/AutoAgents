use autoagents_llm::chat::StructuredOutputFormat;
use crate::protocol::ActorID;

#[derive(Default, Clone)]
pub struct AgentConfig {
    /// The agent's name
    pub name: String,
    /// The agent's description
    pub description: String,
    /// The Agent ID
    pub id: ActorID,
    /// The output schema for the agent
    pub output_schema: Option<StructuredOutputFormat>,
}