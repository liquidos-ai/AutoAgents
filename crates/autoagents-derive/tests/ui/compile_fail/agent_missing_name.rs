use autoagents_derive::agent;

// `name` is required because it becomes the generated `AgentDeriveT::name`.
#[agent(description = "Agent without a name")]
struct MissingNameAgent;

fn main() {}
