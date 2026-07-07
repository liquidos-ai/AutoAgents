use autoagents_derive::AgentOutput;
use serde::{Deserialize, Serialize};

// At least one field must opt in with `#[output]`; otherwise the structured
// output schema would be undocumented from the macro user's perspective.
#[derive(Debug, Serialize, Deserialize, AgentOutput)]
struct BadOutput {
    value: i64,
}

fn main() {}
