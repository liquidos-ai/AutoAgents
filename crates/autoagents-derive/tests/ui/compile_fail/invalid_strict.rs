use autoagents_derive::AgentOutput;
use serde::{Deserialize, Serialize};

// `#[strict]` accepts only a boolean literal to keep generated schema metadata
// deterministic.
#[derive(Debug, Serialize, Deserialize, AgentOutput)]
#[strict("not-a-bool")]
struct BadStrictOutput {
    #[output(description = "Value")]
    value: i64,
}

fn main() {}
