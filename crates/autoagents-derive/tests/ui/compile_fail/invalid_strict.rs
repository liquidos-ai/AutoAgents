use autoagents_derive::AgentOutput;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, AgentOutput)]
#[strict("not-a-bool")]
struct BadStrictOutput {
    #[output(description = "Value")]
    value: i64,
}

fn main() {}
