use autoagents_derive::AgentOutput;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, AgentOutput)]
struct BadOutput {
    value: i64,
}

fn main() {}
