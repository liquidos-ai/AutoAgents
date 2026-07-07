use autoagents_derive::AgentOutput;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Agent output schemas currently support scalar primitives, `Option<T>`, and
// `Vec<T>`-shaped fields. Unsupported containers should fail clearly.
#[derive(Debug, Serialize, Deserialize, AgentOutput)]
struct UnsupportedOutput {
    #[output(description = "Arbitrary values")]
    values: HashMap<String, String>,
}

fn main() {}
