use autoagents_derive::ToolInput;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, ToolInput, Debug)]
struct OrphanArgs {
    #[input(description = "A value")]
    value: String,
}

fn main() {}
