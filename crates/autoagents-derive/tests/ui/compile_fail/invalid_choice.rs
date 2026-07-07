use autoagents_derive::ToolInput;
use serde::{Deserialize, Serialize};

// `choice` values must match the JSON type derived from the Rust field type.
#[derive(Serialize, Deserialize, ToolInput, Debug)]
struct BadChoiceArgs {
    #[input(description = "Mode", choice = [1, 2])]
    mode: String,
}

fn main() {}
