use autoagents_derive::ToolInput;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, ToolInput, Debug)]
struct BadChoiceArgs {
    #[input(description = "Mode", choice = [1, 2])]
    mode: String,
}

fn main() {}
