use autoagents_core::tool::ToolInputT;
use autoagents_derive::ToolInput;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, ToolInput, Debug)]
struct Args {
    #[input(description = "A value")]
    value: String,
}

fn main() {
    let _ = Args::io_schema();
}
