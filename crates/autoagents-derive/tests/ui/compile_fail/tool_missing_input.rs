use autoagents_derive::tool;

// `input` is required because generated tools always expose an args schema.
#[tool(name = "missing-input", description = "Tool without an input type")]
struct MissingInputTool;

fn main() {}
