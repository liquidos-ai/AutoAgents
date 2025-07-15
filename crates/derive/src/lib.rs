extern crate proc_macro;
use agent::{output::OutputParser, AgentParser};
use proc_macro::TokenStream;
use tool::{input::InputParser, ToolParser};
mod agent;
mod tool;

#[proc_macro_derive(ToolInput, attributes(input))]
pub fn input(input: TokenStream) -> TokenStream {
    InputParser::default().parse(input)
}

#[proc_macro_derive(AgentOutput, attributes(output, strict))]
pub fn agent_output(input: TokenStream) -> TokenStream {
    OutputParser::default().parse(input)
}

#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    ToolParser::default().parse(attr, item)
}

#[proc_macro_attribute]
pub fn agent(attr: TokenStream, item: TokenStream) -> TokenStream {
    AgentParser::default().parse(attr, item)
}
