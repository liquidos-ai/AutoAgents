extern crate proc_macro;
use agent::{output::OutputParser, AgentParser};
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};
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

#[proc_macro_derive(AgentHooks)]
pub fn derive_agent_hooks(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    // Correctly handle generics
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let expanded = quote! {
        // bring async_trait in via absolute path to avoid needing use in consumer crate
        #[::autoagents::async_trait]
        impl #impl_generics ::autoagents::core::agent::AgentHooks for #name #ty_generics #where_clause {}
    };

    TokenStream::from(expanded)
}
