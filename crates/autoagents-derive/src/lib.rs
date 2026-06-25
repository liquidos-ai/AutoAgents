extern crate proc_macro;
use agent::{AgentParser, output::OutputParser};
use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse_macro_input};
use tool::{ToolParser, input::InputParser};

mod agent;
mod resolve;
mod schema_emit;
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

    let (core, async_trait) = match resolve::resolve_agent_hooks_paths() {
        Ok(paths) => paths,
        Err(err) => return err.to_compile_error().into(),
    };
    let core = &core;
    let async_trait = &async_trait;

    // Correctly handle generics
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let expanded = quote! {
        #[#async_trait]
        impl #impl_generics #core::agent::AgentHooks for #name #ty_generics #where_clause {}
    };

    TokenStream::from(expanded)
}
