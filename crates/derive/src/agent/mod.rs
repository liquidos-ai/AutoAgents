use proc_macro::TokenStream;
use quote::quote;
use strum::{Display, EnumString};
use syn::{
    bracketed, parse::Parse, parse_macro_input, punctuated::Punctuated, Ident, ItemStruct, LitStr,
    Token, Type,
};

pub(crate) mod output;

pub(crate) struct AgentAttributes {
    pub(crate) name: LitStr,
    pub(crate) description: LitStr,
    pub(crate) tools: Option<Vec<Ident>>,
    pub(crate) output: Option<Type>,
}

#[derive(EnumString, Display)]
pub(crate) enum AgentAttributeKeys {
    #[strum(serialize = "name")]
    Name,
    #[strum(serialize = "description")]
    Description,
    #[strum(serialize = "tools")]
    Tools,
    #[strum(serialize = "output")]
    Output,
    Unknown(String),
}

impl From<Ident> for AgentAttributeKeys {
    fn from(value: Ident) -> Self {
        match value.to_string().as_str() {
            "name" => Self::Name,
            "description" => Self::Description,
            "tools" => Self::Tools,
            "output" => Self::Output,
            other => Self::Unknown(other.to_string()),
        }
    }
}

impl Parse for AgentAttributes {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut description = None;
        let mut tools = None;
        let mut output = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            let key_span = key.span();
            let agent_attr_key: AgentAttributeKeys = key.into();
            // Consume the `=`
            input.parse::<Token![=]>()?;

            match agent_attr_key {
                AgentAttributeKeys::Name => {
                    name = Some(input.parse::<LitStr>()?);
                }
                AgentAttributeKeys::Description => {
                    description = Some(input.parse::<LitStr>()?);
                }
                AgentAttributeKeys::Output => {
                    output = Some(input.parse::<Type>()?);
                }
                AgentAttributeKeys::Tools => {
                    // Parse a bracketed list of identifiers
                    let content;
                    bracketed!(content in input);
                    let punctuated_idents: Punctuated<Ident, Token![,]> =
                        content.parse_terminated(Ident::parse, Token![,])?;
                    tools = Some(punctuated_idents.into_iter().collect::<Vec<Ident>>());
                }
                AgentAttributeKeys::Unknown(other) => {
                    return Err(syn::Error::new(
                        key_span,
                        format!("Unexpected attribute key: {other}"),
                    ))
                }
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }
        Ok(AgentAttributes {
            name: name.ok_or_else(|| {
                syn::Error::new(
                    input.span(),
                    format!("Missing attribute: {}", AgentAttributeKeys::Name),
                )
            })?,
            description: description.ok_or_else(|| {
                syn::Error::new(
                    input.span(),
                    format!("Missing attribute: {}", AgentAttributeKeys::Description),
                )
            })?,
            output,
            tools,
        })
    }
}

#[derive(Debug, Default)]
pub(crate) struct AgentParser {}

impl AgentParser {
    pub fn parse(&self, attr: TokenStream, item: TokenStream) -> TokenStream {
        let agent_attrs = parse_macro_input!(attr as AgentAttributes);
        let input_struct = parse_macro_input!(item as ItemStruct);
        let struct_name = &input_struct.ident;
        let agent_name_literal = agent_attrs.name;
        let agent_description = agent_attrs.description;
        let tool_idents = agent_attrs.tools.unwrap_or_default();
        let output_type = agent_attrs.output;

        let quoted_output_type = match &output_type {
            Some(output_ty) => quote! { #output_ty },
            None => quote! { String },
        };

        let output_schema_impl = match &output_type {
            Some(output_ty) => {
                quote! {
                    fn output_schema(&self) -> Option<Value> {
                        Some(<#output_ty>::structured_output_format())
                    }
                }
            }
            None => {
                quote! {
                    fn output_schema(&self) -> Option<Value> {
                        None
                    }
                }
            }
        };

        let expanded = quote! {
            #input_struct

            impl AgentDeriveT for #struct_name {
                type Output = #quoted_output_type;

                fn name(&self) -> &'static str {
                    #agent_name_literal
                }

                #output_schema_impl

                fn description(&self) -> &'static str {
                    #agent_description
                }

                fn tools(&self) -> Vec<Box<dyn ToolT>> {
                    vec![
                        #(
                            Box::new(#tool_idents{}) as Box<dyn ToolT>
                        ),*
                    ]
                }
            }

            impl std::fmt::Debug for #struct_name {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "{}", #agent_name_literal)
                }
            }
        };
        expanded.into()
    }
}
