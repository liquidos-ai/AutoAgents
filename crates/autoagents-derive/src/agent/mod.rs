use proc_macro::TokenStream;
use quote::quote;
use strum::{Display, EnumString};
use syn::{
    Expr, Ident, ItemStruct, LitStr, Token, Type, bracketed, parse::Parse, parse_macro_input,
    punctuated::Punctuated,
};

pub(crate) mod output;

pub(crate) struct AgentAttributes {
    pub(crate) name: LitStr,
    pub(crate) description: LitStr,
    pub(crate) tools: Option<Vec<Expr>>,
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
                    // Parse a bracketed list of tool expressions
                    let content;
                    bracketed!(content in input);
                    let punctuated_exprs: Punctuated<Expr, Token![,]> =
                        content.parse_terminated(Expr::parse, Token![,])?;
                    tools = Some(punctuated_exprs.into_iter().collect::<Vec<Expr>>());
                }
                AgentAttributeKeys::Unknown(other) => {
                    return Err(syn::Error::new(
                        key_span,
                        format!("Unexpected attribute key: {other}"),
                    ));
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
        let AgentAttributes {
            name: agent_name_literal,
            description: agent_description,
            tools,
            output: output_type,
        } = agent_attrs;
        let tool_initializers = tools
            .unwrap_or_default()
            .into_iter()
            .map(|tool_expr| match tool_expr {
                Expr::Path(expr_path) => quote! { #expr_path {} },
                other => quote! { #other },
            })
            .collect::<Vec<_>>();

        let quoted_output_type = match &output_type {
            Some(output_ty) => quote! { #output_ty },
            None => quote! { String },
        };

        let output_schema_impl = match &output_type {
            Some(output_ty) => {
                quote! {
                    fn output_schema(&self) -> Option<serde_json::Value> {
                        Some(<#output_ty>::structured_output_format())
                    }
                }
            }
            None => {
                quote! {
                    fn output_schema(&self) -> Option<serde_json::Value> {
                        None
                    }
                }
            }
        };

        let expanded = quote! {
            #input_struct

            impl autoagents::core::agent::AgentDeriveT for #struct_name {
                type Output = #quoted_output_type;

                fn name(&self) -> &'static str {
                    #agent_name_literal
                }

                #output_schema_impl

                fn description(&self) -> &'static str {
                    #agent_description
                }

                fn tools(&self) -> Vec<Box<dyn autoagents::core::tool::ToolT>> {
                    vec![
                        #(
                            Box::new(#tool_initializers) as Box<dyn autoagents::core::tool::ToolT>
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
