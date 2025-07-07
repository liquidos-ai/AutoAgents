use proc_macro::TokenStream;
use quote::quote;
use strum::{Display, EnumString};
use syn::{
    bracketed, parse::Parse, parse_macro_input, punctuated::Punctuated, Ident, ItemStruct, LitStr,
    Token,
};

pub(crate) struct AgentAttributes {
    pub(crate) name: LitStr,
    pub(crate) prompt: LitStr,
    pub(crate) tools: Option<Vec<Ident>>,
}

#[derive(EnumString, Display)]
pub(crate) enum AgentAttributeKeys {
    #[strum(serialize = "name")]
    Name,
    #[strum(serialize = "prompt")]
    Prompt,
    #[strum(serialize = "tools")]
    Tools,
    Unknown(String),
}

impl From<Ident> for AgentAttributeKeys {
    fn from(value: Ident) -> Self {
        match value.to_string().as_str() {
            "name" => Self::Name,
            "prompt" => Self::Prompt,
            "tools" => Self::Tools,
            other => Self::Unknown(other.to_string()),
        }
    }
}

impl Parse for AgentAttributes {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut prompt = None;
        let mut tools = None;
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
                AgentAttributeKeys::Prompt => {
                    prompt = Some(input.parse::<LitStr>()?);
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
                        format!("Unexpected attribute key: {}", other),
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
            prompt: prompt.ok_or_else(|| {
                syn::Error::new(
                    input.span(),
                    format!("Missing attribute: {}", AgentAttributeKeys::Prompt),
                )
            })?,
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
        let agent_prompt = agent_attrs.prompt;
        let tool_idents = agent_attrs.tools.unwrap_or_default();

        let expanded = quote! {
            #input_struct

            impl AgentDeriveT for #struct_name {
                fn name(&self) -> &'static str {
                    #agent_name_literal
                }

                fn prompt(&self) -> &'static str {
                    #agent_prompt
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
