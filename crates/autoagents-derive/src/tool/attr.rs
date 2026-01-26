use strum::{Display, EnumString};
use syn::{
    Ident, LitStr, Result, Token, Type,
    parse::{Parse, ParseStream},
};

pub(crate) struct ToolAttributes {
    pub(crate) name: LitStr,
    pub(crate) description: LitStr,
    pub(crate) input: Type,
}

#[derive(EnumString, Display)]
pub(crate) enum ToolAttributeKeys {
    #[strum(serialize = "name")]
    Name,
    #[strum(serialize = "description")]
    Description,
    #[strum(serialize = "input")]
    Input,
    Unknown(String),
}

impl From<Ident> for ToolAttributeKeys {
    fn from(value: Ident) -> Self {
        match value.to_string().as_str() {
            "name" => Self::Name,
            "description" => Self::Description,
            "input" => Self::Input,
            other => Self::Unknown(other.to_string()),
        }
    }
}

impl Parse for ToolAttributes {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut name = None;
        let mut description = None;
        let mut args = None;
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            let key_span = key.span();
            let tool_attr_key: ToolAttributeKeys = key.into();
            // Move forward one token
            input.parse::<Token![=]>()?;

            match tool_attr_key {
                ToolAttributeKeys::Name => {
                    name = Some(input.parse::<LitStr>()?);
                }
                ToolAttributeKeys::Description => {
                    description = Some(input.parse::<LitStr>()?);
                }
                ToolAttributeKeys::Input => {
                    args = Some(input.parse::<Type>()?);
                }
                ToolAttributeKeys::Unknown(other) => {
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
        Ok(ToolAttributes {
            name: name.ok_or_else(|| {
                syn::Error::new(
                    input.span(),
                    format!("Missing attribute: {}", ToolAttributeKeys::Name),
                )
            })?,
            description: description.ok_or_else(|| {
                syn::Error::new(
                    input.span(),
                    format!("Missing attribute: {}", ToolAttributeKeys::Description),
                )
            })?,
            input: args.ok_or_else(|| {
                syn::Error::new(
                    input.span(),
                    format!("Missing attribute: {}", ToolAttributeKeys::Input),
                )
            })?,
        })
    }
}
