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

#[cfg(test)]
mod tests {
    use super::*;
    use syn::Type;

    #[test]
    fn test_tool_attributes_parse_success() {
        let attrs: ToolAttributes =
            syn::parse_str(r#"name = "lookup", description = "Look up values", input = String"#)
                .expect("expected attributes to parse");

        assert_eq!(attrs.name.value(), "lookup");
        assert_eq!(attrs.description.value(), "Look up values");

        match attrs.input {
            Type::Path(path) => {
                let ident = &path.path.segments.last().unwrap().ident;
                assert_eq!(ident, "String");
            }
            _ => panic!("Unexpected type parsed"),
        }
    }

    #[test]
    fn test_tool_attributes_missing_field() {
        let err = match syn::parse_str::<ToolAttributes>(r#"name = "x", input = String"#) {
            Ok(_) => panic!("expected missing description error"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("Missing attribute: description"));
    }

    #[test]
    fn test_tool_attributes_unknown_key() {
        let err = match syn::parse_str::<ToolAttributes>(
            r#"name = "x", description = "y", input = String, extra = "nope""#,
        ) {
            Ok(_) => panic!("expected unknown key error"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("Unexpected attribute key"));
    }
}
