use strum::{Display, EnumString};
use syn::{
    parse::{Parse, ParseBuffer, ParseStream},
    punctuated::Punctuated,
    Ident, Lit, LitInt, LitStr, Result, Token,
};

pub(crate) enum Choice {
    String(LitStr),
    Number(LitInt),
}

#[allow(clippy::inherent_to_string)]
impl Choice {
    pub(crate) fn to_string(&self) -> String {
        match self {
            Self::Number(n) => n.to_string(),
            Self::String(s) => s.value(),
        }
    }
}

impl Parse for Choice {
    fn parse(input: ParseStream) -> Result<Self> {
        // Otherwise, parse a single literal.
        let lit: Lit = input.parse()?;
        match lit {
            Lit::Str(lit_str) => Ok(Choice::String(lit_str)),
            Lit::Int(lit_int) => Ok(Choice::Number(lit_int)),
            _ => Err(input.error("expected a string literal or an integer literal")),
        }
    }
}

pub(crate) struct FieldSchemaAttr {
    pub(crate) description: Option<LitStr>,
    pub(crate) choice: Option<Vec<Choice>>,
}

#[derive(EnumString, Display)]
pub(crate) enum FieldAttributeKeys {
    #[strum(serialize = "description")]
    Description,
    #[strum(serialize = "choice")]
    Choice,
    Unknown(String),
}

impl From<Ident> for FieldAttributeKeys {
    fn from(value: Ident) -> Self {
        match value.to_string().as_str() {
            "description" => Self::Description,
            "choice" => Self::Choice,
            other => Self::Unknown(other.to_string()),
        }
    }
}

impl FieldSchemaAttr {
    fn parse_choice(choice: &ParseBuffer) -> Result<Vec<Choice>> {
        let content;
        let _bracket_token = syn::bracketed!(content in choice);
        let choices: Punctuated<Choice, Token![,]> =
            content.parse_terminated(Choice::parse, Token![,])?;
        Ok(choices.into_iter().collect())
    }
}

impl Parse for FieldSchemaAttr {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut description = None;
        let mut choice: Vec<Choice> = vec![];
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            let key_span = key.span();
            let field_attr_key: FieldAttributeKeys = key.into();
            //Move forward one token
            input.parse::<Token![=]>()?;
            match field_attr_key {
                FieldAttributeKeys::Description => {
                    description = Some(input.parse()?);
                }
                FieldAttributeKeys::Choice => {
                    choice = Self::parse_choice(input)?;
                }
                FieldAttributeKeys::Unknown(other) => {
                    return Err(syn::Error::new(
                        key_span,
                        format!("Unexpected field attribute key: {other}"),
                    ))
                }
            }

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(FieldSchemaAttr {
            description,
            choice: if !choice.is_empty() {
                Some(choice)
            } else {
                None
            },
        })
    }
}
