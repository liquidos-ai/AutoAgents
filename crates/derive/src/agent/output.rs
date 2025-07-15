use super::super::tool::{field::FieldSchemaAttr, json::JsonType};
use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use serde::Serialize;
use std::collections::HashMap;
use strum::{Display, EnumString};
use syn::{
    parse_macro_input, Attribute, Data, DataStruct, DeriveInput, Error, Field, Ident, LitStr,
    Result, Type,
};

#[derive(EnumString, Display)]
enum OutputAttrIdent {
    #[strum(serialize = "output")]
    Output,
}

#[derive(Debug, Serialize)]
pub(crate) struct OutputSchemaProperty {
    #[serde(rename = "type")]
    _type: String,
    description: Option<String>,
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    _enum: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Default)]
pub(crate) struct OutputSchema {
    #[serde(rename = "type")]
    _type: String,
    #[serde(default)]
    properties: HashMap<String, OutputSchemaProperty>,
    #[serde(default)]
    required: Vec<String>,
}

#[derive(Debug, Serialize, Default)]
pub(crate) struct StructuredOutputFormat {
    name: String,
    description: Option<String>,
    #[serde(default)]
    schema: OutputSchema,
    strict: Option<bool>,
}

#[derive(Debug, Default)]
pub(crate) struct OutputParser {
    output_data: StructuredOutputFormat,
    ident: Option<Ident>,
}

impl OutputParser {
    pub fn parse(&mut self, input: TokenStream) -> TokenStream {
        let input = parse_macro_input!(input as DeriveInput);
        let struct_ident = input.ident.clone();
        self.ident = Some(input.ident);

        // Initialize the output data with the struct name
        self.output_data.name = struct_ident.to_string();
        self.output_data.schema._type = JsonType::Object.to_string();

        // Parse the struct attributes for description and strict mode
        self.parse_struct_attributes(&input.attrs);

        // Parse the data structure
        self.parse_data(input.data).unwrap();

        let serialized_data = serde_json::to_string(&self.output_data).unwrap();
        let schema_literal = LitStr::new(&serialized_data, struct_ident.span());

        let expanded = quote! {
            impl AgentOutputT for #struct_ident {
                fn output_schema() -> &'static str {
                    #schema_literal
                }

                fn structured_output_format() -> serde_json::Value {
                    let schema_str = Self::output_schema();
                    serde_json::from_str(schema_str)
                        .expect("Failed to parse output schema")
                }
            }
        };
        TokenStream::from(expanded)
    }

    fn parse_struct_attributes(&mut self, attrs: &[Attribute]) {
        for attr in attrs {
            if attr.path().is_ident("doc") {
                // Extract documentation comments as description
                // Doc comments are stored as #[doc = "content"] attributes
                if let syn::Meta::NameValue(meta) = &attr.meta {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(lit_str),
                        ..
                    }) = &meta.value
                    {
                        let doc_value = lit_str.value().trim().to_string();
                        if !doc_value.is_empty() {
                            self.output_data.description = Some(doc_value);
                        }
                    }
                }
            } else if attr.path().is_ident("strict") {
                // Parse strict attribute
                if let Ok(strict_value) = attr.parse_args::<syn::LitBool>() {
                    self.output_data.strict = Some(strict_value.value);
                }
            }
        }
    }

    fn parse_data(&mut self, input: Data) -> Result<()> {
        match &input {
            Data::Struct(struct_data) => self.parse_struct(struct_data)?,
            _ => {
                return Err(Error::new(
                    proc_macro2::Span::call_site(),
                    "Union or Enums not yet supported!",
                ));
            }
        };
        Ok(())
    }

    fn parse_struct(&mut self, input: &DataStruct) -> Result<()> {
        match &input.fields {
            syn::Fields::Named(fields) => {
                for field in fields.named.iter() {
                    let field_name = field
                        .ident
                        .as_ref()
                        .expect("Couldn't get the field name!")
                        .to_string();
                    let output_property = self.parse_field(field_name.clone(), field)?;
                    self.output_data
                        .schema
                        .properties
                        .insert(field_name, output_property);
                }
            }
            _ => {
                return Err(Error::new(
                    proc_macro2::Span::call_site(),
                    "Tuple or Unit structs not yet supported!",
                ));
            }
        }
        Ok(())
    }

    fn parse_field(&mut self, name: String, field: &Field) -> Result<OutputSchemaProperty> {
        // Check if field is optional (wrapped in Option<T>)
        let (is_optional, inner_type) = self.extract_option_type(&field.ty);

        if !is_optional {
            self.output_data.schema.required.push(name.clone());
        }

        let json_type = self.get_json_type(inner_type.unwrap_or(&field.ty))?;
        let mut field_schema: Option<FieldSchemaAttr> = None;

        // Parse field attributes
        for attr in &field.attrs {
            if attr
                .path()
                .is_ident(OutputAttrIdent::Output.to_string().as_str())
            {
                field_schema = Some(self.parse_field_attributes(attr, &json_type)?);
            }
        }

        if let Some(schema) = field_schema {
            Ok(OutputSchemaProperty {
                _type: json_type.to_string(),
                description: schema.description.map(|lit| lit.value()),
                _enum: schema
                    .choice
                    .map(|choices| choices.iter().map(|choice| choice.to_string()).collect()),
            })
        } else {
            // Default property without attributes
            Ok(OutputSchemaProperty {
                _type: json_type.to_string(),
                description: None,
                _enum: None,
            })
        }
    }

    fn extract_option_type<'a>(&self, ty: &'a Type) -> (bool, Option<&'a Type>) {
        if let Type::Path(type_path) = ty {
            if let Some(segment) = type_path.path.segments.last() {
                if segment.ident == "Option" {
                    if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                        if let Some(syn::GenericArgument::Type(inner_type)) = args.args.first() {
                            return (true, Some(inner_type));
                        }
                    }
                }
            }
        }
        (false, None)
    }

    fn get_json_type(&self, field_type: &Type) -> Result<JsonType> {
        let type_str = field_type.to_token_stream().to_string();
        let json_type = match type_str.as_str() {
            "String" | "str" => JsonType::String,
            "i32" | "u32" | "f64" | "f32" | "u8" | "i64" | "u64" | "i16" | "u16" | "isize"
            | "usize" => JsonType::Number,
            "bool" => JsonType::Boolean,
            _ => {
                // Check if it's a Vec<T>
                if type_str.starts_with("Vec <") {
                    return Ok(JsonType::Array);
                }
                return Err(Error::new(
                    proc_macro2::Span::call_site(),
                    format!("Unsupported data type: {type_str}"),
                ));
            }
        };
        Ok(json_type)
    }

    fn parse_field_attributes(
        &self,
        attribute: &Attribute,
        field_type: &JsonType,
    ) -> Result<FieldSchemaAttr> {
        let attributes = attribute.parse_args::<FieldSchemaAttr>()?;

        // Validate that enum choices match the field type
        if let Some(ref enum_vals) = attributes.choice {
            let invalid_choice = enum_vals.iter().find(|c| {
                match (c, field_type) {
                    (super::super::tool::field::Choice::String(_), JsonType::String) => false,
                    (super::super::tool::field::Choice::Number(_), JsonType::Number) => false,
                    _ => true, // Invalid case
                }
            });

            if invalid_choice.is_some() {
                return Err(Error::new(
                    proc_macro2::Span::call_site(),
                    "Enum choices must match the field type",
                ));
            }
        }

        Ok(attributes)
    }
}
