use super::{
    field::{Choice, FieldSchemaAttr},
    json::JsonType,
};
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
enum InputAttrIdent {
    #[strum(serialize = "input")]
    Input,
}

#[derive(Debug, Serialize)]
pub(crate) struct InputToolProperty {
    description: Option<String>,
    #[serde(rename = "type")]
    _type: String,
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    _enum: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Default)]
pub(crate) struct InputToolParseData {
    properties: HashMap<String, InputToolProperty>,
    required: Vec<String>,
    #[serde(rename = "type")]
    arg_type: String,
}

impl InputToolParseData {
    fn add_required_field(&mut self, field: String) {
        self.required.push(field);
    }

    fn add_property(&mut self, name: String, property: InputToolProperty) {
        self.properties.insert(name, property);
    }

    fn set_type(&mut self, arg_type: String) {
        self.arg_type = arg_type;
    }
}

#[derive(Debug, Default)]
pub(crate) struct InputParser {
    tool_parse_data: InputToolParseData,
    ident: Option<Ident>,
}

impl InputParser {
    pub fn parse(&mut self, input: TokenStream) -> TokenStream {
        let input = parse_macro_input!(input as DeriveInput);
        let struct_ident = input.ident.clone();
        self.ident = Some(input.ident);

        //Safe to unwrap as if it fails here, It will should panic
        self.parse_data(input.data).unwrap();

        let serialized_data =
            serde_json::to_string::<InputToolParseData>(&self.tool_parse_data).unwrap();

        let schema_literal = LitStr::new(&serialized_data, struct_ident.span());
        let expanded = quote! {
            impl ToolInputT for #struct_ident {
                fn io_schema() -> &'static str {
                    #schema_literal
                }
            }
        };
        TokenStream::from(expanded)
    }

    fn parse_data(&mut self, input: Data) -> Result<()> {
        match &input {
            Data::Struct(struct_data) => self.parse_struct(struct_data)?,
            _ => {
                return Err(Error::new(
                    proc_macro2::Span::call_site(),
                    "Uninon or Enums not yet supported!",
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
                    let input_property = self.parse_field(field_name.clone(), field)?;
                    self.tool_parse_data
                        .add_property(field_name, input_property);
                }
            }
            _ => {
                return Err(Error::new(
                    proc_macro2::Span::call_site(),
                    "Uninon or Enums not yet supported!",
                ));
            }
        }
        self.tool_parse_data.set_type(JsonType::Object.to_string());
        Ok(())
    }

    fn parse_field(&mut self, name: String, field: &Field) -> Result<InputToolProperty> {
        //Add the field to required
        self.tool_parse_data.add_required_field(name);

        // Determine JSON schema type from the Rust type.
        let json_type = self.get_json_type(&field.ty)?;
        let mut tool_property: Option<FieldSchemaAttr> = None;

        //Currently handling Input ident only
        for attr in &field.attrs {
            if attr
                .path()
                .is_ident(InputAttrIdent::Input.to_string().as_str())
            {
                tool_property = Some(self.parse_macro_attributes(attr, &json_type)?);
            }
        }

        if let Some(property) = tool_property {
            Ok(InputToolProperty {
                description: property
                    .description
                    .map_or_else(|| None, |f| Some(f.value())),
                _enum: property.choice.map_or_else(
                    || None,
                    |f| Some(f.iter().map(|f| f.to_string()).collect::<Vec<String>>()),
                ),
                _type: json_type.to_string(),
            })
        } else {
            Err(Error::new(
                proc_macro2::Span::call_site(),
                "Coudn't Create the tool arg property",
            ))
        }
    }

    fn get_json_type(&mut self, field_type: &Type) -> Result<JsonType> {
        let type_str = field_type.to_token_stream().clone();
        let type_str = type_str.to_string();
        let json_type: JsonType = match type_str.as_str() {
            "String" => JsonType::String,
            "i32" | "u32" | "f64" | "f32" | "u8" | "i64" => JsonType::Number,
            "bool" => JsonType::Boolean,
            _ => {
                return Err(Error::new(
                    proc_macro2::Span::call_site(),
                    "Unsupported Data Type",
                ));
            }
        };
        Ok(json_type)
    }

    fn parse_macro_attributes(
        &mut self,
        attribute: &Attribute,
        field_type: &JsonType,
    ) -> Result<FieldSchemaAttr> {
        let attributes = attribute.parse_args::<FieldSchemaAttr>()?;

        if let Some(ref enum_vals) = attributes.choice {
            let invalid_choice = enum_vals.iter().find(|c| match (c, field_type) {
                (Choice::String(_), JsonType::String) => false,
                (Choice::Number(_), JsonType::Number) => false,
                _ => true, // Invalid case
            });

            if invalid_choice.is_some() {
                return Err(Error::new(
                    proc_macro2::Span::call_site(),
                    "Choices must be of the same type as the field",
                ));
            }
        }

        Ok(attributes)
    }
}
