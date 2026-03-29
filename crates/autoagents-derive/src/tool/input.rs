use super::field::{Choice, FieldSchemaAttr};
use proc_macro::TokenStream;
use quote::quote;
use schemars::schema::{
    InstanceType, Metadata, ObjectValidation, RootSchema, Schema, SchemaObject, SingleOrVec,
};
use strum::{Display, EnumString};
use syn::{
    Attribute, Data, DataStruct, DeriveInput, Error, Field, GenericArgument, Ident, LitStr,
    PathArguments, Result, Type, parse_macro_input,
};

#[derive(EnumString, Display)]
enum InputAttrIdent {
    #[strum(serialize = "input")]
    Input,
}

#[derive(Debug, Default)]
pub(crate) struct InputParser {
    root_schema: RootSchema,
    ident: Option<Ident>,
}

impl InputParser {
    pub fn parse(&mut self, input: TokenStream) -> TokenStream {
        let input = parse_macro_input!(input as DeriveInput);
        let struct_ident = input.ident.clone();
        self.ident = Some(input.ident);

        if let Err(err) = self.parse_data(input.data) {
            return err.to_compile_error().into();
        }

        // Serialize only the SchemaObject to avoid the top-level $schema and definitions
        // for better compatibility with LLM providers like OpenAI.
        let serialized_data = serde_json::to_string(&self.root_schema.schema).unwrap();

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
                    "Union or Enums not yet supported!",
                ));
            }
        };
        Ok(())
    }

    fn parse_struct(&mut self, input: &DataStruct) -> Result<()> {
        self.root_schema.schema.instance_type =
            Some(SingleOrVec::Single(Box::new(InstanceType::Object)));
        self.root_schema.schema.object = Some(Box::new(ObjectValidation::default()));

        match &input.fields {
            syn::Fields::Named(fields) => {
                for field in fields.named.iter() {
                    let field_name = field
                        .ident
                        .as_ref()
                        .expect("Couldn't get the field name!")
                        .to_string();
                    let (schema, optional) = self.parse_field(field_name.clone(), field)?;

                    let object = self.root_schema.schema.object.as_mut().unwrap();
                    object.properties.insert(field_name.clone(), schema);
                    if !optional {
                        object.required.insert(field_name);
                    }
                }
            }
            _ => {
                return Err(Error::new(
                    proc_macro2::Span::call_site(),
                    "Union or Enums not yet supported!",
                ));
            }
        }
        Ok(())
    }

    fn parse_field(&mut self, _name: String, field: &Field) -> Result<(Schema, bool)> {
        // Determine JSON schema type from the Rust type.
        let (instance_type, optional) = self.get_json_type(&field.ty)?;

        let mut schema_obj = SchemaObject {
            instance_type: Some(SingleOrVec::Single(Box::new(instance_type))),
            ..Default::default()
        };

        let mut tool_property: Option<FieldSchemaAttr> = None;

        for attr in &field.attrs {
            if attr
                .path()
                .is_ident(InputAttrIdent::Input.to_string().as_str())
            {
                tool_property = Some(self.parse_macro_attributes(attr, &instance_type)?);
            }
        }

        if let Some(property) = tool_property {
            let mut metadata = Metadata::default();
            if let Some(desc) = property.description {
                metadata.description = Some(desc.value());
            }
            schema_obj.metadata = Some(Box::new(metadata));

            if let Some(choices) = property.choice {
                let enum_values = choices
                    .into_iter()
                    .map(|c| match c {
                        Choice::String(s) => Ok(serde_json::Value::String(s.value())),
                        Choice::Number(n) => {
                            let parsed = n.base10_parse::<i64>().map_err(|_| {
                                Error::new(
                                    n.span(),
                                    "Numeric `choice` value is out of range for i64",
                                )
                            })?;
                            Ok(serde_json::Value::Number(parsed.into()))
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                schema_obj.enum_values = Some(enum_values);
            }
        }

        Ok((Schema::Object(schema_obj), optional))
    }

    fn get_json_type(&mut self, field_type: &Type) -> Result<(InstanceType, bool)> {
        match field_type {
            Type::Path(path) => {
                let Some(segment) = path.path.segments.last() else {
                    return Err(Error::new(
                        proc_macro2::Span::call_site(),
                        "Invalid type path",
                    ));
                };

                if segment.ident == "Option"
                    && let PathArguments::AngleBracketed(args) = &segment.arguments
                    && let Some(GenericArgument::Type(inner)) = args.args.first()
                {
                    let (instance_type, _) = self.get_json_type(inner)?;
                    return Ok((instance_type, true));
                }

                if segment.ident == "Option" {
                    return Err(Error::new(
                        proc_macro2::Span::call_site(),
                        "Unsupported Option type",
                    ));
                }

                let instance_type = self.get_base_json_type(&segment.ident.to_string());
                Ok((instance_type, false))
            }
            Type::Reference(reference) => self.get_json_type(&reference.elem),
            Type::Group(group) => self.get_json_type(&group.elem),
            Type::Paren(paren) => self.get_json_type(&paren.elem),
            _ => Ok((InstanceType::String, false)),
        }
    }

    fn get_base_json_type(&self, type_str: &str) -> InstanceType {
        match type_str {
            "String" | "str" => InstanceType::String,
            "i32" | "u32" | "u8" | "i64" | "u16" | "usize" | "isize" => InstanceType::Integer,
            "f64" | "f32" => InstanceType::Number,
            "bool" => InstanceType::Boolean,
            "Vec" => InstanceType::Array,
            _ => InstanceType::String,
        }
    }

    fn parse_macro_attributes(
        &mut self,
        attribute: &Attribute,
        instance_type: &InstanceType,
    ) -> Result<FieldSchemaAttr> {
        let attributes = attribute.parse_args::<FieldSchemaAttr>()?;

        if let Some(ref enum_vals) = attributes.choice {
            let invalid_choice = enum_vals.iter().any(|c| {
                !matches!(
                    (c, instance_type),
                    (Choice::String(_), InstanceType::String)
                        | (
                            Choice::Number(_),
                            InstanceType::Integer | InstanceType::Number
                        )
                )
            });

            if invalid_choice {
                return Err(Error::new(
                    proc_macro2::Span::call_site(),
                    "Choices must be of the same type as the field",
                ));
            }
        }

        Ok(attributes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_input_struct_required_and_optional_fields() {
        let input: DeriveInput = syn::parse_str(
            r#"
            struct ToolArgs {
                #[input(description = "Id")]
                id: String,
                #[input(description = "Count")]
                count: Option<u32>,
                #[input(description = "Mode", choice = ["fast", "slow"])]
                mode: String,
            }
            "#,
        )
        .unwrap();

        let mut parser = InputParser::default();
        parser.parse_data(input.data).unwrap();

        let object = parser.root_schema.schema.object.as_ref().unwrap();
        assert!(object.properties.contains_key("id"));
        assert!(object.properties.contains_key("count"));
        assert!(object.properties.contains_key("mode"));

        assert!(object.required.contains("id"));
        assert!(!object.required.contains("count"));

        let mode_schema = object.properties.get("mode").unwrap();
        if let Schema::Object(obj) = mode_schema {
            assert_eq!(
                obj.instance_type,
                Some(SingleOrVec::Single(Box::new(InstanceType::String)))
            );
            assert_eq!(obj.enum_values.as_ref().unwrap().len(), 2);
        } else {
            panic!("Expected Schema::Object");
        }
    }

    #[test]
    fn verify_serialized_schema() {
        let input: DeriveInput = syn::parse_str(
            r#"
            struct ToolArgs {
                #[input(description = "The name")]
                name: String,
                #[input(description = "The age")]
                age: Option<u32>,
            }
            "#,
        )
        .unwrap();

        let mut parser = InputParser::default();
        parser
            .parse_struct(match &input.data {
                Data::Struct(s) => s,
                _ => panic!("Expected struct"),
            })
            .unwrap();

        let serialized_data = serde_json::to_string(&parser.root_schema.schema).unwrap();

        // The serialized schema should be valid JSON
        assert!(serialized_data.contains("\"type\":\"object\""));
        assert!(
            serialized_data.contains("\"name\":{\"description\":\"The name\",\"type\":\"string\"}")
        );
        assert!(
            serialized_data.contains("\"age\":{\"description\":\"The age\",\"type\":\"integer\"}")
        );
        assert!(serialized_data.contains("\"required\":[\"name\"]"));
        assert!(!serialized_data.contains("\"required\":[\"age\"]"));
    }
}
