mod attr;
pub(crate) mod field;
pub(crate) mod input;
pub(crate) mod json;
use attr::ToolAttributes;
use proc_macro::TokenStream;
use quote::quote;
use syn::parse_macro_input;

#[derive(Debug, Default)]
pub(crate) struct ToolParser {}

impl ToolParser {
    pub fn parse(&self, attr: TokenStream, item: TokenStream) -> TokenStream {
        let tool_attrs = parse_macro_input!(attr as ToolAttributes);
        // Parse the struct, not a function
        let input_struct = parse_macro_input!(item as syn::ItemStruct);

        let struct_name = &input_struct.ident;
        let tool_name_literal = tool_attrs.name.clone();
        let tool_description = tool_attrs.description;
        let args_type = tool_attrs.input;

        let expanded = quote! {
            #input_struct

            impl ToolT for #struct_name {
                fn name(&self) -> &'static str {
                    #tool_name_literal
                }
                fn description(&self) -> &'static str {
                    #tool_description
                }
                fn run(&self, args: serde_json::Value) -> Result<serde_json::Value, ToolCallError> {
                    Ok(self.execute(args)?)
                }
                fn args_schema(&self) -> serde_json::Value {
                    // Get the JSON schema string from the input type
                    let params_str = <#args_type as ToolInputT>::io_schema();
                    serde_json::from_str(params_str)
                        .expect("Failed to parse parameters schema")
                }
            }

            impl std::fmt::Debug for #struct_name {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "{}", self.name())
                }
            }
        };

        expanded.into()
    }
}
