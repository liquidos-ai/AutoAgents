mod attr;
pub(crate) mod field;
pub(crate) mod input;
pub(crate) mod json;
use attr::ToolAttributes;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Ident, ItemFn};

#[derive(Debug, Default)]
pub(crate) struct ToolParser {}

impl ToolParser {
    pub fn parse(&self, attr: TokenStream, item: TokenStream) -> TokenStream {
        let tool_attrs = parse_macro_input!(attr as ToolAttributes);
        let input_fn = parse_macro_input!(item as ItemFn);
        let fn_name = &input_fn.sig.ident;
        let tool_name_literal = tool_attrs.name.clone();
        let tool_description = tool_attrs.description;
        let args_type = tool_attrs.input;
        let tool_struct_ident = Ident::new(&tool_attrs.name.value(), fn_name.span());

        let expanded = quote! {
            #input_fn

            pub struct #tool_struct_ident;

            impl ToolT for #tool_struct_ident {
                fn name(&self) -> &'static str {
                    #tool_name_literal
                }
                fn description(&self) -> &'static str {
                    #tool_description
                }
                fn run(&self, args: Value) -> Result<serde_json::Value, ToolCallError> {
                    let typed_args: #args_type = serde_json::from_value(args)?;
                    let result = #fn_name(typed_args)?;
                    Ok(serde_json::to_value(result)?)
                }
                fn args_schema(&self) -> Value {
                    // Retrieve the JSON schema string from the input type.
                    let params_str = <#args_type as ToolInputT>::io_schema();
                    // Parse it into a serde_json::Value.
                    let params_value: serde_json::Value = serde_json::from_str(params_str)
                        .expect("Failed to parse parameters schema");
                    params_value
                }
            }

            impl std::fmt::Debug for #tool_struct_ident {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "{}", self.name())
                }
            }
        };
        expanded.into()
    }
}
