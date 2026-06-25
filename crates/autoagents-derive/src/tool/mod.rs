mod attr;
pub(crate) mod field;
pub(crate) mod input;
pub(crate) mod json;
use crate::resolve;
use attr::ToolAttributes;
use proc_macro::TokenStream;
use quote::quote;
use syn::parse_macro_input;

#[derive(Debug, Default)]
pub(crate) struct ToolParser {}

impl ToolParser {
    pub fn parse(&self, attr: TokenStream, item: TokenStream) -> TokenStream {
        let tool_attrs = parse_macro_input!(attr as ToolAttributes);
        let input_struct = parse_macro_input!(item as syn::ItemStruct);

        let paths = match resolve::resolve_paths() {
            Ok(paths) => paths,
            Err(err) => return err.to_compile_error().into(),
        };
        let core = &paths.core;

        let struct_name = &input_struct.ident;
        let tool_name_literal = tool_attrs.name.clone();
        let tool_description = tool_attrs.description;
        let args_type = tool_attrs.input;
        let output_schema_impl = tool_attrs.output.map(|output_type| {
            quote! {
                fn output_schema(&self) -> Option<::serde_json::Value> {
                    Some(<#output_type as #core::tool::ToolOutputT>::io_schema())
                }
            }
        });

        let expanded = quote! {
            #input_struct

            impl #core::tool::ToolT for #struct_name {
                fn name(&self) -> &str {
                    #tool_name_literal
                }
                fn description(&self) -> &str {
                    #tool_description
                }
                fn args_schema(&self) -> ::serde_json::Value {
                    <#args_type as #core::tool::ToolInputSchema>::io_schema_value()
                }
                #output_schema_impl
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
