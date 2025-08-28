use super::ToolCallError;
use std::fmt::Debug;

#[cfg(feature = "wasmtime")]
#[cfg(not(target_arch = "wasm32"))]
mod wasm;

#[cfg(feature = "wasmtime")]
#[cfg(not(target_arch = "wasm32"))]
pub use wasm::{WasmRuntime, WasmRuntimeError};

pub trait ToolRuntime: Send + Sync + Debug {
    fn execute(&self, args: serde_json::Value) -> Result<serde_json::Value, ToolCallError>;
}
