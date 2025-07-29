use super::ToolCallError;
use std::fmt::Debug;

#[cfg(feature = "wasm")]
mod wasm;

#[cfg(feature = "wasm")]
pub use wasm::{WasmRuntime, WasmRuntimeError};

pub trait ToolRuntime: Send + Sync + Debug {
    fn execute(&self, args: serde_json::Value) -> Result<serde_json::Value, ToolCallError>;
}
