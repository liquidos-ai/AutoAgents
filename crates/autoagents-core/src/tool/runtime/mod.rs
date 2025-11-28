use super::ToolCallError;
use async_trait::async_trait;
use std::fmt::Debug;

#[cfg(feature = "wasmtime")]
#[cfg(not(target_arch = "wasm32"))]
mod wasm;

#[cfg(feature = "wasmtime")]
#[cfg(not(target_arch = "wasm32"))]
pub use wasm::{WasmRuntime, WasmRuntimeError};

/// Runtime behavior for tools.
#[async_trait]
pub trait ToolRuntime: Send + Sync + Debug {
    /// Execute the tool with the provided JSON arguments, returning a JSON
    /// value on success or a `ToolCallError` on failure.
    async fn execute(&self, args: serde_json::Value) -> Result<serde_json::Value, ToolCallError>;
}
