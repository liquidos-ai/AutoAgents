use super::ToolCallError;
use async_trait::async_trait;
use std::fmt::Debug;

#[cfg(feature = "wasmtime")]
#[cfg(not(target_arch = "wasm32"))]
mod wasm;

#[cfg(feature = "wasmtime")]
#[cfg(not(target_arch = "wasm32"))]
pub use wasm::{WasmRuntime, WasmRuntimeError};

#[async_trait]
pub trait ToolRuntime: Send + Sync + Debug {
    async fn execute(&self, args: serde_json::Value) -> Result<serde_json::Value, ToolCallError>;
}
