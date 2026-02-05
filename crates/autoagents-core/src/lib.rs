// Runtime-dependent modules (not available in WASM)
#[cfg(not(target_arch = "wasm32"))]
pub mod actor;
#[cfg(not(target_arch = "wasm32"))]
pub mod environment;
#[cfg(not(target_arch = "wasm32"))]
pub mod runtime;

// Agent module with conditional compilation
pub mod agent;

// Common modules available on all platforms
mod channel;
pub mod document;
pub mod embeddings;
pub mod error;
#[cfg(not(target_arch = "wasm32"))]
mod event_fanout;
pub mod one_or_many;
pub mod protocol;
pub mod readers;
pub mod tool;
pub mod utils;
pub mod vector_store;

#[cfg(test)]
mod tests;

// Re-export ractor only for non-WASM targets
#[cfg(not(target_arch = "wasm32"))]
pub mod ractor {
    pub use ractor::*;
}
