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
pub mod error;
pub mod protocol;
pub mod tool;

#[cfg(test)]
mod tests;

// Re-export ractor only for non-WASM targets
#[cfg(not(target_arch = "wasm32"))]
pub mod ractor {
    pub use ractor::*;
    #[cfg(feature = "cluster")]
    pub mod cluster {
        pub use ractor_cluster::*;
    }
}
