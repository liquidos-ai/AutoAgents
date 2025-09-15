pub mod tokenizer;

/// Neural network components.
pub mod nn;

/// Text generation components.
pub mod generation;

pub mod chat;
#[cfg(feature = "tiny")]
mod tiny;

#[cfg(feature = "tiny")]
pub use tiny::TinyLlamaBuilder;

#[cfg(feature = "llama3")]
mod llama3;

#[cfg(feature = "llama3")]
pub use llama3::{LLama3ModelConfig, Llama3Builder, Llama3Model};

pub use nn::llama::*;

#[cfg(test)]
mod tests {
    #[cfg(any(not(feature = "test-non-default"), feature = "test-ndarray"))]
    pub type TestBackend = burn::backend::NdArray<f32, i32>;

    pub type TestTensor<const D: usize> = burn::tensor::Tensor<TestBackend, D>;
}
