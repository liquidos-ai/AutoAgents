pub mod tokenizer;

/// Neural network components.
pub mod nn;

/// Text generation components.
pub mod generation;

#[cfg(feature = "tiny")]
mod tiny;

#[cfg(feature = "tiny")]
pub use tiny::{TinyLlama, TinyLlamaBuilder};

pub use nn::llama::*;

#[cfg(test)]
mod tests {
    #[cfg(any(not(feature = "test-non-default"), feature = "test-ndarray"))]
    pub type TestBackend = burn::backend::NdArray<f32, i32>;

    pub type TestTensor<const D: usize> = burn::tensor::Tensor<TestBackend, D>;
}
