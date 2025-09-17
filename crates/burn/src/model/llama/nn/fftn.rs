use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, SwiGlu, SwiGluConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Tensor};

#[derive(Config, Debug)]
/// Configuration to create a [feed-forward transformation network](FeedForward).
pub struct FeedForwardConfig {
    /// The size of the model.
    pub d_model: usize,
    /// The size of the hidden inner features.
    pub hidden_size: usize,
}

/// Feed-forward transformation network.
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    // Swish gated linear unit with trainable parameters.
    swiglu: SwiGlu<B>,
    /// Outer linear.
    w2: Linear<B>,
}

impl FeedForwardConfig {
    /// Initialize a new [feed-forward transformation network](FeedForward).
    pub fn init<B: Backend>(&self, device: &Device<B>) -> FeedForward<B> {
        let swiglu = SwiGluConfig::new(self.d_model, self.hidden_size)
            .with_bias(false)
            .init(device);
        let w2 = LinearConfig::new(self.hidden_size, self.d_model)
            .with_bias(false)
            .init(device);

        FeedForward { swiglu, w2 }
    }
}
impl<B: Backend> FeedForward<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, seq_length, d_model]`
    /// - output: `[batch_size, seq_length, d_model]`
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.w2.forward(self.swiglu.forward(input))
    }
}
