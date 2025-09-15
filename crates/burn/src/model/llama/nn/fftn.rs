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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::llama::tests::TestBackend;
    use burn::{
        module::Reinitializer,
        tensor::{TensorData, Tolerance},
    };

    #[test]
    fn test_fftn() {
        let device: Device<TestBackend> = Default::default();
        let batch_size = 2;
        let seq_length = 2;
        let d_model = 4;
        let hidden_size = 8;

        let config = FeedForwardConfig::new(d_model, hidden_size);
        let transformer: FeedForward<TestBackend> = config.init(&device);

        let input = Tensor::arange(0..(batch_size * seq_length * d_model) as i64, &device)
            .reshape([batch_size, seq_length, d_model])
            .float();

        let nn = Reinitializer::new()
            .range_float(0.0, 5.0)
            .apply(transformer);

        let output = nn.forward(input);

        let expected = TensorData::from([
            [
                [8661.133, 9206.727, 9752.319, 10297.912],
                [73848.63, 78356.14, 82863.65, 87371.16],
            ],
            [
                [202911.13, 215216.48, 227521.86, 239827.2],
                [395848.63, 419787.78, 443726.94, 467666.06],
            ],
        ]);

        output
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::balanced());
    }
}
