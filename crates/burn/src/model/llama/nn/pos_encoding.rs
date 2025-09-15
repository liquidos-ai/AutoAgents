use burn::prelude::Backend;
use burn::{config::Config, nn::RotaryEncoding, tensor::Tensor};

/// Tracks the state of rotary positional encodings during autoregressive inference.
///
/// Manages shifting of precomputed frequency tables when the sequence length exceeds
/// the initially allocated range. Used to avoid recomputing RoPE values on-the-fly
/// while maintaining correct positional alignment across decoding steps.
#[derive(Debug)]
pub struct PositionalEncodingState<B: Backend> {
    /// Rotary positional encoding (RoPE).
    pub rope: RotaryEncoding<B>,
    /// RoPE maximum sequence length.
    pub max_seq_len: usize,
    /// The next position.
    pub next_position: usize,
    /// The current sequence length.
    pub curr_seq_len: usize,
    /// The index start offset.
    pub start_offset: usize,
}

impl<B: Backend> PositionalEncodingState<B> {
    pub fn new(rope: RotaryEncoding<B>) -> Self {
        // Initial max position corresponds to the RoPE max seq len on initialization
        let max_seq_len = rope.freq_complex.dims()[0];
        Self {
            rope,
            max_seq_len,
            next_position: 0,
            curr_seq_len: 0,
            start_offset: 0,
        }
    }

    pub fn prepare(&mut self, seq_len: usize) {
        self.curr_seq_len = seq_len;
        self.next_position += seq_len;
        if self.next_position > self.max_seq_len + self.start_offset {
            let start = self.position();
            self.rope.shift(start);
            self.start_offset = start;
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.rope.forward(x)
    }

    pub fn apply<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.rope.apply(x, self.index())
    }

    /// Returns the absolute sequence position since the beginning,
    /// regardless of shifting.
    pub fn position(&self) -> usize {
        // The absolute sequence position should not include the current sequence
        // input so we subtract to get the current generation position.
        self.next_position - self.curr_seq_len
    }

    /// Returns the next index position for the pre-computed frequencies.
    pub fn index(&self) -> usize {
        let mut index = self.position();
        if self.start_offset > 0 {
            index -= self.start_offset
        }
        index
    }
}

/// Rotary positional encoding (RoPE)
#[derive(Config, Debug)]
pub struct RopeConfig {
    pub theta: f32,
    #[config(default = "None")]
    pub scaled: Option<RopeFrequencyScaling>,
}

/// RoPE frequency scaling.
#[derive(Config, Debug)]
pub struct RopeFrequencyScaling {
    #[config(default = "8.")]
    pub scale_factor: f32,
    #[config(default = "1.")]
    pub low_freq_factor: f32,
    #[config(default = "4.")]
    pub high_freq_factor: f32,
    #[config(default = "8192.")]
    pub old_context_len: f32,
}

impl RopeFrequencyScaling {
    /// Applies frequency scaling by parts following Llama 3.1's scheme.
    ///
    /// Adapted from: https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/model.py#L45
    pub fn freq_scaling_by_parts<B: Backend>(&self, freqs: Tensor<B, 1>) -> Tensor<B, 1> {
        let low_freq_wavelen = self.old_context_len / self.low_freq_factor;
        let high_freq_wavelen = self.old_context_len / self.high_freq_factor;

        let wavelen = freqs.clone().recip().mul_scalar(2. * core::f32::consts::PI);

        // if wavelen >= high_freq_wavelen
        let cond = wavelen.clone().greater_equal_elem(high_freq_wavelen);
        let smooth = wavelen
            .clone()
            .recip()
            .mul_scalar(self.old_context_len)
            .sub_scalar(self.low_freq_factor)
            .div_scalar(self.high_freq_factor - self.low_freq_factor);
        // (1 - smooth) * freq / scale_factor + smooth * freq
        let new_freqs = smooth
            .clone()
            .neg()
            .add_scalar(1.)
            .mul(freqs.clone().div_scalar(self.scale_factor))
            .add(smooth.clone().mul(freqs.clone()));
        let new_freqs = freqs.clone().mask_where(cond, new_freqs);

        // if wavelen > low_freq_wavelen
        let cond = wavelen.clone().greater_elem(low_freq_wavelen);
        let new_freqs = new_freqs.mask_where(cond, freqs.clone().div_scalar(self.scale_factor));

        // if wavelen < high_freq_wavelen
        let cond = wavelen.lower_elem(high_freq_wavelen);

        new_freqs.mask_where(cond, freqs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::model::llama::tests::{TestBackend, TestTensor};
    use burn::{
        nn::RotaryEncodingConfig,
        tensor::{Device, TensorData, Tolerance},
    };

    #[test]
    fn test_rope() {
        let device = Default::default();

        let max_seq_len = 16;
        let rope = RopeConfig::new(500000.0)
            .with_scaled(Some(RopeFrequencyScaling::new().with_scale_factor(32.)));
        let scaling = rope.scaled.unwrap();
        let freq_scaling_fn = move |x| scaling.freq_scaling_by_parts(x);

        let rope = RotaryEncodingConfig::new(max_seq_len * 2, 4 / 2)
            .with_theta(rope.theta)
            .init_with_frequency_scaling(freq_scaling_fn, &device);

        let input = TestTensor::<4>::from([[
            [[-0.60253906, -0.035308838], [0.41357422, 0.15100098]],
            [[-0.044677734, -0.094177246], [0.60546875, 0.2442627]],
        ]]);

        let output = rope.apply(input, 0);
        let expected = TensorData::from([[
            [[-0.60253906, -0.035308838], [0.09643555, 0.42944336]],
            [[-0.044677734, -0.094177246], [0.12194824, 0.64160156]],
        ]]);

        output
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::relative(0.05));
    }

    #[test]
    fn test_rope_shift() {
        let device: Device<TestBackend> = Default::default();

        let max_seq_len = 16;
        let rope = RopeConfig::new(500000.0)
            .with_scaled(Some(RopeFrequencyScaling::new().with_scale_factor(32.)));
        let scaling = rope.scaled.unwrap();
        let freq_scaling_fn = move |x| scaling.freq_scaling_by_parts(x);

        let rope = RotaryEncodingConfig::new(max_seq_len, 4 / 2)
            .with_theta(rope.theta)
            .init_with_frequency_scaling::<TestBackend>(freq_scaling_fn, &device);

        let mut pos_encoding = PositionalEncodingState::new(rope);
        assert_eq!(pos_encoding.max_seq_len, max_seq_len);

        // Input prompt
        pos_encoding.prepare(14);
        assert_eq!(pos_encoding.position(), 0);
        assert_eq!(pos_encoding.index(), 0);

        // Next token
        pos_encoding.prepare(1);
        assert_eq!(pos_encoding.position(), 14);
        assert_eq!(pos_encoding.index(), 14);

        // Next token
        pos_encoding.prepare(1);
        assert_eq!(pos_encoding.position(), 15);
        assert_eq!(pos_encoding.index(), 15);

        // Next prompt
        pos_encoding.prepare(8); // should apply shift
        assert_eq!(pos_encoding.position(), 16);
        assert_eq!(pos_encoding.index(), 0);

        // Next token
        pos_encoding.prepare(1);
        assert_eq!(pos_encoding.position(), 24);
        assert_eq!(pos_encoding.index(), 8);

        // Next prompt
        pos_encoding.prepare(8);
        assert_eq!(pos_encoding.position(), 25);
        assert_eq!(pos_encoding.index(), 0);

        // Next token
        pos_encoding.prepare(1);
        assert_eq!(pos_encoding.position(), 33);
        assert_eq!(pos_encoding.index(), 8);
    }
}
