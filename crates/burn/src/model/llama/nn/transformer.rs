use burn::{
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig},
    tensor::{backend::Backend, Bool, Device, Int, Tensor},
};

use super::super::{
    generation::GenerationError,
    nn::{
        attention::*,
        fftn::{FeedForward, FeedForwardConfig},
    },
};

use super::pos_encoding::PositionalEncodingState;

/// Configuration to create a Llama [decoder-only transformer](Transformer).
#[derive(Config, Debug)]
pub struct TransformerConfig {
    /// The size of the vocabulary.
    pub vocab_size: usize,
    /// The number of transformer blocks.
    pub n_layers: usize,
    /// The size of the model.
    pub d_model: usize,
    /// The size of the feed-forward hidden inner features.
    pub hidden_size: usize,
    /// The number of heads.
    pub n_heads: usize,
    /// The number of key-value heads.
    pub n_kv_heads: usize,
    /// Maximum token sequence length.
    #[config(default = "512")]
    pub max_seq_len: usize,
    /// RMSNorm epsilon.
    #[config(default = "1e-5")]
    pub norm_eps: f64,
}

impl TransformerConfig {
    /// Initialize a new [decoder-only transformer](Transformer).
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Transformer<B> {
        let tok_embeddings = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        let layers = (0..self.n_layers)
            .map(|_| {
                TransformerBlockConfig::new(
                    self.n_layers,
                    self.d_model,
                    self.hidden_size,
                    self.n_heads,
                    self.n_kv_heads,
                    self.norm_eps,
                )
                .init(device)
            })
            .collect::<Vec<_>>();
        let norm = RmsNormConfig::new(self.d_model)
            .with_epsilon(self.norm_eps)
            .init(device);
        let output = LinearConfig::new(self.d_model, self.vocab_size)
            .with_bias(false)
            .init(device);

        Transformer {
            tok_embeddings,
            layers,
            norm,
            output,
        }
    }
}

/// Llama decoder-only transformer.
#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    tok_embeddings: Embedding<B>,
    layers: Vec<TransformerBlock<B>>,
    norm: RmsNorm<B>,
    // NOTE: Starting with Llama 3.2, the weights of the output layer are tied with the embedding
    output: Linear<B>,
}

#[derive(Clone, Debug)]
pub struct TransformerCache<B: Backend> {
    layers: Vec<KeyValueCache<B>>,
    device: Device<B>,
    max_seq_len: usize,
    curr_seq_len: usize,
}

impl<B: Backend> TransformerCache<B> {
    pub fn new(config: &TransformerConfig, max_batch_size: usize, device: &Device<B>) -> Self {
        let cache = (0..config.n_layers)
            .map(|_| {
                KeyValueCache::new(
                    max_batch_size,
                    config.n_kv_heads,
                    config.max_seq_len,
                    config.d_model / config.n_heads,
                    device,
                )
            })
            .collect::<Vec<_>>();

        Self {
            layers: cache,
            device: device.clone(),
            max_seq_len: config.max_seq_len,
            curr_seq_len: 0,
        }
    }

    pub fn prepare(
        &mut self,
        seq_len: usize,
    ) -> Result<Option<Tensor<B, 4, Bool>>, GenerationError> {
        if seq_len > self.max_seq_len {
            return Err(GenerationError::MaxSequenceLengthExceeded {
                actual: seq_len,
                max: self.max_seq_len,
            });
        }

        self.curr_seq_len += seq_len;
        if self.curr_seq_len > self.max_seq_len {
            let num_removed = self.curr_seq_len - self.max_seq_len;
            self.layers
                .iter_mut()
                .for_each(|cache| cache.prepare(num_removed));
            self.curr_seq_len -= num_removed;
        }

        Ok(self.mask_attn(seq_len))
    }

    fn mask_attn(&self, seq_len: usize) -> Option<Tensor<B, 4, Bool>> {
        if seq_len <= 1 {
            return None;
        }

        let mask = Tensor::<B, 2, Bool>::tril_mask(
            [seq_len, self.curr_seq_len],
            (self.curr_seq_len - seq_len) as i64, // offset
            &self.device,
        );

        Some(mask.unsqueeze::<4>())
    }

    pub fn reset(&mut self) {
        self.curr_seq_len = 0;
        self.layers.iter_mut().for_each(|cache| cache.reset());
    }
}

impl<B: Backend> Transformer<B> {
    pub fn forward(
        &self,
        input: Tensor<B, 2, Int>,
        cache: &mut TransformerCache<B>,
        pos_encoding: &PositionalEncodingState<B>,
        mask: Option<Tensor<B, 4, Bool>>,
    ) -> Tensor<B, 3> {
        let mut h = self.tok_embeddings.forward(input);

        for (layer, c) in self.layers.iter().zip(cache.layers.iter_mut()) {
            h = layer.forward(h, c, pos_encoding, mask.clone());
        }

        let h = self.norm.forward(h);
        self.output.forward(h)
    }
}

/// Configuration to create a [decoder-only transformer block](TransformerBlock).
#[derive(Config, Debug)]
pub struct TransformerBlockConfig {
    /// The number of transformer blocks.
    pub n_layers: usize,
    /// The size of the model.
    pub d_model: usize,
    /// The size of the feed-forward hidden inner features.
    pub hidden_size: usize,
    /// The number of heads.
    pub n_heads: usize,
    /// The number of key-value heads.
    pub n_kv_heads: usize,
    /// RMSNorm epsilon.
    pub norm_eps: f64,
}

impl TransformerBlockConfig {
    /// Initialize a new [decoder-only transformer block](TransformerBlock).
    pub fn init<B: Backend>(&self, device: &Device<B>) -> TransformerBlock<B> {
        let attention =
            MultiHeadAttentionConfig::new(self.d_model, self.n_heads, self.n_kv_heads).init(device);
        let feed_forward = FeedForwardConfig::new(self.d_model, self.hidden_size).init(device);
        let attention_norm = RmsNormConfig::new(self.d_model)
            .with_epsilon(self.norm_eps)
            .init(device);
        let ffn_norm = RmsNormConfig::new(self.d_model)
            .with_epsilon(self.norm_eps)
            .init(device);

        TransformerBlock {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        }
    }
}

/// Decoder-only transformer block.
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    /// Self-attention.
    attention: MultiHeadAttention<B>,
    /// Feed-forward transformation.
    feed_forward: FeedForward<B>,
    /// Attention pre-normalization.
    attention_norm: RmsNorm<B>,
    /// Feed-forward pre-normalization.
    ffn_norm: RmsNorm<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        cache: &mut KeyValueCache<B>,
        pos_encoding: &PositionalEncodingState<B>,
        mask: Option<Tensor<B, 4, Bool>>,
    ) -> Tensor<B, 3> {
        let h = input.clone()
            + self.attention.forward_cache(
                self.attention_norm.forward(input),
                cache,
                pos_encoding,
                mask,
            );
        h.clone() + self.feed_forward.forward(self.ffn_norm.forward(h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::model::llama::tests::{TestBackend, TestTensor};
    use burn::{
        module::Reinitializer,
        nn::RotaryEncodingConfig,
        tensor::{TensorData, Tolerance},
    };

    #[test]
    fn test_rms_norm() {
        let device = Default::default();

        let rms = RmsNormConfig::new(4).with_epsilon(1e-5).init(&device);
        let input = TestTensor::<3>::from([[
            [0.0025997162, 0.0030002594, -0.006000519, 0.006000519],
            [0.0010004044, 0.00080013275, 0.0015001297, -0.01600647],
        ]]);

        let output = rms.forward(input);
        let expected = TensorData::from([[
            [0.45996094, 0.5307617, -1.0615234, 1.0615234],
            [0.11553955, 0.09240723, 0.17321777, -1.8486328],
        ]]);

        output
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::relative(0.05));
    }

    #[test]
    fn test_transformer() {
        let device: Device<TestBackend> = Default::default();
        let config = TransformerConfig::new(8, 2, 8, 16, 2, 1);
        let transformer: Transformer<TestBackend> = config.init(&device);

        let batch_size = 2;
        let seq_length = 2;

        let mut cache = TransformerCache::new(&config, batch_size, &device);

        let rope = RotaryEncodingConfig::new(seq_length * 2, config.d_model / config.n_heads)
            .init(&device);
        let rope = PositionalEncodingState::new(rope);

        let input = Tensor::arange(0..(batch_size * seq_length) as i64, &device)
            .reshape([batch_size, seq_length]);

        let transformer = Reinitializer::new()
            .range_float(0.0, 5.0)
            .apply(transformer);

        let mask = cache.prepare(seq_length).unwrap();
        let output = transformer.forward(input, &mut cache, &rope, mask);

        let expected = TensorData::from([
            [
                [
                    56.37573, 57.77283, 59.169933, 60.567043, 61.964146, 63.361248, 64.758354,
                    66.15546,
                ],
                [
                    56.374626, 57.77171, 59.168793, 60.56588, 61.962963, 63.360046, 64.75713,
                    66.15422,
                ],
            ],
            [
                [
                    56.374252, 57.771328, 59.168407, 60.565487, 61.962566, 63.359642, 64.75672,
                    66.1538,
                ],
                [
                    56.37408, 57.771156, 59.168232, 60.565304, 61.96238, 63.359455, 64.75653,
                    66.15361,
                ],
            ],
        ]);
        output
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::relative(0.001));
    }

    pub struct ForwardCacheTestCase<B: Backend> {
        cache: TransformerCache<B>,
        config: TransformerConfig,
        device: B::Device,
    }

    impl<B: Backend> ForwardCacheTestCase<B> {
        fn new(config: TransformerConfig, device: B::Device) -> Self {
            Self {
                cache: TransformerCache::new(&config, 1, &device),
                config,
                device,
            }
        }

        fn forward_seq(&mut self, seq_len: usize) {
            let x = Tensor::ones(
                [
                    1,
                    self.config.n_kv_heads,
                    seq_len,
                    self.config.d_model / self.config.n_heads,
                ],
                &self.device,
            );
            self.cache.prepare(seq_len).unwrap();
            self.forward(x);
        }

        fn forward(&mut self, x: Tensor<B, 4>) {
            for cache in self.cache.layers.iter_mut() {
                // - input:  `[batch_size, num_heads, seq_len_input, d_model]`
                // - output: `[batch_size, num_heads, seq_len_previous + seq_len_input, d_model]`
                cache.forward(x.clone(), x.clone());
            }
        }

        fn assert_eq_cache_len(&self, len: usize) {
            for cache in self.cache.layers.iter() {
                assert_eq!(cache.len(), len);
            }
        }
    }

    #[test]
    fn test_transformer_cache_should_shrink() {
        let max_seq_len = 8;
        let num_heads = 2;
        let num_kv_heads = 1;
        let d_model = 4;
        let config = TransformerConfig::new(8, 2, d_model, 4, num_heads, num_kv_heads)
            .with_max_seq_len(max_seq_len);

        let mut model = ForwardCacheTestCase::<TestBackend>::new(config, Default::default());
        assert_eq!(model.cache.max_seq_len, max_seq_len);
        assert_eq!(model.cache.curr_seq_len, 0);

        let seq_len = 4;
        model.forward_seq(seq_len);
        assert_eq!(model.cache.curr_seq_len, seq_len);
        model.assert_eq_cache_len(seq_len);

        let seq_len = 1;
        model.forward_seq(seq_len);
        assert_eq!(model.cache.curr_seq_len, 5);
        model.assert_eq_cache_len(5);

        let seq_len = 1;
        model.forward_seq(seq_len);
        assert_eq!(model.cache.curr_seq_len, 6);
        model.assert_eq_cache_len(6);

        // Shrink: any subsequent calls will shift the cache and have `max_seq_len`
        let seq_len = 6;
        model.forward_seq(seq_len);
        assert_eq!(model.cache.curr_seq_len, max_seq_len);
        model.assert_eq_cache_len(max_seq_len);

        let seq_len = 1;
        model.forward_seq(seq_len);
        assert_eq!(model.cache.curr_seq_len, max_seq_len);
        model.assert_eq_cache_len(max_seq_len);

        let seq_len = 1;
        model.forward_seq(seq_len);
        assert_eq!(model.cache.curr_seq_len, max_seq_len);
        model.assert_eq_cache_len(max_seq_len);
    }

    #[test]
    fn test_transformer_cache_exceeded_max_seq_len() {
        let max_seq_len = 8;
        let num_heads = 2;
        let num_kv_heads = 1;
        let d_model = 4;
        let config = TransformerConfig::new(8, 2, d_model, 4, num_heads, num_kv_heads)
            .with_max_seq_len(max_seq_len);
        let mut cache = TransformerCache::<TestBackend>::new(&config, 1, &Default::default());

        // When the previous inputs and generated tokens are accumulated and provided as context
        // with a new input, or the input sequence simply exceeds the max_seq_len, the cache should
        // return an error
        assert!(matches!(
            cache.prepare(16),
            Err(GenerationError::MaxSequenceLengthExceeded { actual: 16, max: 8 })
        ));
    }
}
