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
    pub(crate) tok_embeddings: Embedding<B>,
    pub(crate) layers: Vec<TransformerBlock<B>>,
    pub(crate) norm: RmsNorm<B>,
    // NOTE: Starting with Llama 3.2, the weights of the output layer are tied with the embedding
    pub(crate) output: Linear<B>,
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
