use super::kv_cache::KeyValueCache;
use crate::model::llama::nn::pos_encoding::PositionalEncodingState;
use burn::{
    nn::{Linear, LinearConfig, RotaryEncoding},
    prelude::*,
    tensor::activation::softmax,
};

/// Configuration to create a [multi-head attention](MultiHeadAttention) module.
#[derive(Config, Debug)]
pub struct MultiHeadAttentionConfig {
    /// The size of the model.
    pub d_model: usize,
    /// The number of heads.
    pub n_heads: usize,
    /// The number of key-value heads.
    pub n_kv_heads: usize,
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    /// Query projection.
    wq: Linear<B>,
    /// Key projection.
    wk: Linear<B>,
    /// Value projection.
    wv: Linear<B>,
    /// Output projection.
    wo: Linear<B>,

    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl<B: Backend> MultiHeadAttention<B> {
    /// Applies masked self-attention in a non-cached (non-incremental) setting.
    ///
    /// This function is intended for scenarios where the entire input sequence
    /// is available.
    ///
    /// # Shapes
    ///
    /// - query: `[batch_size, seq_length_1, d_model]`
    /// - key: `[batch_size, seq_length_2, d_model]`
    /// - value: `[batch_size, seq_length_2, d_model]`
    /// - output: `[batch_size, seq_length_1, d_model]`
    pub fn forward_masked(&self, input: Tensor<B, 3>, rope: &RotaryEncoding<B>) -> Tensor<B, 3> {
        let device = input.device();
        let [batch_size, seq_len, hidden_size] = input.dims();

        let (q, k, v) = self.forward_projection(input);

        // Start position is 0
        let q = rope.forward(q);
        let k = rope.forward(k);

        let mask = if seq_len > 1 {
            let mask = Tensor::<B, 2, Bool>::tril_mask([seq_len, seq_len], 0, &device);
            Some(mask.unsqueeze::<4>())
        } else {
            None
        };

        let output = self.forward_attention(q, k, v, mask, batch_size, seq_len, hidden_size);
        self.wo.forward(output)
    }

    /// Applies the forward pass on the input tensors.
    ///
    /// # Shapes
    ///
    /// - query: `[batch_size, seq_length_1, d_model]`
    /// - key: `[batch_size, seq_length_2, d_model]`
    /// - value: `[batch_size, seq_length_2, d_model]`
    /// - output: `[batch_size, seq_length_1, d_model]`
    pub fn forward_cache(
        &self,
        input: Tensor<B, 3>,
        cache: &mut KeyValueCache<B>,
        pos_encoding: &PositionalEncodingState<B>,
        mask: Option<Tensor<B, 4, Bool>>,
    ) -> Tensor<B, 3> {
        let device = input.device();
        let [batch_size, seq_len, hidden_size] = input.dims();

        let (q, k, v) = self.forward_projection(input);

        let q = pos_encoding.apply(q);
        let k = pos_encoding.apply(k);

        // Key-value caching
        let (k, v) = cache.forward(k, v);

        let mask = if seq_len > 1 {
            match mask {
                Some(mask) => Some(mask),
                None => {
                    // We ensure that the correct mask is applied
                    let cache_seq_len = cache.len();
                    let mask = Tensor::<B, 2, Bool>::tril_mask(
                        [seq_len, cache_seq_len],
                        (cache_seq_len - seq_len) as i64, // offset
                        &device,
                    );

                    Some(mask.unsqueeze::<4>())
                }
            }
        } else {
            None
        };

        let output = self.forward_attention(q, k, v, mask, batch_size, seq_len, hidden_size);

        self.wo.forward(output)
    }

    fn forward_projection(
        &self,
        input: Tensor<B, 3>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [batch_size, seq_len, _hidden_size] = input.dims();

        let q = self.wq.forward(input.clone());
        let k = self.wk.forward(input.clone());
        let v = self.wv.forward(input);

        // [batch_size, num_heads, seq_len, head_dim]
        let q = q
            .reshape([batch_size, seq_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch_size, seq_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch_size, seq_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        (q, k, v)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_attention(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        mask: Option<Tensor<B, 4, Bool>>,
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
    ) -> Tensor<B, 3> {
        let k = self.repeat_kv(k);
        let v = self.repeat_kv(v);

        // Attention scores
        let mut scores = q
            .matmul(k.swap_dims(2, 3))
            .div_scalar((self.head_dim as f32).sqrt());

        if let Some(mask) = mask {
            scores = scores.mask_fill(mask, f32::NEG_INFINITY);
        }

        let scores = softmax(scores, 3);
        let output = scores.matmul(v);

        output
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, hidden_size])
    }

    /// Repeats a key or value tensor for grouped query attention.
    fn repeat_kv(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let n_rep = self.n_heads / self.n_kv_heads;
        if n_rep == 1 {
            x
        } else {
            let [batch_size, num_kv_heads, seq_len, head_dim] = x.dims();

            x.unsqueeze_dim::<5>(2)
                .expand([batch_size, num_kv_heads, n_rep, seq_len, head_dim])
                .reshape([batch_size, num_kv_heads * n_rep, seq_len, head_dim])
        }
    }
}

impl MultiHeadAttentionConfig {
    /// Initialize a new [multi-head attention](MultiHeadAttention) module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> MultiHeadAttention<B> {
        let head_dim = self.d_model / self.n_heads;

        let wq = LinearConfig::new(self.d_model, self.n_heads * head_dim)
            .with_bias(false)
            .init(device);
        let wk = LinearConfig::new(self.d_model, self.n_kv_heads * head_dim)
            .with_bias(false)
            .init(device);
        let wv = LinearConfig::new(self.d_model, self.n_kv_heads * head_dim)
            .with_bias(false)
            .init(device);
        let wo = LinearConfig::new(self.n_heads * head_dim, self.d_model)
            .with_bias(false)
            .init(device);

        MultiHeadAttention {
            wq,
            wk,
            wv,
            wo,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim,
        }
    }
}
