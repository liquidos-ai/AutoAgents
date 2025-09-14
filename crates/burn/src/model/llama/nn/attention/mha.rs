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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::llama::tests::TestBackend;
    use burn::{module::Reinitializer, nn::RotaryEncodingConfig, tensor::Tolerance};

    #[test]
    pub fn test_attention_empty_cache() {
        let seq_length = 3;
        let batch_size = 2;
        let config = MultiHeadAttentionConfig::new(32, 2, 2);
        let device: Device<TestBackend> = Default::default();
        let mha = config.init::<TestBackend>(&device);

        let mha = Reinitializer::new().random_float(0, -2.0, 2.0).apply(mha);

        let shape = Shape::from([batch_size, seq_length, config.d_model]);
        let input = Tensor::arange(0..shape.num_elements() as i64, &device)
            .reshape(shape)
            .float();

        let mut cache = KeyValueCache::new(
            batch_size,
            config.n_heads,
            seq_length,
            config.d_model,
            &device,
        );

        let rope = RotaryEncodingConfig::new(seq_length * 2, config.d_model / config.n_heads)
            .init(&device);
        let rope = PositionalEncodingState::new(rope);

        let output = mha.forward_cache(input, &mut cache, &rope, None);
        let expected = arange_mha_expected_value();

        output
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::balanced());
    }

    #[test]
    pub fn test_attention_masked() {
        let seq_length = 3;
        let batch_size = 2;
        let config = MultiHeadAttentionConfig::new(32, 2, 2);
        let device: Device<TestBackend> = Default::default();
        let mha = config.init::<TestBackend>(&device);

        let mha = Reinitializer::new().random_float(0, -2.0, 2.0).apply(mha);

        let shape = Shape::from([batch_size, seq_length, config.d_model]);
        let input = Tensor::arange(0..shape.num_elements() as i64, &device)
            .reshape(shape)
            .float();

        let rope = RotaryEncodingConfig::new(seq_length * 2, config.d_model / config.n_heads)
            .init(&device);

        let output = mha.forward_masked(input, &rope);
        let expected = arange_mha_expected_value();

        output
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::balanced());
    }

    #[test]
    pub fn test_attention_decoding() {
        let seq_length = 3;
        let batch_size = 2;
        let config = MultiHeadAttentionConfig::new(32, 2, 2);
        let device: Device<TestBackend> = Default::default();
        let mha = config.init::<TestBackend>(&device);

        let mha = Reinitializer::new().random_float(0, -2.0, 2.0).apply(mha);

        let shape = Shape::from([batch_size, seq_length, config.d_model]);
        let input = Tensor::arange(0..shape.num_elements() as i64, &device)
            .reshape(shape)
            .float();

        let rope = RotaryEncodingConfig::new(seq_length * 2, config.d_model / config.n_heads)
            .init(&device);
        let rope = PositionalEncodingState::new(rope);

        let mut cache = KeyValueCache::new(
            batch_size,
            config.n_heads,
            seq_length,
            config.d_model,
            &device,
        );

        let out_1 = mha.forward_cache(
            input
                .clone()
                .slice([0..batch_size, 0..1, 0..config.d_model]),
            &mut cache,
            &rope,
            None,
        );
        let out_2 = mha.forward_cache(
            input
                .clone()
                .slice([0..batch_size, 1..2, 0..config.d_model]),
            &mut cache,
            &rope,
            None,
        );
        let out_3 = mha.forward_cache(
            input
                .clone()
                .slice([0..batch_size, 2..3, 0..config.d_model]),
            &mut cache,
            &rope,
            None,
        );

        let output = Tensor::cat(vec![out_1, out_2, out_3], 1);

        let expected = arange_mha_expected_value();

        output
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::balanced());
    }

    fn arange_mha_expected_value() -> TensorData {
        TensorData::from([
            [
                [
                    1370.383, 307.15445, -1173.6564, -384.0224, 710.8423, 120.73452, 1430.4589,
                    -873.6865, -1580.7152, -72.31566, -601.4169, -764.4472, -564.9576, -1620.3185,
                    -1091.127, 815.09235, 99.62501, -239.23195, -736.08026, 851.33997, -1124.7622,
                    893.9406, -333.98218, 901.82874, -239.55196, 1240.419, -758.5158, 71.56546,
                    -821.08093, 1148.4462, 207.87701, -2532.043,
                ],
                [
                    3307.7605, 811.50775, -2769.4685, -720.05865, 1910.8195, -36.026237, 3551.262,
                    -2659.366, -3429.0422, -253.68767, -1836.4116, -2041.1503, -1913.1508,
                    -3154.4177, -2310.1633, 2156.062, 409.66623, -1703.5918, -2312.558, 1067.0983,
                    -3079.6296, 1771.3408, -889.1188, 2533.0588, -1266.8694, 4139.4883, -1989.368,
                    -68.18975, -1964.7156, 2947.4482, 1607.4972, -5652.1143,
                ],
                [
                    5245.1396, 1315.8612, -4365.281, -1056.0946, 3110.7983, -192.7872, 5672.0645,
                    -4445.0454, -5277.3696, -435.05988, -3071.407, -3317.853, -3261.3428,
                    -4688.516, -3529.1992, 3497.0308, 719.70715, -3167.9507, -3889.0356, 1282.8573,
                    -5034.4966, 2648.7417, -1444.2544, 4164.2886, -2294.186, 7038.5557, -3220.2207,
                    -207.94545, -3108.3496, 4746.45, 3007.1165, -8772.186,
                ],
            ],
            [
                [
                    7182.5176, 1820.2142, -5961.094, -1392.1312, 4310.775, -349.54813, 7792.8696,
                    -6230.7256, -7125.696, -616.432, -4306.401, -4594.5566, -4609.5366, -6222.6167,
                    -4748.2363, 4838.0015, 1029.7491, -4632.3105, -5465.5127, 1498.6154,
                    -6989.3643, 3526.1414, -1999.3912, 5795.5186, -3321.5046, 9937.625, -4451.0723,
                    -347.70007, -4251.9844, 6545.4526, 4406.7363, -11892.258,
                ],
                [
                    9119.896, 2324.568, -7556.9063, -1728.1674, 5510.7515, -506.31003, 9913.672,
                    -8016.405, -8974.022, -797.8044, -5541.3975, -5871.258, -5957.729, -7756.715,
                    -5967.272, 6178.9707, 1339.7911, -6096.67, -7041.9907, 1714.3739, -8944.231,
                    4403.541, -2554.5278, 7426.749, -4348.822, 12836.694, -5681.925, -487.45474,
                    -5395.618, 8344.454, 5806.3564, -15012.33,
                ],
                [
                    11057.273, 2828.9211, -9152.718, -2064.2043, 6710.732, -663.0684, 12034.475,
                    -9802.082, -10822.352, -979.175, -6776.3916, -7147.9614, -7305.922, -9290.814,
                    -7186.308, 7519.9404, 1649.8324, -7561.0293, -8618.468, 1930.1324, -10899.1,
                    5280.942, -3109.664, 9057.98, -5376.138, 15735.763, -6912.776, -627.2097,
                    -6539.253, 10143.456, 7205.976, -18132.396,
                ],
            ],
        ])
    }
}
