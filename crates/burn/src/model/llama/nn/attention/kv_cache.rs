use burn::prelude::Backend;
use burn::tensor::{Device, Tensor};

use super::cache::AutoregressiveCache;

/// Key-value cache for autoregressive models.
#[derive(Debug, Clone)]
pub struct KeyValueCache<B: Backend> {
    key: AutoregressiveCache<B, 4>,
    value: AutoregressiveCache<B, 4>,
}

impl<B: Backend> KeyValueCache<B> {
    /// Create a new [key-value cache](KeyValueCache).
    pub fn new(
        max_batch_size: usize,
        num_heads: usize,
        max_seq_len: usize,
        d_model: usize,
        device: &Device<B>,
    ) -> Self {
        Self {
            key: AutoregressiveCache::new(
                [max_batch_size, num_heads, max_seq_len, d_model],
                2,
                device,
            ),
            value: AutoregressiveCache::new(
                [max_batch_size, num_heads, max_seq_len, d_model],
                2,
                device,
            ),
        }
    }

    /// Computes the complete keys and values.
    pub fn forward(
        &mut self,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let k = self.key.append(key);
        let v = self.value.append(value);
        (k, v)
    }

    /// Returns the cached sequence length.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        // We can assume key and value have the same length
        self.key.len()
    }

    pub fn prepare(&mut self, num_tokens: usize) {
        self.key.prepare(num_tokens);
        self.value.prepare(num_tokens);
    }

    /// Reset key-value cache.
    /// Use between different contexts (i.e., for each new prompt).
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.key.reset();
        self.value.reset();
    }
}
