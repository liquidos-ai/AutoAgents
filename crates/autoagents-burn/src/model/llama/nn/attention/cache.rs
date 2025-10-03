use std::ops::Range;

use burn::tensor::{backend::Backend, Device, Tensor};

/// Strategy for managing the autoregressive cache when its capacity is exceeded.
#[derive(Debug, Clone, Default)]
pub(crate) enum CacheStrategy {
    /// Shrinks the cache by copying the remaining tokens into a new buffer,
    /// removing the oldest tokens beyond the context limit.
    #[allow(dead_code)]
    Shrink,

    /// Shifts the remaining tokens to the start of the existing buffer in-place,
    /// overwriting the oldest tokens.
    #[default]
    Shift,
}

#[derive(Debug, Clone)]
/// Cache that keeps track of a tensor state in an autoregressive decoding process.
pub(crate) struct AutoregressiveCache<B: Backend, const D: usize> {
    cache: Tensor<B, D>,
    seq_dim: usize,
    cur_seq_len: usize,
    strategy: CacheStrategy,
}

impl<const D: usize, B: Backend> AutoregressiveCache<B, D> {
    /// Creates a new empty cache.
    pub fn new(shape: [usize; D], seq_dim: usize, device: &Device<B>) -> Self {
        Self {
            cache: Tensor::empty(shape, device),
            seq_dim,
            cur_seq_len: 0,
            strategy: CacheStrategy::default(),
        }
    }

    #[allow(dead_code)]
    /// Sets the cache management strategy.
    pub fn with_strategy(mut self, strategy: CacheStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Reset the cache state.
    pub fn reset(&mut self) {
        // Note: we don't need to clear the tensor since we track the current seq length
        self.cur_seq_len = 0;
    }

    /// Add the new tokens to the current cache and returns all tokens decoded since the beginning.
    ///
    /// # Shapes
    ///
    /// - input:  `[batch_size, num_heads, seq_len_input, d_model]`
    /// - output: `[batch_size, num_heads, seq_len_previous + seq_len_input, d_model]`
    pub fn append(&mut self, tokens: Tensor<B, D>) -> Tensor<B, D> {
        let shape = tokens.shape();
        let seq_len_input = shape.dims[self.seq_dim];

        let new_seq_len = self.cur_seq_len + seq_len_input;

        let mut indices_added_tokens = Vec::with_capacity(shape.dims.len());
        let mut indices_output = Vec::with_capacity(shape.dims.len());

        for (i, shape) in shape.dims.iter().enumerate() {
            if i == self.seq_dim {
                indices_added_tokens.push(self.cur_seq_len..new_seq_len);
                indices_output.push(0..new_seq_len);
            } else {
                indices_added_tokens.push(0..*shape);
                indices_output.push(0..*shape);
            }
        }
        self.cache.inplace(|cache| {
            cache.slice_assign::<D, [Range<usize>; D]>(
                indices_added_tokens.try_into().unwrap(),
                tokens,
            )
        });

        self.cur_seq_len = new_seq_len;

        self.cache
            .clone()
            .slice::<D, [Range<usize>; D]>(indices_output.try_into().unwrap())
    }

    /// Prepare the cache by applying the configured strategy to make room for new tokens.
    ///
    /// `num_tokens` is the number of past tokens to discard or shift, depending on the strategy.
    pub fn prepare(&mut self, num_tokens: usize) {
        match self.strategy {
            CacheStrategy::Shrink => self.shrink(num_tokens),
            CacheStrategy::Shift => self.shift(num_tokens),
        }
    }

    /// Shrink the cache to fit in `max_seq_len` while making place for the new tokens being
    /// decoded.
    fn shrink(&mut self, num_removed: usize) {
        let old_cur_seq_len = self.cur_seq_len;
        self.cur_seq_len -= num_removed;

        let shape = self.cache.shape();
        let device = self.cache.device();

        let mut slices_prev = Vec::with_capacity(shape.dims.len());
        let mut slices_curr = Vec::with_capacity(shape.dims.len());

        for (i, shape) in shape.dims.iter().enumerate() {
            if i == self.seq_dim {
                slices_prev.push(num_removed..old_cur_seq_len);
                slices_curr.push(0..self.cur_seq_len);
            } else {
                slices_prev.push(0..*shape);
                slices_curr.push(0..*shape);
            }
        }

        self.cache.inplace(|cache| {
            let prev_slice = cache.slice::<D, [Range<usize>; D]>(slices_prev.try_into().unwrap());
            let new_cache = Tensor::empty(shape, &device);

            new_cache
                .slice_assign::<D, [Range<usize>; D]>(slices_curr.try_into().unwrap(), prev_slice)
        });
    }

    /// Shift the cache to fit in `max_seq_len` while making place for the new tokens being
    /// decoded.
    fn shift(&mut self, num_shifted: usize) {
        let old_cur_seq_len = self.cur_seq_len;
        self.cur_seq_len -= num_shifted;

        let shape = self.cache.shape();

        let mut slices_prev = Vec::with_capacity(shape.dims.len());
        let mut slices_curr = Vec::with_capacity(shape.dims.len());

        for (i, shape) in shape.dims.iter().enumerate() {
            if i == self.seq_dim {
                slices_prev.push(num_shifted..old_cur_seq_len);
                slices_curr.push(0..self.cur_seq_len);
            } else {
                slices_prev.push(0..*shape);
                slices_curr.push(0..*shape);
            }
        }

        // Shift tail -> head
        self.cache.inplace(|cache| {
            let prev_slice = cache
                .clone()
                .slice::<D, [Range<usize>; D]>(slices_prev.try_into().unwrap());

            cache.slice_assign::<D, [Range<usize>; D]>(slices_curr.try_into().unwrap(), prev_slice)
        });
    }

    /// Returns the cached sequence length.
    pub fn len(&self) -> usize {
        self.cur_seq_len
    }

    #[allow(dead_code)]
    pub fn device(&self) -> Device<B> {
        self.cache.device()
    }
}
