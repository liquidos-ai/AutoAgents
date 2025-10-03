use burn::tensor::{backend::Backend, Int, Tensor};
use burn::tensor::{DType, TensorData};
use rand::{
    distr::{weighted::WeightedIndex, Distribution},
    rngs::StdRng,
    SeedableRng,
};

pub async fn manual_argmax<B: Backend>(logits: Tensor<B, 2>) -> Tensor<B, 2, Int> {
    // Pull logits data to CPU
    let data = logits.clone().into_data_async().await;

    // decode the bytes into f32 values
    let values: Vec<f32> = match data.dtype {
        DType::F32 => bytemuck::cast_slice::<u8, f32>(&data.bytes).to_vec(),
        DType::F16 => {
            let halves: &[half::f16] = bytemuck::cast_slice(&data.bytes);
            halves.iter().map(|h| f32::from(*h)).collect()
        }
        _ => panic!("Unexpected dtype {:?}", data.dtype),
    };

    let shape = data.shape.clone(); // [batch, vocab]
    let batch = shape[0];
    let vocab = shape[1];

    // compute argmax per row
    let mut indices: Vec<i64> = Vec::with_capacity(batch);
    for b in 0..batch {
        let start = b * vocab;
        let end = start + vocab;
        let row = &values[start..end];
        let mut max_idx = 0usize;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        indices.push(max_idx as i64); // Int expects i64
    }

    // // wrap into a TensorData with shape [batch, 1] for Tensor<B, 2, Int>
    // let indices_bytes: Vec<u8> = bytemuck::cast_slice(&indices).to_vec();
    let indices_data = TensorData::new(indices, vec![batch, 1]);

    // build Tensor<B, 2, Int>
    Tensor::<B, 2, Int>::from_data(indices_data, &logits.device())
}

#[allow(clippy::large_enum_variant)]
pub enum Sampler {
    TopP(TopP),
    Argmax,
}

impl Sampler {
    pub async fn sample<B: Backend>(&mut self, logits: Tensor<B, 2>) -> Tensor<B, 2, Int> {
        match self {
            Self::TopP(s) => s.sample(logits).await,
            Self::Argmax => logits.argmax(1),
        }
    }
}

#[async_trait::async_trait]
pub trait Sampling {
    async fn sample<B: Backend>(&mut self, logits: Tensor<B, 2>) -> Tensor<B, 2, Int>;
}

/// Top-p sampling (nucleus sampling) selects the smallest set of tokens whose cumulative
/// probability mass exceed the threshold p.
pub struct TopP {
    /// Probability threshold for sampling.
    p: f64,
    /// RNG.
    rng: StdRng,
}

impl TopP {
    pub fn new(p: f64, seed: u64) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        Self { p, rng }
    }
}

#[async_trait::async_trait]
impl Sampling for TopP {
    async fn sample<B: Backend>(&mut self, probs: Tensor<B, 2>) -> Tensor<B, 2, Int> {
        assert_eq!(
            probs.dims()[0],
            1,
            "Naive top-p sampling only supports single-batch tensors"
        );
        let (probs_sort, probs_idx) = probs.sort_descending_with_indices(1);

        // TODO: cumsum + Distribution::Multinomial support

        #[cfg(target_arch = "wasm32")]
        let mut probs_sort = {
            // asynchronously pull data from the tensor
            let data = probs_sort.to_data_async().await;
            // then iterate it
            data.iter::<f64>().collect::<Vec<_>>()
        };

        #[cfg(not(target_arch = "wasm32"))]
        let mut probs_sort = {
            let data = probs_sort.to_data();
            data.iter::<f64>().collect::<Vec<_>>()
        };

        let mut cumsum = 0.;
        probs_sort.iter_mut().for_each(|x| {
            if cumsum >= self.p {
                *x = 0.0;
            } else {
                cumsum += *x;
            }
        });

        let next_token_idx = WeightedIndex::new(probs_sort)
            .unwrap()
            .sample(&mut self.rng);

        probs_idx.slice([0..1, next_token_idx..next_token_idx + 1])
    }
}
