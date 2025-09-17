use std::time::Instant;

use super::super::{tokenizer::Tokenizer, Llama};
use super::{GenerationContext, Sampler};
use crate::model::llama::generation::stream_sender::StreamSender;
use autoagents_llm::error::LLMError;
use burn::{prelude::*, tensor::activation::softmax};

pub(crate) fn temperature_scaled_softmax<B: Backend>(
    logits: Tensor<B, 2>,
    temperature: f64,
) -> Tensor<B, 2> {
    softmax(logits / temperature, 1)
}

/// Generated text sample output.
pub struct GenerationOutput {
    /// The number of generated tokens.
    pub tokens: usize,
    /// The time it took to produce the output tokens (generation + decoding).
    // pub time: std::time::Duration,
    pub result: String,
}

#[derive(Debug)]
pub enum GenerationError {
    MaxSequenceLengthExceeded { actual: usize, max: usize },
}

impl<B: Backend, T: Tokenizer + 'static> Llama<B, T> {
    /// Generate text sample based on the provided prompt.
    ///
    /// # Arguments
    /// - `prompt`: The prompt string to use for generating the samples.
    /// - `sample_len`: The number of new tokens to generate (i.e., the number of generation steps to take).
    /// - `temperature`: Temperature value for controlling randomness in sampling (scales logits by `1 / temperature`).
    ///   High values result in more random sampling.
    /// - `sampler`: The sampling strategy to use when selecting the next token based on the predicted probabilities.
    ///
    /// # Returns
    /// The generated text along with some other metadata (see [GenerationOutput]).
    pub async fn generate(
        &mut self,
        prompt: &str,
        sample_len: usize,
        temperature: f64,
        sampler: &mut Sampler,
        emitter: Option<StreamSender>,
    ) -> Result<GenerationOutput, GenerationError> {
        let input_tokens = self.tokenize(prompt);
        let prompt_len = input_tokens.dims()[0];

        let mut state = GenerationContext::new(
            prompt_len + sample_len,
            self.tokenizer.clone(),
            &self.device,
            emitter,
        )
        .await;
        state.append(input_tokens);

        let mut input_pos = Tensor::<B, 1, Int>::arange(0..prompt_len as i64, &self.device);
        // let now = Instant::now();

        for _ in 0..sample_len {
            if state.should_stop() {
                break;
            }

            let x = state
                .tokens
                .clone()
                .select(0, input_pos.clone())
                .reshape([1, -1]);

            let [_, seq_len] = x.dims();

            // Prepare cache and RoPE for current sequence length and position
            let mask = self.cache.prepare(seq_len)?;
            self.pos_encoding.prepare(seq_len);

            let logits = self
                .model
                .forward(x, &mut self.cache, &self.pos_encoding, mask);

            let [batch_size, seq_len, _vocab_size] = logits.dims();
            let mut next_token_logits = logits
                .slice([0..batch_size, seq_len - 1..seq_len])
                .squeeze(1); // [batch_size=1, vocab_size]

            if temperature > 0.0 {
                next_token_logits = temperature_scaled_softmax(next_token_logits, temperature);
            };

            let next_token = sampler.sample(next_token_logits).await.squeeze(0);
            // Update with the new generated token
            state.update(next_token).await;

            // Advance
            let t = input_pos.dims()[0];
            input_pos = input_pos.slice(t - 1..t) + 1;
        }

        let num_tokens = state.num_tokens_generated();

        // Decode the generated tokens to text
        let generated_text = state.get_generated_text().await;

        Ok(GenerationOutput {
            tokens: num_tokens,
            // time: now.elapsed(),
            result: generated_text,
        })
    }
}
