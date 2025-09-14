use super::super::tokenizer::Tokenizer;
use crate::model::llama::generation::streaming::StreamingDecoder;
use burn::prelude::Backend;
use burn::tensor::{Int, Tensor};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc::{Receiver, Sender},
    Arc, Mutex,
};

#[derive(Clone)]
/// The text generation context, used to check when a stop token has been reached.
pub struct GenerationContext<B: Backend> {
    pub tokens: Tensor<B, 1, Int>,
    num_tokens: usize,
    stop: Arc<AtomicBool>,
    num_generated: Arc<AtomicUsize>,
    sender: Sender<Tensor<B, 1, Int>>,
    generated_text: Arc<Mutex<String>>,
}

impl<B: Backend> GenerationContext<B> {
    /// Create a new generation context.
    pub fn new<T: Tokenizer + 'static>(
        max_sample_len: usize,
        tokenizer: T,
        device: &B::Device,
    ) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel::<Tensor<B, 1, Int>>();
        let stop = Arc::new(AtomicBool::new(false));
        let num_generated = Arc::new(AtomicUsize::new(0));
        let generated_text = Arc::new(Mutex::new(String::new()));

        let mut generation = TokenGeneration::new(
            tokenizer,
            stop.clone(),
            num_generated.clone(),
            generated_text.clone(),
        );

        std::thread::spawn(move || {
            for tokens in receiver.iter() {
                let tokens = tokens
                    .into_data()
                    .convert::<u32>()
                    .into_vec::<u32>()
                    .unwrap();

                generation.process(tokens);
            }
        });

        Self {
            tokens: Tensor::empty([max_sample_len], device),
            num_tokens: 0,
            stop,
            num_generated,
            sender,
            generated_text,
        }
    }

    /// Add generated tokens to the state (without checking for stop condition).
    pub fn append(&mut self, tokens: Tensor<B, 1, Int>) {
        let num_tokens_prev = self.num_tokens;
        self.num_tokens += tokens.shape().num_elements();
        self.tokens
            .inplace(|toks| toks.slice_assign(num_tokens_prev..self.num_tokens, tokens));
    }

    /// Update the state with newly generated tokens.
    pub fn update(&mut self, tokens: Tensor<B, 1, Int>) {
        self.append(tokens.clone());

        if !self.should_stop() {
            let _ = self.sender.send(tokens);
        }
    }

    /// True if the state previously detected a stop token.
    pub fn should_stop(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Returns the number of tokens generated.
    pub fn num_tokens_generated(&self) -> usize {
        self.num_generated.load(Ordering::Relaxed)
    }

    /// Returns the generated text.
    pub fn get_generated_text(&self) -> String {
        self.generated_text.lock().unwrap().clone()
    }
}

struct TokenGeneration<T: Tokenizer> {
    decoder: StreamingDecoder<T>,
    stop_tokens: Vec<u32>,
    stop: Arc<AtomicBool>,
    num_tokens_generated: Arc<AtomicUsize>,
    num_generated: usize,
    generated_text: Arc<Mutex<String>>,
}

impl<T: Tokenizer> TokenGeneration<T> {
    fn new(
        tokenizer: T,
        stop: Arc<AtomicBool>,
        num_tokens_generated: Arc<AtomicUsize>,
        generated_text: Arc<Mutex<String>>,
    ) -> Self {
        Self {
            stop_tokens: tokenizer.stop_ids(),
            decoder: StreamingDecoder::new(tokenizer),
            stop,
            num_tokens_generated,
            num_generated: 0,
            generated_text,
        }
    }

    fn process(&mut self, tokens: Vec<u32>) {
        let mut finished = false;
        let mut generated = Vec::new();

        self.num_generated += tokens.len();

        for token in tokens {
            if self.stop_tokens.contains(&token) {
                finished = true;
            }

            if !finished {
                generated.push(token);
            }
        }

        if !generated.is_empty() {
            if let Some(text) = self.decoder.push_tokens(&generated) {
                let mut gen_text = self.generated_text.lock().unwrap();
                gen_text.push_str(&text);
            }
        }

        if finished {
            self.stop.store(true, Ordering::Relaxed);
        }

        self.num_tokens_generated
            .store(self.num_generated, Ordering::Relaxed);
    }
}
