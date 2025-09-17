use super::super::tokenizer::Tokenizer;
use crate::model::llama::generation::stream_sender::StreamSender;
use crate::model::llama::generation::streaming::StreamingDecoder;
use crate::utils::{spawn_future, CustomMutex};
use autoagents_llm::chat::{StreamChoice, StreamDelta, StreamResponse};
use burn::prelude::Backend;
use burn::tensor::{Int, Tensor};
use futures_util::SinkExt;
use log::error;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::mpsc;
#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::mpsc::Sender;

#[cfg(target_arch = "wasm32")]
use futures::channel::mpsc;
#[cfg(target_arch = "wasm32")]
use futures::channel::mpsc::{unbounded, UnboundedSender as Sender};

#[derive(Clone)]
/// The text generation context, used to check when a stop token has been reached.
pub struct GenerationContext<B: Backend> {
    pub tokens: Tensor<B, 1, Int>,
    num_tokens: usize,
    stop: Arc<AtomicBool>,
    num_generated: Arc<AtomicUsize>,
    sender: Sender<Tensor<B, 1, Int>>,
    generated_text: Arc<CustomMutex<String>>,
}

impl<B: Backend> GenerationContext<B> {
    /// Create a new generation context.
    pub async fn new<T: Tokenizer + 'static>(
        max_sample_len: usize,
        tokenizer: T,
        device: &B::Device,
        emitter: Option<StreamSender>,
    ) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let (sender, mut receiver) = mpsc::channel::<Tensor<B, 1, Int>>(100);
        #[cfg(target_arch = "wasm32")]
        let (sender, mut receiver) = unbounded::<Tensor<B, 1, Int>>();

        let stop = Arc::new(AtomicBool::new(false));
        let num_generated = Arc::new(AtomicUsize::new(0));
        let generated_text = Arc::new(CustomMutex::new(String::new()));

        let mut generation = TokenGeneration::new(
            tokenizer,
            stop.clone(),
            num_generated.clone(),
            generated_text.clone(),
            emitter,
        );

        spawn_future(async move {
            #[cfg(not(target_arch = "wasm32"))]
            while let Some(tokens) = receiver.recv().await {
                let tokens = tokens
                    .into_data()
                    .convert::<u32>()
                    .into_vec::<u32>()
                    .unwrap();

                generation.process(tokens).await;
            }

            #[cfg(target_arch = "wasm32")]
            {
                use futures::StreamExt;
                while let Some(tokens) = receiver.next().await {
                    let data = tokens.into_data_async().await;
                    // Convert to Vec<u32>
                    let tokens = match data.convert::<u32>().into_vec::<u32>() {
                        Ok(v) => v,
                        Err(e) => {
                            error!("failed to convert tensor to Vec<u32> (wasm): {:?}", e);
                            continue;
                        }
                    };

                    generation.process(tokens).await;
                }
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
    pub async fn update(&mut self, tokens: Tensor<B, 1, Int>) {
        self.append(tokens.clone());

        if !self.should_stop() {
            #[cfg(not(target_arch = "wasm32"))]
            {
                if let Err(e) = self.sender.send(tokens).await {
                    error!("error sending update: {:?}", e.to_string());
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                if let Err(e) = self.sender.unbounded_send(tokens) {
                    error!("error sending update: {:?}", e.to_string());
                }
            }
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
    pub async fn get_generated_text(&self) -> String {
        self.generated_text.lock().await.clone()
    }
}

struct TokenGeneration<T: Tokenizer> {
    decoder: StreamingDecoder<T>,
    stop_tokens: Vec<u32>,
    stop: Arc<AtomicBool>,
    num_tokens_generated: Arc<AtomicUsize>,
    num_generated: usize,
    generated_text: Arc<CustomMutex<String>>,
    emitter: Option<StreamSender>,
}

impl<T: Tokenizer> TokenGeneration<T> {
    fn new(
        tokenizer: T,
        stop: Arc<AtomicBool>,
        num_tokens_generated: Arc<AtomicUsize>,
        generated_text: Arc<CustomMutex<String>>,
        emitter: Option<StreamSender>,
    ) -> Self {
        Self {
            stop_tokens: tokenizer.stop_ids(),
            decoder: StreamingDecoder::new(tokenizer),
            stop,
            num_tokens_generated,
            num_generated: 0,
            generated_text,
            emitter,
        }
    }

    async fn process(&mut self, tokens: Vec<u32>) {
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
                let mut gen_text = self.generated_text.lock().await;
                gen_text.push_str(&text);
                let response = StreamResponse {
                    choices: vec![StreamChoice {
                        delta: StreamDelta {
                            content: Some(text),
                            tool_calls: None,
                        },
                    }],
                    usage: None,
                };
                if let Some(emitter) = self.emitter.as_ref() {
                    emitter.send(Ok(response)).await;
                    #[cfg(not(target_arch = "wasm32"))]
                    tokio::task::yield_now().await;
                    #[cfg(target_arch = "wasm32")]
                    {
                        use futures_util::future::FutureExt;
                        futures_util::future::ready(()).then(|_| async {}).await;
                    }
                }
            }
        }

        if finished {
            self.stop.store(true, Ordering::Relaxed);
        }

        self.num_tokens_generated
            .store(self.num_generated, Ordering::Relaxed);
    }
}
