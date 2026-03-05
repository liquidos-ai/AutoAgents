//! Streaming TTS pipeline.
//!
//! Accepts a token stream (e.g., from an LLM), chunks tokens into sentences via
//! [`SentenceChunker`], synthesizes each sentence concurrently, and yields audio
//! chunks in the correct sequential order.
//!
//! # Architecture
//!
//! Two concurrent tasks run in parallel (inspired by LiveKit Agents):
//!
//! ```text
//! token_stream
//!   │  push_token() each token
//!   ▼
//! SentenceChunker ──► (seq_idx, sentence) ──► mpsc channel
//!   ▼
//! tokio::spawn per sentence → tts.generate_speech(sentence)
//!   ▼
//! results: (seq_idx, AudioChunk) → BTreeMap reorder buffer
//!   ▼  yield in sequential order
//! output Stream<AudioChunk>
//! ```
//!
//! The BTreeMap reorder buffer is critical: sentence 2 may finish TTS before
//! sentence 1, but audio must be emitted in the original text order.

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::Stream;
use tokio::sync::mpsc;

use crate::error::TTSResult;
use crate::provider::TTSSpeechProvider;
use crate::types::{AudioChunk, SpeechRequest};

use super::chunker::{ChunkerConfig, SentenceChunker};

/// A streaming TTS pipeline that chunks token streams into sentences and
/// synthesizes them concurrently with ordered output.
pub struct StreamingTtsPipeline<T: TTSSpeechProvider + Send + Sync + 'static> {
    tts: Arc<T>,
    config: ChunkerConfig,
}

impl<T: TTSSpeechProvider + Send + Sync + 'static> StreamingTtsPipeline<T> {
    /// Create a new pipeline with the given TTS provider and default chunker config.
    pub fn new(tts: Arc<T>) -> Self {
        Self {
            tts,
            config: ChunkerConfig::default(),
        }
    }

    /// Create a new pipeline with a custom chunker configuration.
    pub fn with_config(tts: Arc<T>, config: ChunkerConfig) -> Self {
        Self { tts, config }
    }

    /// Run the pipeline.
    ///
    /// Consumes a stream of text tokens (e.g., from an LLM) and produces an
    /// ordered stream of audio chunks. Each sentence is synthesized concurrently,
    /// but audio is yielded in the original text order.
    ///
    /// # Arguments
    /// * `token_stream` - Stream of text tokens (individual words/subwords from LLM)
    /// * `base_request` - Template `SpeechRequest`; the `text` field is replaced
    ///   per sentence chunk while voice/format/sample_rate are preserved.
    ///
    /// # Returns
    /// A stream of `Result<AudioChunk, TTSError>` in sequential sentence order.
    pub fn run<S>(
        &self,
        token_stream: S,
        base_request: SpeechRequest,
    ) -> OrderedAudioStream
    where
        S: Stream<Item = String> + Send + 'static,
    {
        let (result_tx, result_rx) = mpsc::channel::<(usize, TTSResult<Vec<AudioChunk>>)>(32);
        let tts = Arc::clone(&self.tts);
        let config = self.config.clone();
        let base = base_request;

        // Spawn the producer task: chunks tokens into sentences, dispatches TTS
        tokio::spawn(async move {
            Self::producer_task(token_stream, config, tts, base, result_tx).await;
        });

        OrderedAudioStream::new(result_rx)
    }

    /// Producer task: reads tokens, chunks into sentences, spawns TTS for each.
    async fn producer_task<S>(
        token_stream: S,
        config: ChunkerConfig,
        tts: Arc<T>,
        base_request: SpeechRequest,
        result_tx: mpsc::Sender<(usize, TTSResult<Vec<AudioChunk>>)>,
    ) where
        S: Stream<Item = String> + Send + 'static,
    {
        use futures::StreamExt;

        let mut chunker = SentenceChunker::new(config);
        let mut seq_idx: usize = 0;

        let mut token_stream = std::pin::pin!(token_stream);

        // Process incoming tokens
        while let Some(token) = token_stream.next().await {
            for sentence in chunker.push_token(&token) {
                let idx = seq_idx;
                seq_idx += 1;
                Self::spawn_tts_task(
                    idx,
                    sentence,
                    Arc::clone(&tts),
                    base_request.clone(),
                    result_tx.clone(),
                );
            }
        }

        // LLM stream ended — flush any remaining text
        if let Some(sentence) = chunker.force_flush() {
            let idx = seq_idx;
            Self::spawn_tts_task(idx, sentence, tts, base_request, result_tx);
        }

        // result_tx is dropped here, signalling the consumer that no more
        // sentences will arrive. The spawned TTS tasks still hold clones.
    }

    /// Spawn a TTS synthesis task for a single sentence.
    fn spawn_tts_task(
        seq_idx: usize,
        sentence: String,
        tts: Arc<T>,
        base_request: SpeechRequest,
        result_tx: mpsc::Sender<(usize, TTSResult<Vec<AudioChunk>>)>,
    ) {
        tokio::spawn(async move {
            let request = SpeechRequest {
                text: sentence,
                voice: base_request.voice,
                format: base_request.format,
                sample_rate: base_request.sample_rate,
            };

            let result = match tts.generate_speech(request).await {
                Ok(response) => {
                    // Convert SpeechResponse into a single AudioChunk
                    let chunk = AudioChunk {
                        samples: response.audio.samples,
                        is_final: false, // will be set by the consumer for the last chunk
                    };
                    Ok(vec![chunk])
                }
                Err(e) => Err(e),
            };

            // If the receiver is dropped, we just silently discard
            let _ = result_tx.send((seq_idx, result)).await;
        });
    }
}

/// Ordered audio stream that yields chunks in sequential sentence order.
///
/// Internally buffers out-of-order TTS results in a BTreeMap and yields
/// from the front only when the next expected sequence number is available.
pub struct OrderedAudioStream {
    /// Channel receiving (seq_idx, result) from TTS tasks
    result_rx: mpsc::Receiver<(usize, TTSResult<Vec<AudioChunk>>)>,
    /// Reorder buffer: seq_idx → audio chunks
    buffer: BTreeMap<usize, TTSResult<Vec<AudioChunk>>>,
    /// Next sequence index to yield
    next_seq: usize,
    /// Pending chunks from the current sequence (being drained)
    pending_chunks: Vec<AudioChunk>,
    /// Whether the channel is closed (no more results coming)
    channel_closed: bool,
    /// Whether we've sent the final chunk
    done: bool,
}

impl OrderedAudioStream {
    fn new(result_rx: mpsc::Receiver<(usize, TTSResult<Vec<AudioChunk>>)>) -> Self {
        Self {
            result_rx,
            buffer: BTreeMap::new(),
            next_seq: 0,
            pending_chunks: Vec::new(),
            channel_closed: false,
            done: false,
        }
    }

    /// Try to drain any buffered results that match the next expected sequence.
    fn try_drain_buffered(&mut self) -> Option<TTSResult<AudioChunk>> {
        // First, drain any pending chunks from a previous sequence
        if let Some(chunk) = self.pending_chunks.pop() {
            return Some(Ok(chunk));
        }

        // Check if the next expected sequence is in the buffer
        if let Some(result) = self.buffer.remove(&self.next_seq) {
            self.next_seq += 1;
            match result {
                Ok(mut chunks) => {
                    if chunks.is_empty() {
                        return None;
                    }
                    // Reverse so we can pop from the end efficiently
                    chunks.reverse();
                    self.pending_chunks = chunks;
                    self.pending_chunks.pop().map(Ok)
                }
                Err(e) => Some(Err(e)),
            }
        } else {
            None
        }
    }
}

impl Stream for OrderedAudioStream {
    type Item = TTSResult<AudioChunk>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        if this.done {
            return Poll::Ready(None);
        }

        loop {
            // Try to yield from buffered/pending data
            if let Some(item) = this.try_drain_buffered() {
                return Poll::Ready(Some(item));
            }

            // If the channel is closed and buffer is empty, we're done
            if this.channel_closed && this.buffer.is_empty() && this.pending_chunks.is_empty() {
                this.done = true;
                return Poll::Ready(None);
            }

            // Try to receive more results
            match this.result_rx.poll_recv(cx) {
                Poll::Ready(Some((seq_idx, result))) => {
                    this.buffer.insert(seq_idx, result);
                    // Loop back to try draining
                }
                Poll::Ready(None) => {
                    // Channel closed — all producers done
                    this.channel_closed = true;
                    // Loop back to drain remaining buffer
                }
                Poll::Pending => {
                    return Poll::Pending;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::TTSError;
    use crate::types::{AudioData, AudioFormat, SpeechResponse, VoiceIdentifier};
    use async_trait::async_trait;
    use futures::StreamExt;

    /// A mock TTS provider that returns samples derived from the input text length.
    struct MockTtsProvider;

    #[async_trait]
    impl TTSSpeechProvider for MockTtsProvider {
        async fn generate_speech(&self, request: SpeechRequest) -> TTSResult<SpeechResponse> {
            // Small delay to simulate real TTS
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            let len = request.text.len();
            Ok(SpeechResponse {
                audio: AudioData {
                    samples: vec![0.5_f32; len],
                    channels: 1,
                    sample_rate: 24000,
                },
                text: request.text,
                duration_ms: len as u64,
            })
        }
    }

    #[tokio::test]
    async fn test_pipeline_single_sentence() {
        let tts = Arc::new(MockTtsProvider);
        let pipeline = StreamingTtsPipeline::with_config(
            tts,
            ChunkerConfig {
                min_chunk_chars: 1,
                max_chunk_chars: 250,
            },
        );

        let tokens = futures::stream::iter(vec!["Hello world.".to_string()]);
        let request = SpeechRequest {
            text: String::new(),
            voice: VoiceIdentifier::new("test"),
            format: AudioFormat::Wav,
            sample_rate: Some(24000),
        };

        let mut stream = pipeline.run(tokens, request);
        let mut chunks = Vec::new();
        while let Some(result) = stream.next().await {
            chunks.push(result.unwrap());
        }
        assert!(!chunks.is_empty());
    }

    #[tokio::test]
    async fn test_pipeline_multiple_sentences() {
        let tts = Arc::new(MockTtsProvider);
        let pipeline = StreamingTtsPipeline::with_config(
            tts,
            ChunkerConfig {
                min_chunk_chars: 1,
                max_chunk_chars: 250,
            },
        );

        // Token-by-token, yielding two sentences
        let tokens = futures::stream::iter(
            "Hello world. How are you today?"
                .split_inclusive(' ')
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
        );

        let request = SpeechRequest {
            text: String::new(),
            voice: VoiceIdentifier::new("test"),
            format: AudioFormat::Wav,
            sample_rate: Some(24000),
        };

        let mut stream = pipeline.run(tokens, request);
        let mut chunks = Vec::new();
        while let Some(result) = stream.next().await {
            chunks.push(result.unwrap());
        }
        // Should have at least 2 chunks (one per sentence)
        assert!(chunks.len() >= 2, "Expected >= 2 chunks, got {}", chunks.len());
    }

    #[tokio::test]
    async fn test_pipeline_ordered_output() {
        /// A mock that delays longer for shorter text to test reordering.
        struct SlowShortTts;

        #[async_trait]
        impl TTSSpeechProvider for SlowShortTts {
            async fn generate_speech(&self, request: SpeechRequest) -> TTSResult<SpeechResponse> {
                // Shorter sentences take LONGER → forces out-of-order completion
                let delay_ms = if request.text.len() < 20 { 50 } else { 5 };
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                let marker = request.text.len() as f32;
                Ok(SpeechResponse {
                    audio: AudioData {
                        samples: vec![marker; 10],
                        channels: 1,
                        sample_rate: 24000,
                    },
                    text: request.text,
                    duration_ms: 10,
                })
            }
        }

        let tts = Arc::new(SlowShortTts);
        let pipeline = StreamingTtsPipeline::with_config(
            tts,
            ChunkerConfig {
                min_chunk_chars: 1,
                max_chunk_chars: 250,
            },
        );

        // Two sentences: first is short (slow), second is long (fast)
        let tokens = futures::stream::iter(vec![
            "Hi! ".to_string(),
            "This is a much longer second sentence for testing. ".to_string(),
        ]);

        let request = SpeechRequest {
            text: String::new(),
            voice: VoiceIdentifier::new("test"),
            format: AudioFormat::Wav,
            sample_rate: Some(24000),
        };

        let mut stream = pipeline.run(tokens, request);
        let mut sample_markers = Vec::new();
        while let Some(result) = stream.next().await {
            let chunk = result.unwrap();
            if !chunk.samples.is_empty() {
                sample_markers.push(chunk.samples[0]);
            }
        }

        // Output should be in original order (short sentence first, then long)
        // despite the short sentence taking longer to synthesize
        assert!(sample_markers.len() >= 2);
        // First chunk marker should be for shorter text
        assert!(
            sample_markers[0] < sample_markers[1],
            "Chunks should be in original sentence order: {:?}",
            sample_markers
        );
    }

    #[tokio::test]
    async fn test_pipeline_empty_stream() {
        let tts = Arc::new(MockTtsProvider);
        let pipeline = StreamingTtsPipeline::new(tts);

        let tokens = futures::stream::empty::<String>();
        let request = SpeechRequest {
            text: String::new(),
            voice: VoiceIdentifier::new("test"),
            format: AudioFormat::Wav,
            sample_rate: Some(24000),
        };

        let mut stream = pipeline.run(tokens, request);
        let mut count = 0;
        while let Some(_result) = stream.next().await {
            count += 1;
        }
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_pipeline_tts_error_propagation() {
        struct FailingTts;

        #[async_trait]
        impl TTSSpeechProvider for FailingTts {
            async fn generate_speech(&self, _request: SpeechRequest) -> TTSResult<SpeechResponse> {
                Err(TTSError::Other(
                    "synthesis failed".to_string(),
                    "test".to_string(),
                ))
            }
        }

        let tts = Arc::new(FailingTts);
        let pipeline = StreamingTtsPipeline::with_config(
            tts,
            ChunkerConfig {
                min_chunk_chars: 1,
                max_chunk_chars: 250,
            },
        );

        let tokens = futures::stream::iter(vec!["Hello world.".to_string()]);
        let request = SpeechRequest {
            text: String::new(),
            voice: VoiceIdentifier::new("test"),
            format: AudioFormat::Wav,
            sample_rate: Some(24000),
        };

        let mut stream = pipeline.run(tokens, request);
        let result = stream.next().await;
        assert!(result.is_some());
        assert!(result.unwrap().is_err());
    }
}
