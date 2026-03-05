//! TTS utilities for sentence chunking and streaming pipeline.
//!
//! This module provides:
//! - [`SentenceChunker`]: Splits text into natural sentence boundaries for TTS
//! - [`StreamingTtsPipeline`]: Concurrent pipeline that chunks LLM token streams
//!   and synthesizes speech in parallel with reordered output

mod chunker;
mod streaming_pipeline;

pub use chunker::{ChunkerConfig, SentenceChunker};
pub use streaming_pipeline::StreamingTtsPipeline;
