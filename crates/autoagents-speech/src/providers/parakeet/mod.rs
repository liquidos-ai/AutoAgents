//! Parakeet STT provider for AutoAgents Speech framework
//!
//! This module provides Parakeet (NVIDIA) implementations for the AutoAgents STT trait system.
//!
//! # Supported Models
//!
//! - **TDT**: Multilingual model with 25 language support and timestamp capabilities
//! - **Nemotron**: Streaming-optimized model for real-time transcription (English only)
//! - **EOU**: Real-time streaming with end-of-utterance detection (English only)
//!
//! # Examples
//!
//! ## TDT (Multilingual with timestamps)
//!
//! ```no_run
//! use autoagents_speech::providers::parakeet::{Parakeet, ParakeetConfig, ModelVariant};
//! use autoagents_speech::{STTSpeechProvider, TranscriptionRequest, AudioData};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create provider with TDT model
//!     let config = ParakeetConfig::new(ModelVariant::TDT, "./models/tdt");
//!     let provider = Parakeet::new(config)?;
//!
//!     // Transcribe with timestamps
//!     let request = TranscriptionRequest {
//!         audio: AudioData {
//!             samples: vec![0.0; 16000], // 1 second of audio
//!             sample_rate: 16000,
//!             channels: 1,
//!         },
//!         language: None, // Auto-detect
//!         include_timestamps: true,
//!     };
//!
//!     let response = provider.transcribe(request).await?;
//!     println!("Transcription: {}", response.text);
//!
//!     if let Some(timestamps) = response.timestamps {
//!         for token in timestamps {
//!             println!("[{:.2}s - {:.2}s] {}", token.start, token.end, token.text);
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Nemotron (Streaming for real-time)
//!
//! ```no_run
//! use autoagents_speech::providers::parakeet::{Parakeet, ParakeetConfig, ModelVariant};
//! use autoagents_speech::{STTSpeechProvider, TranscriptionRequest, AudioData};
//! use futures::StreamExt;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create provider with Nemotron model
//!     let config = ParakeetConfig::new(ModelVariant::Nemotron, "./models/nemotron");
//!     let provider = Parakeet::new(config)?;
//!
//!     // Stream transcription
//!     let request = TranscriptionRequest {
//!         audio: AudioData {
//!             samples: vec![0.0; 16000 * 10], // 10 seconds of audio
//!             sample_rate: 16000,
//!             channels: 1,
//!         },
//!         language: Some("en".to_string()),
//!         include_timestamps: false,
//!     };
//!
//!     let mut stream = provider.transcribe_stream(request).await?;
//!
//!     while let Some(chunk) = stream.next().await {
//!         let chunk = chunk?;
//!         print!("{}", chunk.text);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## EOU (Streaming with End-of-Utterance Detection)
//!
//! ```no_run
//! use autoagents_speech::providers::parakeet::{Parakeet, ParakeetConfig, ModelVariant};
//! use autoagents_speech::{STTSpeechProvider, TranscriptionRequest, AudioData};
//! use futures::StreamExt;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create provider with EOU model
//!     let config = ParakeetConfig::new(ModelVariant::EOU, "./models/eou");
//!     let provider = Parakeet::new(config)?;
//!
//!     // Stream transcription with EOU detection
//!     let request = TranscriptionRequest {
//!         audio: AudioData {
//!             samples: vec![0.0; 16000 * 10], // 10 seconds of audio
//!             sample_rate: 16000,
//!             channels: 1,
//!         },
//!         language: Some("en".to_string()),
//!         include_timestamps: false,
//!     };
//!
//!     let mut stream = provider.transcribe_stream(request).await?;
//!
//!     while let Some(chunk) = stream.next().await {
//!         let chunk = chunk?;
//!         
//!         if chunk.is_final {
//!             // End of utterance detected
//!             println!("\n[EOU detected] {}", chunk.text);
//!         } else {
//!             print!("{}", chunk.text);
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```

pub mod config;
pub mod error;
pub mod model;

mod provider;
mod stt;

// Re-exports
pub use config::ParakeetConfig;
pub use error::{ParakeetError, Result};
pub use model::ModelVariant;
pub use provider::Parakeet;
