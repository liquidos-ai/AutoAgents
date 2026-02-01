//! Shared audio playback utilities for examples
//!
//! This module provides optional audio playback functionality using rodio.
//! It gracefully handles cases where audio devices are not available.

use rodio::{OutputStream, OutputStreamHandle, Sink};
use std::sync::Arc;

/// Audio player that handles playback of audio samples
pub struct AudioPlayer {
    _stream: OutputStream,
    #[allow(dead_code)]
    stream_handle: Arc<OutputStreamHandle>,
    sink: Sink,
}

impl AudioPlayer {
    /// Try to create a new audio player
    /// Returns None if no audio device is available
    pub fn try_new() -> Option<Self> {
        let (_stream, stream_handle) = match OutputStream::try_default() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("⚠️  No audio device available: {}", e);
                eprintln!("   Audio playback disabled. Will save to file only.");
                return None;
            }
        };

        let stream_handle = Arc::new(stream_handle);
        let sink = match Sink::try_new(&stream_handle) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("⚠️  Failed to create audio sink: {}", e);
                eprintln!("   Audio playback disabled. Will save to file only.");
                return None;
            }
        };

        Some(AudioPlayer {
            _stream,
            stream_handle,
            sink,
        })
    }

    /// Play audio samples (mono)
    pub fn play_samples(&self, samples: &[f32], sample_rate: u32) {
        let source = rodio::buffer::SamplesBuffer::new(1, sample_rate, samples.to_vec());
        self.sink.append(source);
    }

    /// Wait until all audio has finished playing
    pub fn wait_until_end(&self) {
        self.sink.sleep_until_end();
    }

    /// Check if audio is currently playing
    #[allow(dead_code)]
    pub fn is_playing(&self) -> bool {
        !self.sink.empty()
    }
}

/// Helper to check if user wants playback disabled via environment variable
pub fn is_playback_enabled() -> bool {
    std::env::var("NO_PLAY")
        .map(|v| v != "1" && v.to_lowercase() != "true")
        .unwrap_or(true)
}

/// Print instructions for disabling playback
pub fn print_playback_info() {
    if is_playback_enabled() {
        println!("🔊 Audio playback enabled");
        println!("   To disable: set NO_PLAY=1 environment variable");
    } else {
        println!("🔇 Audio playback disabled (NO_PLAY=1)");
    }
    println!();
}
