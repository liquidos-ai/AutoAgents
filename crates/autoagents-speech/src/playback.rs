use rodio::{OutputStream, OutputStreamHandle, Sink};
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum AudioPlayerError {
    #[error("Failed to initialize audio output stream")]
    InitFailed,
    #[error("Failed to create audio sink")]
    FailedToCreateAudioSink,
}

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
    pub fn try_new() -> Result<Self, AudioPlayerError> {
        let (_stream, stream_handle) = match OutputStream::try_default() {
            Ok(s) => s,
            Err(_) => return Err(AudioPlayerError::InitFailed),
        };

        let stream_handle = Arc::new(stream_handle);
        let sink = match Sink::try_new(&stream_handle) {
            Ok(s) => s,
            Err(_) => return Err(AudioPlayerError::FailedToCreateAudioSink),
        };

        Ok(AudioPlayer {
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
