use rodio::{OutputStream, OutputStreamBuilder, Sink};

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
    sink: Sink,
}

impl AudioPlayer {
    /// Try to create a new audio player
    /// Returns None if no audio device is available
    pub fn try_new() -> Result<Self, AudioPlayerError> {
        let stream = match OutputStreamBuilder::open_default_stream() {
            Ok(s) => s,
            Err(_) => return Err(AudioPlayerError::InitFailed),
        };

        let sink = Sink::connect_new(stream.mixer());

        Ok(AudioPlayer {
            _stream: stream,
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
