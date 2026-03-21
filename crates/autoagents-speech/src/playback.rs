use std::num::{NonZeroU16, NonZeroU32};

use rodio::{DeviceSinkBuilder, MixerDeviceSink, Player, buffer::SamplesBuffer};

#[derive(Debug, thiserror::Error)]
pub enum AudioPlayerError {
    #[error("Failed to initialize audio output stream")]
    InitFailed,
}

/// Audio player that handles playback of audio samples
pub struct AudioPlayer {
    _stream: MixerDeviceSink,
    sink: Player,
}

impl AudioPlayer {
    /// Try to create a new audio player
    /// Returns None if no audio device is available
    pub fn try_new() -> Result<Self, AudioPlayerError> {
        let mut stream = match DeviceSinkBuilder::open_default_sink() {
            Ok(s) => s,
            Err(_) => return Err(AudioPlayerError::InitFailed),
        };
        stream.log_on_drop(false);

        let sink = Player::connect_new(stream.mixer());

        Ok(AudioPlayer {
            _stream: stream,
            sink,
        })
    }

    /// Play audio samples (mono)
    pub fn play_samples(&self, samples: &[f32], sample_rate: u32) {
        let Some(sample_rate) = NonZeroU32::new(sample_rate) else {
            return;
        };
        let channels =
            NonZeroU16::new(1).expect("mono playback channel count must always be non-zero");
        let source = SamplesBuffer::new(channels, sample_rate, samples.to_vec());
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
