//! Conversion utilities between pocket-tts types and autoagents-speech types

use crate::AudioData;

/// Convert pocket-tts audio samples to AudioData
pub fn samples_to_audio_data(samples: Vec<f32>, sample_rate: u32) -> AudioData {
    AudioData {
        samples,
        channels: 1, // Pocket-TTS outputs mono audio
        sample_rate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_samples_to_audio_data() {
        let samples = vec![0.0, 0.5, -0.5, 1.0];
        let audio = samples_to_audio_data(samples.clone(), 24000);

        assert_eq!(audio.samples, samples);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.sample_rate, 24000);
    }
}
