use super::error::{VadError, VadResult};
use super::result::VadThresholds;

/// Configuration for the Silero VAD engine.
#[derive(Debug, Clone)]
pub struct VadConfig {
    pub sample_rate: u32,
}

impl VadConfig {
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }

    pub fn validate(&self) -> VadResult<()> {
        if self.sample_rate != 8_000 && self.sample_rate != 16_000 {
            return Err(VadError::UnsupportedSampleRate(self.sample_rate));
        }
        Ok(())
    }
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
        }
    }
}

/// Configuration for VAD-driven segmentation.
#[derive(Debug, Clone)]
pub struct SegmenterConfig {
    pub window_ms: u32,
    pub min_speech_ms: u32,
    pub min_silence_ms: u32,
    pub pre_roll_ms: u32,
    pub max_segment_ms: u32,
    pub thresholds: VadThresholds,
}

impl SegmenterConfig {
    pub fn window_samples(&self, sample_rate: u32) -> usize {
        let samples = (sample_rate as f32 * self.window_ms as f32 / 1000.0).round() as usize;
        samples.max(1)
    }

    pub fn with_window_ms(mut self, ms: u32) -> Self {
        self.window_ms = ms;
        self
    }

    pub fn with_min_speech_ms(mut self, ms: u32) -> Self {
        self.min_speech_ms = ms;
        self
    }

    pub fn with_min_silence_ms(mut self, ms: u32) -> Self {
        self.min_silence_ms = ms;
        self
    }

    pub fn with_pre_roll_ms(mut self, ms: u32) -> Self {
        self.pre_roll_ms = ms;
        self
    }

    pub fn with_max_segment_ms(mut self, ms: u32) -> Self {
        self.max_segment_ms = ms;
        self
    }

    pub fn with_thresholds(mut self, thresholds: VadThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            window_ms: 100,
            min_speech_ms: 300,
            min_silence_ms: 500,
            pre_roll_ms: 200,
            max_segment_ms: 30_000,
            thresholds: VadThresholds::default(),
        }
    }
}

/// Configuration for the VAD + STT pipeline.
#[derive(Debug, Clone, Default)]
pub struct VadSttConfig {
    pub language: Option<String>,
    pub include_timestamps: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_sample_rate() {
        let config = VadConfig::new(22_050);
        let err = config.validate().unwrap_err();
        match err {
            VadError::UnsupportedSampleRate(rate) => assert_eq!(rate, 22_050),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn accepts_valid_sample_rate() {
        let config = VadConfig::new(16_000);
        assert!(config.validate().is_ok());
    }
}
