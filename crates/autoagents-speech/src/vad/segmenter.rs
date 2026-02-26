use super::VadEngine;
use super::config::SegmenterConfig;
use super::error::{VadError, VadResult};
use super::result::{VadOutput, VadStatus};
use crate::{AudioData, SharedAudioData};
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentEndReason {
    Silence,
    MaxDuration,
    EndOfStream,
}

#[derive(Debug, Clone)]
pub struct SpeechSegment {
    pub audio: SharedAudioData,
    pub start_ms: u64,
    pub end_ms: u64,
    pub reason: SegmentEndReason,
}

pub struct VadSegmenter<E: VadEngine> {
    vad: E,
    config: SegmenterConfig,
    sample_rate: u32,
    window_samples: usize,
    pre_roll_max_samples: usize,
    pending_samples: Vec<f32>,
    pre_roll: VecDeque<f32>,
    pending_speech: Vec<f32>,
    speech_buffer: Vec<f32>,
    in_speech: bool,
    pending_speech_ms: u32,
    silence_ms: u32,
    segment_ms: u32,
    processed_samples: u64,
    segment_start_sample: Option<u64>,
}

impl<E: VadEngine> VadSegmenter<E> {
    pub fn new(mut vad: E, config: SegmenterConfig) -> VadResult<Self> {
        let sample_rate = vad.sample_rate();
        if sample_rate == 0 {
            return Err(VadError::InvalidInput(
                "sample rate must be greater than zero".to_string(),
            ));
        }
        if config.window_ms == 0 {
            return Err(VadError::InvalidInput(
                "window_ms must be greater than zero".to_string(),
            ));
        }
        if config.max_segment_ms < config.window_ms {
            return Err(VadError::InvalidInput(
                "max_segment_ms must be >= window_ms".to_string(),
            ));
        }
        let window_samples = config.window_samples(sample_rate);

        let pre_roll_max_samples =
            ((sample_rate as f32 * config.pre_roll_ms as f32) / 1000.0) as usize;

        vad.reset();

        Ok(Self {
            vad,
            config,
            sample_rate,
            window_samples,
            pre_roll_max_samples,
            pending_samples: Vec::new(),
            pre_roll: VecDeque::new(),
            pending_speech: Vec::new(),
            speech_buffer: Vec::new(),
            in_speech: false,
            pending_speech_ms: 0,
            silence_ms: 0,
            segment_ms: 0,
            processed_samples: 0,
            segment_start_sample: None,
        })
    }

    pub fn window_samples(&self) -> usize {
        self.window_samples
    }

    pub fn window_ms(&self) -> u32 {
        self.config.window_ms
    }

    /// Returns true while speech is actively being accumulated (between speech onset and silence
    /// timeout). Use this to gate downstream processing on non-silent audio only.
    pub fn in_speech(&self) -> bool {
        self.in_speech
    }

    pub fn process_audio(&mut self, audio: &AudioData) -> VadResult<Vec<SpeechSegment>> {
        if audio.sample_rate != self.sample_rate {
            return Err(VadError::InvalidInput(format!(
                "expected {} Hz audio, got {} Hz",
                self.sample_rate, audio.sample_rate
            )));
        }
        if audio.channels != 1 {
            return Err(VadError::InvalidInput(format!(
                "expected mono audio, got {} channels",
                audio.channels
            )));
        }

        self.process_samples(&audio.samples)
    }

    pub fn process_samples(&mut self, samples: &[f32]) -> VadResult<Vec<SpeechSegment>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        self.pending_samples.extend_from_slice(samples);
        let mut segments = Vec::new();

        while self.pending_samples.len() >= self.window_samples {
            let window: Vec<f32> = self.pending_samples.drain(..self.window_samples).collect();
            let segment = self.process_window(&window)?;
            if let Some(segment) = segment {
                segments.push(segment);
            }
            self.processed_samples += self.window_samples as u64;
        }

        Ok(segments)
    }

    pub fn finalize(&mut self) -> VadResult<Option<SpeechSegment>> {
        if !self.in_speech || self.speech_buffer.is_empty() {
            self.reset_state();
            return Ok(None);
        }

        let end_sample = self.processed_samples + self.pending_samples.len() as u64;
        if !self.pending_samples.is_empty() {
            self.speech_buffer
                .extend_from_slice(&std::mem::take(&mut self.pending_samples));
        }
        let start_sample = self.segment_start_sample.unwrap_or(end_sample);
        let segment = SpeechSegment {
            audio: SharedAudioData::new(AudioData {
                samples: std::mem::take(&mut self.speech_buffer),
                sample_rate: self.sample_rate,
                channels: 1,
            }),
            start_ms: samples_to_ms(start_sample, self.sample_rate),
            end_ms: samples_to_ms(end_sample, self.sample_rate),
            reason: SegmentEndReason::EndOfStream,
        };

        self.reset_state();
        Ok(Some(segment))
    }

    fn process_window(&mut self, window: &[f32]) -> VadResult<Option<SpeechSegment>> {
        if !self.in_speech {
            self.push_pre_roll(window);
        }

        let vad_output: VadOutput = self.vad.compute(window)?;
        let status = vad_output.status(self.config.thresholds);

        match status {
            VadStatus::Speech => self.on_speech(window),
            VadStatus::Silence => self.on_silence(window),
            VadStatus::Unknown => {
                if self.in_speech {
                    self.on_speech(window)
                } else {
                    self.on_silence(window)
                }
            }
        }
    }

    fn on_speech(&mut self, window: &[f32]) -> VadResult<Option<SpeechSegment>> {
        self.silence_ms = 0;
        if !self.in_speech {
            self.pending_speech_ms = self.pending_speech_ms.saturating_add(self.config.window_ms);
            self.pending_speech.extend_from_slice(window);

            if self.pending_speech_ms >= self.config.min_speech_ms {
                self.start_segment();
            }

            return Ok(None);
        }

        self.speech_buffer.extend_from_slice(window);
        self.segment_ms = self.segment_ms.saturating_add(self.config.window_ms);
        if self.segment_ms >= self.config.max_segment_ms {
            return Ok(self.finish_segment(SegmentEndReason::MaxDuration));
        }

        Ok(None)
    }

    fn on_silence(&mut self, window: &[f32]) -> VadResult<Option<SpeechSegment>> {
        if !self.in_speech {
            self.pending_speech.clear();
            self.pending_speech_ms = 0;
            return Ok(None);
        }

        self.silence_ms = self.silence_ms.saturating_add(self.config.window_ms);
        self.segment_ms = self.segment_ms.saturating_add(self.config.window_ms);
        self.speech_buffer.extend_from_slice(window);

        if self.segment_ms >= self.config.max_segment_ms {
            return Ok(self.finish_segment(SegmentEndReason::MaxDuration));
        }

        if self.silence_ms >= self.config.min_silence_ms {
            return Ok(self.finish_segment(SegmentEndReason::Silence));
        }

        Ok(None)
    }

    fn start_segment(&mut self) {
        let pre_roll_samples = self.pre_roll.len() as u64;
        let pending_samples = self.pending_speech.len() as u64;
        let end_sample = self.processed_samples + self.window_samples as u64;

        self.in_speech = true;
        self.segment_ms = self.pending_speech_ms;
        self.pending_speech_ms = 0;
        self.speech_buffer.clear();
        self.speech_buffer.extend(self.pre_roll.drain(..));
        self.speech_buffer.extend_from_slice(&self.pending_speech);
        self.pending_speech.clear();

        let start_sample = end_sample.saturating_sub(pre_roll_samples + pending_samples);
        self.segment_start_sample = Some(start_sample);
    }

    fn finish_segment(&mut self, reason: SegmentEndReason) -> Option<SpeechSegment> {
        if self.speech_buffer.is_empty() {
            self.reset_state();
            return None;
        }

        let end_sample = self.processed_samples + self.window_samples as u64;
        let start_sample = self.segment_start_sample.unwrap_or(end_sample);
        let segment = SpeechSegment {
            audio: SharedAudioData::new(AudioData {
                samples: std::mem::take(&mut self.speech_buffer),
                sample_rate: self.sample_rate,
                channels: 1,
            }),
            start_ms: samples_to_ms(start_sample, self.sample_rate),
            end_ms: samples_to_ms(end_sample, self.sample_rate),
            reason,
        };

        self.reset_state();
        Some(segment)
    }

    fn push_pre_roll(&mut self, window: &[f32]) {
        if self.pre_roll_max_samples == 0 {
            return;
        }

        self.pre_roll.extend(window.iter().copied());
        while self.pre_roll.len() > self.pre_roll_max_samples {
            self.pre_roll.pop_front();
        }
    }

    fn reset_state(&mut self) {
        self.in_speech = false;
        self.pending_speech_ms = 0;
        self.silence_ms = 0;
        self.segment_ms = 0;
        self.pending_speech.clear();
        self.pre_roll.clear();
        self.speech_buffer.clear();
        self.segment_start_sample = None;
        self.vad.reset();
    }
}

fn samples_to_ms(samples: u64, sample_rate: u32) -> u64 {
    if sample_rate == 0 {
        return 0;
    }
    (samples * 1000) / sample_rate as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vad::VadEngine;
    use crate::vad::VadOutput;

    struct MockVad {
        sample_rate: u32,
        threshold: f32,
    }

    impl MockVad {
        fn new(sample_rate: u32, threshold: f32) -> Self {
            Self {
                sample_rate,
                threshold,
            }
        }
    }

    impl VadEngine for MockVad {
        fn sample_rate(&self) -> u32 {
            self.sample_rate
        }

        fn reset(&mut self) {}

        fn compute(&mut self, samples: &[f32]) -> VadResult<VadOutput> {
            let avg = samples.iter().map(|v| v.abs()).sum::<f32>() / samples.len() as f32;
            let prob = if avg >= self.threshold { 0.9 } else { 0.1 };
            Ok(VadOutput { probability: prob })
        }
    }

    #[test]
    fn segments_speech_from_silence() {
        let vad = MockVad::new(16_000, 0.2);
        let config = SegmenterConfig::default()
            .with_window_ms(100)
            .with_min_speech_ms(200)
            .with_min_silence_ms(200)
            .with_pre_roll_ms(100)
            .with_max_segment_ms(2_000);

        let mut segmenter = VadSegmenter::new(vad, config).unwrap();

        let silence = vec![0.0; segmenter.window_samples()];
        let speech = vec![0.8; segmenter.window_samples()];

        let mut segments = Vec::new();
        segments.extend(segmenter.process_samples(&silence).unwrap());
        segments.extend(segmenter.process_samples(&speech).unwrap());
        segments.extend(segmenter.process_samples(&speech).unwrap());
        segments.extend(segmenter.process_samples(&silence).unwrap());
        segments.extend(segmenter.process_samples(&silence).unwrap());

        assert_eq!(segments.len(), 1);
        let segment = &segments[0];
        assert!(segment.audio.samples.len() >= segmenter.window_samples());
        assert_eq!(segment.reason, SegmentEndReason::Silence);
        assert!(segment.end_ms >= segment.start_ms);
    }

    #[test]
    fn finalize_emits_segment() {
        let vad = MockVad::new(16_000, 0.2);
        let config = SegmenterConfig::default()
            .with_window_ms(100)
            .with_min_speech_ms(100)
            .with_min_silence_ms(500);

        let mut segmenter = VadSegmenter::new(vad, config).unwrap();
        let speech = vec![0.8; segmenter.window_samples()];

        segmenter.process_samples(&speech).unwrap();
        segmenter.process_samples(&speech).unwrap();

        let segment = segmenter.finalize().unwrap().unwrap();
        assert_eq!(segment.reason, SegmentEndReason::EndOfStream);
        assert!(!segment.audio.samples.is_empty());
    }
}
