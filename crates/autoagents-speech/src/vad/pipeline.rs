use super::VadEngine;
use super::config::VadSttConfig;
use super::error::VadError;
use super::segmenter::{SpeechSegment, VadSegmenter};
use crate::{STTError, STTSpeechProvider, TranscriptionRequest, TranscriptionResponse};

#[derive(Debug, Clone)]
pub struct SegmentTranscription {
    pub segment: SpeechSegment,
    pub transcription: TranscriptionResponse,
}

#[derive(Debug, thiserror::Error)]
pub enum VadPipelineError {
    #[error("VAD error: {0}")]
    Vad(#[from] VadError),
    #[error("STT error: {0}")]
    Stt(#[from] STTError),
}

pub struct VadSttPipeline<V: VadEngine, S: STTSpeechProvider> {
    segmenter: VadSegmenter<V>,
    stt: S,
    config: VadSttConfig,
}

impl<V: VadEngine, S: STTSpeechProvider> VadSttPipeline<V, S> {
    pub fn new(segmenter: VadSegmenter<V>, stt: S, config: VadSttConfig) -> Self {
        Self {
            segmenter,
            stt,
            config,
        }
    }

    pub fn window_samples(&self) -> usize {
        self.segmenter.window_samples()
    }

    pub fn window_ms(&self) -> u32 {
        self.segmenter.window_ms()
    }

    pub async fn process_audio(
        &mut self,
        audio: &crate::AudioData,
    ) -> Result<Vec<SegmentTranscription>, VadPipelineError> {
        let segments = self.segmenter.process_audio(audio)?;
        self.transcribe_segments(segments).await
    }

    pub async fn finalize(&mut self) -> Result<Option<SegmentTranscription>, VadPipelineError> {
        let segment = match self.segmenter.finalize()? {
            Some(segment) => segment,
            None => return Ok(None),
        };

        let transcription = self.transcribe_segment(&segment).await?;
        Ok(Some(SegmentTranscription {
            segment,
            transcription,
        }))
    }

    async fn transcribe_segments(
        &self,
        segments: Vec<SpeechSegment>,
    ) -> Result<Vec<SegmentTranscription>, VadPipelineError> {
        let mut transcriptions = Vec::with_capacity(segments.len());
        for segment in segments {
            let transcription = self.transcribe_segment(&segment).await?;
            transcriptions.push(SegmentTranscription {
                segment,
                transcription,
            });
        }
        Ok(transcriptions)
    }

    async fn transcribe_segment(
        &self,
        segment: &SpeechSegment,
    ) -> Result<TranscriptionResponse, VadPipelineError> {
        let request = TranscriptionRequest {
            audio: segment.audio.clone(),
            language: self.config.language.clone(),
            include_timestamps: self.config.include_timestamps,
        };
        let response = self.stt.transcribe(request).await?;
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vad::VadOutput;
    use crate::vad::config::SegmenterConfig;
    use crate::vad::error::VadResult;
    use crate::vad::{VadEngine, VadSegmenter};
    use crate::{AudioData, STTResult};
    use async_trait::async_trait;

    struct MockVad;

    impl VadEngine for MockVad {
        fn sample_rate(&self) -> u32 {
            16_000
        }

        fn reset(&mut self) {}

        fn compute(&mut self, samples: &[f32]) -> VadResult<VadOutput> {
            let avg = samples.iter().map(|v| v.abs()).sum::<f32>() / samples.len() as f32;
            let prob = if avg > 0.2 { 0.9 } else { 0.1 };
            Ok(VadOutput { probability: prob })
        }
    }

    struct MockStt;

    #[async_trait]
    impl STTSpeechProvider for MockStt {
        async fn transcribe(
            &self,
            request: TranscriptionRequest,
        ) -> STTResult<TranscriptionResponse> {
            Ok(TranscriptionResponse {
                text: format!("{} samples", request.audio.samples.len()),
                timestamps: None,
                duration_ms: 1,
            })
        }
    }

    #[tokio::test]
    async fn pipeline_emits_transcriptions() {
        let config = SegmenterConfig::default()
            .with_window_ms(100)
            .with_min_speech_ms(100)
            .with_min_silence_ms(100);

        let segmenter = VadSegmenter::new(MockVad, config).unwrap();
        let mut pipeline = VadSttPipeline::new(segmenter, MockStt, VadSttConfig::default());
        let samples = vec![0.8; pipeline.window_samples() * 2];
        let audio = AudioData {
            samples,
            sample_rate: 16_000,
            channels: 1,
        };

        let results = pipeline.process_audio(&audio).await.unwrap();
        assert_eq!(results.len(), 0);

        let silence = vec![0.0; pipeline.window_samples() * 2];
        let silence_audio = AudioData {
            samples: silence,
            sample_rate: 16_000,
            channels: 1,
        };
        let results = pipeline.process_audio(&silence_audio).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].transcription.text.contains("samples"));
    }
}
