//! Voice Activity Detection (VAD) utilities.

mod config;
mod error;
mod pipeline;
mod result;
mod segmenter;
mod silero;

pub use config::{SegmenterConfig, VadConfig, VadDevice, VadSttConfig};
pub use error::{VadError, VadResult};
pub use pipeline::{SegmentTranscription, VadPipelineError, VadSttPipeline};
pub use result::{VadOutput, VadStatus, VadThresholds};
pub use segmenter::{SegmentEndReason, SpeechSegment, VadSegmenter};
pub use silero::SileroVad;
#[cfg(feature = "vad-compare")]
pub use silero::VadState;

/// Trait abstraction for VAD engines.
pub trait VadEngine: Send {
    fn sample_rate(&self) -> u32;
    fn reset(&mut self);
    fn compute(&mut self, samples: &[f32]) -> VadResult<VadOutput>;
}
