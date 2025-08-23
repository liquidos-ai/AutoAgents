use anyhow::Result;
pub mod actor;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::{collections::VecDeque, path::PathBuf};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub struct STTProcessor {
    context: WhisperContext,
}

impl STTProcessor {
    pub async fn new(model_path: PathBuf) -> Result<Self> {
        println!("MODEL path:{:?}", model_path.to_str());
        let context = WhisperContext::new_with_params(
            model_path.to_str().unwrap(),
            WhisperContextParameters::default(),
        )
        .expect("failed to load model");

        Ok(Self { context })
    }

    pub fn transcribe_audio(&mut self, pcm_data: &[f32]) -> Result<String> {
        println!("🎵 Processing {} audio samples for STT", pcm_data.len());
        // Set full parameters
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_translate(false);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_language(Some("en"));
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        // Create a state and run the model
        println!("Running whisper transcription...");
        let mut state = self.context.create_state().expect("failed to create state");
        state
            .full(params, &pcm_data[..])
            .expect("failed to run model");

        // Retrieve and print the segments
        let num_segments = state.full_n_segments();
        println!("\nTranscription results ({} segments):", num_segments);

        // Iterate through the segments using the iterator
        let mut segment_string = String::new();
        for segment in state.as_iter() {
            segment_string.push_str(segment.to_str().unwrap_or_default());
        }

        Ok(segment_string)
    }

    pub fn process_file(&mut self, file_path: &str) -> Result<String> {
        let reader = hound::WavReader::open(file_path)?;

        let spec = reader.spec();
        let samples: Vec<f32> = if spec.sample_format == hound::SampleFormat::Int {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i16>()
                .map(|s| s.expect("failed to read sample") as f32 / max_val)
                .collect()
        } else {
            reader
                .into_samples::<f32>()
                .map(|s| s.expect("failed to read sample"))
                .collect()
        };

        self.transcribe_audio(&samples)
    }
}
