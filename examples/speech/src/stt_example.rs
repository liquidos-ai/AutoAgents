use autoagents_speech::audio_capture::{AudioCapture, AudioCaptureConfig};
use autoagents_speech::{STTSpeechProvider, SharedAudioData, TranscriptionRequest};
use std::path::PathBuf;

use crate::vad_stt::build_parakeet_batch_provider;

#[derive(Debug)]
pub struct SttArgs {
    pub audio_file: PathBuf,
    pub language: Option<String>,
}

pub async fn run(args: SttArgs) -> Result<(), Box<dyn std::error::Error>> {
    let provider = build_parakeet_batch_provider()?;

    let audio =
        AudioCapture::read_audio_with_config(&args.audio_file, AudioCaptureConfig::default())?;
    let request = TranscriptionRequest {
        audio: SharedAudioData::new(audio),
        language: args.language,
        include_timestamps: true,
    };

    let response = provider.transcribe(request).await?;
    println!("Transcription: {}", response.text);
    println!("Duration: {} ms", response.duration_ms);

    if let Some(timestamps) = response.timestamps {
        let preview = timestamps.len().min(10);
        println!("First {preview} word-level timestamps:");
        for token in timestamps.iter().take(preview) {
            println!("  {:.2}s - {:.2}s  {}", token.start, token.end, token.text);
        }
    }

    Ok(())
}
