//! Basic TTS generation example
//!
//! This example demonstrates how to:
//! - Initialize the Pocket-TTS provider
//! - Generate speech from text using a predefined voice
//! - Play audio in real-time (optional)
//! - Save the audio to a WAV file
//!
//! Usage:
//!   cargo run --example basic_generation
//!
//! To disable audio playback:
//!   NO_PLAY=1 cargo run --example basic_generation

#[path = "common/mod.rs"]
mod common;
use common::audio_playback;

use autoagents_speech::{
    AudioFormat, PocketTTSConfig, PocketTTSProvider, SpeechRequest, TTSSpeechProvider,
    VoiceIdentifier,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("Initializing Pocket-TTS provider...");
    println!("Note: First run requires downloading ~100MB model from HuggingFace.");
    println!("      Set HF_TOKEN environment variable if you get a 401 error.");
    println!();

    // Create provider with default configuration
    // This will download the model from HuggingFace on first run
    let provider = match PocketTTSProvider::new(PocketTTSConfig::default()) {
        Ok(p) => {
            println!("Provider initialized successfully!");
            println!();
            p
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!();
            eprintln!("This example requires:");
            eprintln!("  1. Internet connection");
            eprintln!("  2. HuggingFace token: export HF_TOKEN=your_token");
            eprintln!("  3. Get token from: https://huggingface.co/settings/tokens");
            return Err(e.into());
        }
    };

    // Check if audio playback is enabled
    audio_playback::print_playback_info();

    // Initialize audio player if playback is enabled
    let audio_player = if audio_playback::is_playback_enabled() {
        audio_playback::AudioPlayer::try_new()
    } else {
        None
    };

    // Example 1: Generate speech with Alba voice (female, French)
    println!("Example 1: Generating speech with Alba voice...");
    let request = SpeechRequest {
        text: "Hello! This is a test of the Pocket TTS system.".to_string(),
        voice: VoiceIdentifier::new("alba"),
        format: AudioFormat::Wav,
        sample_rate: Some(24000),
    };

    let response = provider.generate_speech(request).await?;

    println!("  Generated {} samples", response.audio.samples.len());
    println!("  Sample rate: {} Hz", response.audio.sample_rate);
    println!("  Duration: {} ms", response.duration_ms);

    // Play audio if available
    if let Some(ref player) = audio_player {
        println!("  🔊 Playing audio...");
        player.play_samples(&response.audio.samples, response.audio.sample_rate);
        player.wait_until_end();
    }

    // Save to file
    let output_path = "output_alba.wav";
    save_audio_to_file(
        &response.audio.samples,
        response.audio.sample_rate,
        output_path,
    )?;
    println!("  ✓ Saved to: {}", output_path);
    println!();

    // Example 2: Generate speech with Marius voice (male, French)
    println!("Example 2: Generating speech with Marius voice...");
    let request = SpeechRequest {
        text: "The quick brown fox jumps over the lazy dog.".to_string(),
        voice: VoiceIdentifier::new("marius"),
        format: AudioFormat::Wav,
        sample_rate: Some(24000),
    };

    let response = provider.generate_speech(request).await?;

    println!("  Generated {} samples", response.audio.samples.len());
    println!("  Duration: {} ms", response.duration_ms);

    // Play audio if available
    if let Some(ref player) = audio_player {
        println!("  🔊 Playing audio...");
        player.play_samples(&response.audio.samples, response.audio.sample_rate);
        player.wait_until_end();
    }

    let output_path = "output_marius.wav";
    save_audio_to_file(
        &response.audio.samples,
        response.audio.sample_rate,
        output_path,
    )?;
    println!("  ✓ Saved to: {}", output_path);
    println!();

    println!("✓ Done! Check the output WAV files.");
    Ok(())
}

/// Save audio samples to a WAV file
fn save_audio_to_file(
    samples: &[f32],
    sample_rate: u32,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Simple WAV writer (mono, f32 PCM)
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;

    for &sample in samples {
        writer.write_sample(sample)?;
    }

    writer.finalize()?;
    Ok(())
}
