//! Quick streaming test with short text
//!
//! This is a shorter version of the streaming example for faster testing.

#[path = "common/mod.rs"]
mod common;
use common::audio_playback;

use autoagents_speech::{
    AudioFormat, PocketTTSConfig, PocketTTSProvider, SpeechRequest, TTSSpeechProvider,
    VoiceIdentifier,
};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Quick Streaming Test");
    println!("===================\n");

    // Initialize provider
    println!("Initializing provider...");
    let provider = PocketTTSProvider::new(PocketTTSConfig::default())?;
    println!("✓ Provider initialized\n");

    // Check if audio playback is enabled
    audio_playback::print_playback_info();

    // Initialize audio player if playback is enabled
    let audio_player = if audio_playback::is_playback_enabled() {
        audio_playback::AudioPlayer::try_new()
    } else {
        None
    };

    // Short text for quick streaming
    let text = "Hello, this is a streaming test.";
    println!("Generating speech for: \"{}\"\n", text);

    // Create speech request
    let request = SpeechRequest {
        text: text.to_string(),
        voice: VoiceIdentifier::new("alba"),
        format: AudioFormat::Wav,
        sample_rate: Some(24000),
    };

    println!("Streaming audio...\n");

    // Get the audio stream
    let mut stream = provider.generate_speech_stream(request).await?;

    // Collect all audio chunks
    let mut all_samples = Vec::new();
    let mut chunk_count = 0;
    let sample_rate = 24000;

    // Process each chunk as it arrives
    while let Some(result) = stream.next().await {
        match result {
            Ok(chunk) => {
                chunk_count += 1;
                let chunk_samples = chunk.samples.len();

                // Play chunk immediately if audio player is available
                if let Some(ref player) = audio_player {
                    player.play_samples(&chunk.samples, sample_rate);
                    println!(
                        "  🔊 Chunk {}: {} samples (playing...)",
                        chunk_count, chunk_samples
                    );
                } else {
                    println!("  Chunk {}: {} samples", chunk_count, chunk_samples);
                }

                // Also collect for saving to file
                all_samples.extend_from_slice(&chunk.samples);
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                break;
            }
        }
    }

    // Wait for audio playback to complete
    if let Some(ref player) = audio_player {
        if player.is_playing() {
            println!("\n⏳ Waiting for playback to finish...");
            player.wait_until_end();
        }
    }

    println!();
    println!("✓ Streaming complete!");
    println!("  Total chunks: {}", chunk_count);
    println!("  Total samples: {}", all_samples.len());
    println!(
        "  Duration: {:.2} seconds",
        all_samples.len() as f32 / sample_rate as f32
    );
    println!();

    // Save the complete audio
    let output_path = "output_quick_streaming.wav";
    save_audio_to_file(&all_samples, sample_rate, output_path)?;
    println!("✓ Saved to: {}", output_path);

    Ok(())
}

/// Save audio samples to a WAV file
fn save_audio_to_file(
    samples: &[f32],
    sample_rate: u32,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
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
