//! Real-time audio generation and playback example
//!
//! This example demonstrates true real-time streaming where audio starts
//! playing as soon as the first chunk is generated, minimizing latency.
//!
//! Usage:
//!   cargo run --release --example realtime_playback --features library
//!
//! To disable audio playback:
//!   NO_PLAY=1 cargo run --release --example realtime_playback --features library

#[path = "common/mod.rs"]
mod common;
use common::audio_playback;

use autoagents_pocket_tts::{PocketTTSConfig, PocketTTSProvider};
use autoagents_tts::{AudioFormat, SpeechRequest, TTSSpeechProvider, VoiceIdentifier};
use futures::StreamExt;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Real-Time Audio Generation & Playback");
    println!("=====================================");
    println!();

    // Initialize provider
    println!("⏳ Initializing TTS provider...");
    let start = Instant::now();
    let provider = PocketTTSProvider::new(PocketTTSConfig::default())?;
    println!("✓ Provider ready in {:.2}s", start.elapsed().as_secs_f64());
    println!();

    // Check if audio playback is enabled
    audio_playback::print_playback_info();

    // Initialize audio player if playback is enabled
    let audio_player = if audio_playback::is_playback_enabled() {
        match audio_playback::AudioPlayer::try_new() {
            Some(player) => {
                println!("✓ Audio player initialized");
                Some(player)
            }
            None => {
                println!("⚠️  Audio player unavailable - will save to file only");
                None
            }
        }
    } else {
        println!("🔇 Audio playback disabled");
        None
    };
    println!();

    // Text to generate
    let text = "Testing Metal acceleration with real-time playback. This combines GPU acceleration with live audio output";

    println!("📝 Input text ({} chars):", text.len());
    println!("   \"{}\"", text);
    println!();

    // Create speech request
    let request = SpeechRequest {
        text: text.to_string(),
        voice: VoiceIdentifier::Predefined("alba".to_string()),
        format: AudioFormat::Wav,
        sample_rate: Some(24000),
    };

    println!("🎙️  Starting real-time generation...");
    let gen_start = Instant::now();
    
    // Get the audio stream
    let mut stream = provider.generate_speech_stream(request).await?;

    // Track metrics
    let mut all_samples = Vec::new();
    let mut chunk_count = 0;
    let sample_rate = 24000;
    let mut first_chunk_time: Option<f64> = None;
    let mut total_gen_time = 0.0;

    // Process each chunk as it arrives - THIS IS REAL-TIME!
    while let Some(result) = stream.next().await {
        match result {
            Ok(chunk) => {
                chunk_count += 1;
                let chunk_samples = chunk.samples.len();
                
                // Record time to first chunk (latency)
                if first_chunk_time.is_none() {
                    let latency = gen_start.elapsed().as_secs_f64();
                    first_chunk_time = Some(latency);
                    println!();
                    println!("⚡ First chunk received in {:.2}s (latency)", latency);
                    println!("🔊 Audio playback starting NOW...");
                    println!();
                }
                
                // Play chunk IMMEDIATELY - this is real-time streaming!
                if let Some(ref player) = audio_player {
                    player.play_samples(&chunk.samples, sample_rate);
                }
                
                // Also collect for saving to file
                all_samples.extend_from_slice(&chunk.samples);
                
                // Show progress
                let duration_so_far = all_samples.len() as f32 / sample_rate as f32;
                print!("\r  Chunk {}: {} samples | {:.2}s audio generated", 
                       chunk_count, chunk_samples, duration_so_far);
                std::io::Write::flush(&mut std::io::stdout()).ok();
            }
            Err(e) => {
                eprintln!("\n❌ Error receiving chunk: {}", e);
                break;
            }
        }
    }
    
    total_gen_time = gen_start.elapsed().as_secs_f64();
    println!();
    println!();

    // Wait for audio playback to complete (if playing)
    if let Some(ref player) = audio_player {
        if player.is_playing() {
            println!("⏳ Waiting for audio playback to finish...");
            player.wait_until_end();
            println!("✓ Playback complete");
        }
    }

    // Calculate metrics
    let total_duration = all_samples.len() as f32 / sample_rate as f32;
    let rtf = total_gen_time / total_duration as f64; // Real-Time Factor

    println!();
    println!("📊 Performance Metrics:");
    println!("   ├─ Time to first chunk: {:.2}s (latency)", first_chunk_time.unwrap_or(0.0));
    println!("   ├─ Total generation time: {:.2}s", total_gen_time);
    println!("   ├─ Audio duration: {:.2}s", total_duration);
    println!("   ├─ Real-Time Factor (RTF): {:.2}x", rtf);
    println!("   ├─ Total chunks: {}", chunk_count);
    println!("   └─ Total samples: {}", all_samples.len());
    println!();

    if rtf < 1.0 {
        println!("✅ Real-time capable! (RTF < 1.0 means generation is faster than playback)");
    } else {
        println!("⚠️  Not quite real-time (RTF > 1.0 means generation is slower than playback)");
    }
    println!();

    // Save the complete audio
    let output_path = "output_realtime.wav";
    save_audio_to_file(&all_samples, sample_rate, output_path)?;
    println!("💾 Saved complete audio to: {}", output_path);

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
