//! TTS Integration Example
//!
//! This example demonstrates how to use TTS with AutoAgents.
//! It shows the basic TTS configuration and usage patterns.
//!
//! Usage:
//!   cargo run -p tts-agent-example

use autoagents_core::agent::AgentConfig;
use autoagents_tts::{
    AudioFormat, AudioStoragePolicy, SpeechRequest, TTSMode, TTSSpeechProvider, VoiceIdentifier,
};
use autoagents_pocket_tts::{PocketTTSConfig, PocketTTSProvider};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    println!("TTS Integration Example");
    println!("=======================");
    println!();

    // Initialize the TTS provider
    println!("Initializing Pocket-TTS provider...");
    println!("Note: This will download ~100MB model from HuggingFace on first run.");
    println!();
    
    let tts_provider = match PocketTTSProvider::new(PocketTTSConfig::default()) {
        Ok(provider) => {
            println!("TTS provider initialized!");
            println!();
            Arc::new(provider)
        }
        Err(e) => {
            eprintln!("Error initializing TTS provider: {}", e);
            eprintln!();
            eprintln!("This example requires:");
            eprintln!("  1. Internet connection to download the model");
            eprintln!("  2. ~500MB free disk space");
            eprintln!("  3. HuggingFace access (models are public but may require auth)");
            eprintln!();
            eprintln!("To run this example successfully:");
            eprintln!("  1. Ensure you have internet access");
            eprintln!("  2. Set HF_TOKEN environment variable if needed:");
            eprintln!("     export HF_TOKEN=your_huggingface_token");
            eprintln!();
            eprintln!("Model will be cached at: ~/.cache/pocket-tts/");
            return Err(e.into());
        }
    };

    // Example TTS configuration that would be used in an agent
    let _config = AgentConfig::default()
        .with_tts_mode(TTSMode::TextAndAudio)
        .with_audio_storage_policy(AudioStoragePolicy::OutputOnly)
        .with_default_voice("alba".to_string());

    println!("Agent TTS Configuration:");
    println!("  TTS Mode: TextAndAudio");
    println!("  Storage Policy: OutputOnly");
    println!("  Default Voice: Alba");
    println!();

    // Generate some example speech
    println!("Generating speech examples...");
    println!();

    // Example 1: Simple text
    println!("1. Generating: 'Welcome to AutoAgents TTS integration'");
    let request = SpeechRequest {
        text: "Welcome to AutoAgents TTS integration".to_string(),
        voice: VoiceIdentifier::Predefined("alba".to_string()),
        format: AudioFormat::Wav,
        sample_rate: Some(24000),
    };
    
    let response = tts_provider.generate_speech(request).await?;
    println!("   Generated {} samples ({} ms)", 
        response.audio.samples.len(), 
        response.duration_ms
    );
    save_audio(&response.audio.samples, response.audio.sample_rate, "output1.wav")?;
    println!("   Saved to: output1.wav");
    println!();

    // Example 2: Different voice
    println!("2. Generating with Marius voice: 'This is a test of the TTS system'");
    let request = SpeechRequest {
        text: "This is a test of the TTS system".to_string(),
        voice: VoiceIdentifier::Predefined("marius".to_string()),
        format: AudioFormat::Wav,
        sample_rate: Some(24000),
    };
    
    let response = tts_provider.generate_speech(request).await?;
    println!("   Generated {} samples ({} ms)", 
        response.audio.samples.len(), 
        response.duration_ms
    );
    save_audio(&response.audio.samples, response.audio.sample_rate, "output2.wav")?;
    println!("   Saved to: output2.wav");
    println!();

    println!("Example completed!");
    println!();
    println!("Integration Notes:");
    println!("- TTS providers can be passed to agents via .tts() builder method");
    println!("- Agents with TTS enabled will automatically generate audio");
    println!("- Use context.speak() within agent logic to generate audio");
    println!("- Audio is stored according to the configured storage policy");

    Ok(())
}

/// Helper function to save audio to a WAV file
fn save_audio(
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
