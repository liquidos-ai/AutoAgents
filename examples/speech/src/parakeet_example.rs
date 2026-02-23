//! Parakeet STT provider examples
//!
//! This example demonstrates all four transcription modes:
//! 1. File transcription (non-streaming) - Using TDT model with timestamps
//! 2. File transcription (streaming) - Using Nemotron model
//! 3. Microphone transcription (non-streaming) - Capture then transcribe with TDT
//! 4. Microphone transcription (streaming) - Real-time with EOU detection
//!
//! Prerequisites:
//! - Download Parakeet models from altunenes/parakeet-rs Huggingface repository using:
//!   - huggingface-cli download altunenes/parakeet-rs --local-dir path_to_parakeet_models
//!
//! - Set environment variables:
//!   export PARAKEET_TDT_PATH=path_to_parakeet_models/tdt
//!   export PARAKEET_NEMOTRON_PATH=path_to_parakeet_models/nemotron-speech-streaming-en-0.6b
//!   export PARAKEET_EOU_PATH=path_to_parakeet_models/realtime_eou_120m-v1-onnx
//!   export AUDIO_FILE_PATH=/path/to/audio.wav (optional, for file examples)
//!
//! Usage:
//!   # File transcription (non-streaming with timestamps)
//!   cargo run -p speech-examples --features parakeet -- --usecase parakeet --mode file
//!
//!   # File transcription (streaming)
//!   cargo run -p speech-examples --features parakeet -- --usecase parakeet --mode file-stream
//!
//!   # Microphone transcription (non-streaming, 5 second capture)
//!   cargo run -p speech-examples --features parakeet -- --usecase parakeet --mode mic
//!
//!   # Microphone transcription (streaming with EOU detection)
//!   cargo run -p speech-examples --features parakeet -- --usecase parakeet --mode mic-stream
//!
//!   # All modes sequentially
//!   cargo run -p speech-examples --features parakeet -- --usecase parakeet --mode all

use autoagents_speech::providers::parakeet::{ModelVariant, Parakeet, ParakeetConfig};
use autoagents_speech::{AudioData, STTSpeechProvider, TranscriptionRequest};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use futures::StreamExt;
use std::env;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::util::{load_audio_from_wav, capture_audio_from_mic};

const PARAKEET_PATH_ENV: &str = "PARAKEET_PATH";
const DEFAULT_TDT_SUBDIR: &str = "tdt";
const DEFAULT_NEMOTRON_SUBDIR: &str = "nemotron-speech-streaming-en-0.6b";
const DEFAULT_EOU_SUBDIR: &str = "realtime_eou_120m-v1-onnx";

fn resolve_model_path(env_var: &str, subdir: &str) -> Result<String, Box<dyn std::error::Error>> {
    if let Ok(path) = env::var(env_var) {
        return Ok(path);
    }

    if let Ok(base) = env::var(PARAKEET_PATH_ENV) {
        let joined = Path::new(&base).join(subdir);
        return Ok(joined.to_string_lossy().into_owned());
    }

    Err(format!(
        "Model path for {} not provided; set {} or combine {} with the {} subdirectory",
        env_var, env_var, PARAKEET_PATH_ENV, subdir
    )
    .into())
}

fn require_audio_file_path() -> Result<String, Box<dyn std::error::Error>> {
    env::var("AUDIO_FILE_PATH").map_err(|_| {
        "AUDIO_FILE_PATH environment variable is required for the file transcription examples"
            .into()
    })
}

#[derive(Debug, Clone)]
pub enum TranscriptionMode {
    FileNonStreaming,
    FileStreaming,
    MicNonStreaming,
    MicStreaming,
    All,
}

impl std::str::FromStr for TranscriptionMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "file" => Ok(TranscriptionMode::FileNonStreaming),
            "file-stream" => Ok(TranscriptionMode::FileStreaming),
            "mic" => Ok(TranscriptionMode::MicNonStreaming),
            "mic-stream" => Ok(TranscriptionMode::MicStreaming),
            "all" => Ok(TranscriptionMode::All),
            _ => Err(format!(
                "Invalid mode: {}. Use: file, file-stream, mic, mic-stream, or all",
                s
            )),
        }
    }
}

pub async fn run(mode: TranscriptionMode) -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║         Parakeet STT Provider - All Examples             ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    match mode {
        TranscriptionMode::FileNonStreaming => {
            run_file_non_streaming().await?;
        }
        TranscriptionMode::FileStreaming => {
            run_file_streaming().await?;
        }
        TranscriptionMode::MicNonStreaming => {
            run_mic_non_streaming().await?;
        }
        TranscriptionMode::MicStreaming => {
            run_mic_streaming().await?;
        }
        TranscriptionMode::All => {
            run_file_non_streaming().await?;
            println!();
            run_file_streaming().await?;
            println!();
            run_mic_non_streaming().await?;
            println!();
            run_mic_streaming().await?;
        }
    }

    Ok(())
}

/// Example 1: File transcription with TDT model (non-streaming with timestamps)
async fn run_file_non_streaming() -> Result<(), Box<dyn std::error::Error>> {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example 1: File Transcription (Non-Streaming with Timestamps)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let model_path = resolve_model_path("PARAKEET_TDT_PATH", DEFAULT_TDT_SUBDIR)?;

    println!("Initializing TDT model from: {}", model_path);
    println!("Note: TDT supports 25 languages and word-level timestamps");
    println!();

    let config = ParakeetConfig::new(ModelVariant::TDT, model_path)
        .with_execution_provider("cpu".to_string());

    let provider = match Parakeet::new(config) {
        Ok(p) => {
            println!("✓ Provider initialized successfully!");
            p
        }
        Err(e) => {
            eprintln!("✗ Error initializing provider: {}", e);
            eprintln!();
            eprintln!("Make sure:");
            eprintln!("  1. Model path is correct (PARAKEET_TDT_PATH env var)");
            eprintln!("  2. Model files exist in the specified directory");
            return Err(e.into());
        }
    };

    // Load audio file
    let audio_file = require_audio_file_path()?;

    println!("Loading audio from: {}", audio_file);
    let audio = match load_audio_from_wav(&audio_file) {
        Ok(a) => {
            println!(
                "✓ Loaded {} samples at {} Hz, {} channel(s)",
                a.samples.len(),
                a.sample_rate,
                a.channels
            );
            a
        }
        Err(e) => {
            eprintln!("✗ Error loading audio: {}", e);
            eprintln!();
            eprintln!("Set AUDIO_FILE_PATH environment variable to a valid WAV file");
            return Err(e.into());
        }
    };

    // Transcribe with timestamps
    println!();
    println!("Transcribing (this may take a moment)...");

    let request = TranscriptionRequest {
        audio,
        language: Some("en".to_string()),
        include_timestamps: true,
    };

    let response = provider.transcribe(request).await?;

    println!();
    println!("╭─────────────────────────────────────────────────────────╮");
    println!("│ Transcription Result                                    │");
    println!("╰─────────────────────────────────────────────────────────╯");
    println!();
    println!("Text: {}", response.text);
    println!();
    println!("Duration: {} ms", response.duration_ms);

    if let Some(timestamps) = response.timestamps {
        println!();
        println!("Word-level Timestamps:");
        println!("┌────────────┬────────────┬──────────────────────────┐");
        println!("│   Start    │    End     │          Word            │");
        println!("├────────────┼────────────┼──────────────────────────┤");
        for token in timestamps.iter().take(20) {
            // Limit to first 20
            println!(
                "│ {:>8.2}s │ {:>8.2}s │ {:<24} │",
                token.start, token.end, token.text
            );
        }
        if timestamps.len() > 20 {
            println!("│    ...     │    ...     │          ...             │");
        }
        println!("└────────────┴────────────┴──────────────────────────┘");
    }

    Ok(())
}

/// Example 2: File transcription with Nemotron model (streaming)
async fn run_file_streaming() -> Result<(), Box<dyn std::error::Error>> {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example 2: File Transcription (Streaming)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let model_path = resolve_model_path("PARAKEET_NEMOTRON_PATH", DEFAULT_NEMOTRON_SUBDIR)?;

    println!("Initializing Nemotron model from: {}", model_path);
    println!("Note: Nemotron supports streaming ASR with punctuation (English only)");
    println!();

    let config = ParakeetConfig::new(ModelVariant::Nemotron, model_path)
        .with_execution_provider("cpu".to_string())
        .with_language("en".to_string());

    let provider = match Parakeet::new(config) {
        Ok(p) => {
            println!("✓ Provider initialized successfully!");
            p
        }
        Err(e) => {
            eprintln!("✗ Error initializing provider: {}", e);
            eprintln!();
            eprintln!("Make sure:");
            eprintln!("  1. Model path is correct (PARAKEET_NEMOTRON_PATH env var)");
            eprintln!("  2. Model files exist in the specified directory");
            return Err(e.into());
        }
    };

    // Load audio file
    let audio_file = require_audio_file_path()?;

    println!("Loading audio from: {}", audio_file);
    let audio = match load_audio_from_wav(&audio_file) {
        Ok(a) => {
            println!(
                "✓ Loaded {} samples at {} Hz, {} channel(s)",
                a.samples.len(),
                a.sample_rate,
                a.channels
            );
            a
        }
        Err(e) => {
            eprintln!("✗ Error loading audio: {}", e);
            eprintln!();
            eprintln!("Set AUDIO_FILE_PATH environment variable to a valid WAV file");
            return Err(e.into());
        }
    };

    println!();
    println!("Streaming transcription (chunks of 560ms)...");
    println!();
    println!("╭─────────────────────────────────────────────────────────╮");
    println!("│ Streaming Output:                                       │");
    println!("╰─────────────────────────────────────────────────────────╯");
    println!();

    let request = TranscriptionRequest {
        audio,
        language: Some("en".to_string()),
        include_timestamps: false,
    };

    let mut stream = provider.transcribe_stream(request).await?;
    let mut full_text = String::new();

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                print!("{}", chunk.text);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
                full_text.push_str(&chunk.text);
            }
            Err(e) => {
                eprintln!("\n✗ Error processing chunk: {}", e);
            }
        }
    }

    println!("\n");
    println!("╭─────────────────────────────────────────────────────────╮");
    println!("│ Complete Transcription:                                 │");
    println!("╰─────────────────────────────────────────────────────────╯");
    println!("{}", full_text);

    Ok(())
}

/// Example 3: Microphone capture and transcription (non-streaming)
async fn run_mic_non_streaming() -> Result<(), Box<dyn std::error::Error>> {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example 3: Microphone Transcription (Non-Streaming)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Check if microphone is available
    let host = cpal::default_host();
    if host.default_input_device().is_none() {
        println!("⚠ No input device (microphone) available. Skipping microphone example.");
        println!("This is normal in CI/automated environments.");
        println!();
        return Ok(());
    }

    let model_path = resolve_model_path("PARAKEET_TDT_PATH", DEFAULT_TDT_SUBDIR)?;

    println!("Initializing TDT model from: {}", model_path);

    let config = ParakeetConfig::new(ModelVariant::TDT, model_path)
        .with_execution_provider("cpu".to_string())
        .with_language("en".to_string());

    let provider = match Parakeet::new(config) {
        Ok(p) => {
            println!("✓ Provider initialized successfully!");
            p
        }
        Err(e) => {
            eprintln!("✗ Error initializing provider: {}", e);
            return Err(e.into());
        }
    };

    println!();
    println!("Capturing audio from microphone for 5 seconds...");
    println!("Speak now!");
    println!();

    let audio = capture_audio_from_mic(5)?;

    println!("✓ Captured {} samples", audio.samples.len());
    println!();
    println!("Transcribing...");

    let request = TranscriptionRequest {
        audio,
        language: Some("en".to_string()),
        include_timestamps: false,
    };

    let response = provider.transcribe(request).await?;

    println!();
    println!("╭─────────────────────────────────────────────────────────╮");
    println!("│ Transcription Result:                                   │");
    println!("╰─────────────────────────────────────────────────────────╯");
    println!("{}", response.text);
    println!();
    println!("Duration: {} ms", response.duration_ms);

    Ok(())
}

/// Example 4: Real-time microphone transcription with EOU detection
async fn run_mic_streaming() -> Result<(), Box<dyn std::error::Error>> {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example 4: Microphone Transcription (Streaming with EOU)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Check if microphone is available
    let host = cpal::default_host();
    if host.default_input_device().is_none() {
        println!("⚠ No input device (microphone) available. Skipping microphone example.");
        println!("This is normal in CI/automated environments.");
        println!();
        return Ok(());
    }

    let model_path = resolve_model_path("PARAKEET_EOU_PATH", DEFAULT_EOU_SUBDIR)?;
    
    // Check for verbose mode
    let verbose = env::var("VERBOSE").is_ok() || env::var("DEBUG").is_ok();

    println!("Initializing EOU model from: {}", model_path);
    println!("Note: EOU model detects end-of-utterance automatically");
    if verbose {
        println!("Verbose mode: ENABLED");
    }
    println!();

    let config = ParakeetConfig::new(ModelVariant::EOU, model_path)
        .with_execution_provider("cpu".to_string())
        .with_language("en".to_string());

    let provider = match Parakeet::new(config) {
        Ok(p) => {
            println!("✓ Provider initialized successfully!");
            p
        }
        Err(e) => {
            eprintln!("✗ Error initializing provider: {}", e);
            return Err(e.into());
        }
    };

    println!();
    println!("Starting real-time transcription...");
    println!("Speak into your microphone. End-of-utterance will be detected automatically.");
    println!("Press Ctrl+C to stop.");
    println!();
    println!("╭─────────────────────────────────────────────────────────╮");
    println!("│ Real-time Transcription:                                │");
    println!("╰─────────────────────────────────────────────────────────╯");
    println!();
    println!("Listening... (the model needs ~1 second of audio before it starts transcribing)");
    println!();

    // Reset provider state
    provider.reset().await;

    // Setup microphone with continuous buffering
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("No input device available")?;

    let default_config = device.default_input_config()?;
    let sample_rate = default_config.sample_rate().0;
    let channels = default_config.channels();

    if verbose {
        println!("Microphone: {} Hz, {} channels", sample_rate, channels);
        println!("Sample format: {:?}", default_config.sample_format());
        println!();
    }

    // Buffer for accumulating audio samples continuously
    let samples = Arc::new(Mutex::new(Vec::new()));
    let samples_clone = samples.clone();

    // Build audio stream that continuously captures in background
    let stream = match default_config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &default_config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                samples_clone.lock().unwrap().extend_from_slice(data);
            },
            |err| eprintln!("Audio error: {}", err),
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &default_config.into(),
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                let float_samples: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                samples_clone.lock().unwrap().extend(float_samples);
            },
            |err| eprintln!("Audio error: {}", err),
            None,
        )?,
        _ => return Err("Unsupported sample format".into()),
    };

    stream.play()?;

    let chunk_duration = Duration::from_millis(160); // 160ms chunks
    let mut last_process_time = tokio::time::Instant::now();
    let mut chunks_processed = 0;
    let mut current_utterance = String::new();
    let target_samples_16k = 2560; // 160ms at 16kHz

    // Calculate how many samples we need at the mic's native sample rate
    let target_samples_native = if sample_rate == 16000 {
        target_samples_16k
    } else {
        (target_samples_16k as f32 * sample_rate as f32 / 16000.0) as usize
    };
    let target_samples_raw = target_samples_native * channels as usize;

    for i in 0..100 {
        // Run for ~16 seconds (100 chunks × 160ms)
        
        // Wait for next chunk interval (maintain precise timing)
        let elapsed_since_last = last_process_time.elapsed();
        if elapsed_since_last < chunk_duration {
            tokio::time::sleep(chunk_duration - elapsed_since_last).await;
        }
        last_process_time = tokio::time::Instant::now();

        // Get samples from continuously-filling buffer
        let mut audio_buffer = samples.lock().unwrap();

        if audio_buffer.len() < target_samples_raw {
            // Not enough data yet, wait (this should be rare after startup)
            drop(audio_buffer);
            if verbose {
                eprintln!("[Chunk {}] Waiting for audio buffer to fill...", chunks_processed);
            }
            continue;
        }

        // Extract exactly what we need for this chunk
        let mut chunk: Vec<f32> = audio_buffer.drain(..target_samples_raw).collect();
        drop(audio_buffer);

        // Convert to mono if needed
        if channels > 1 {
            chunk = chunk
                .chunks(channels as usize)
                .map(|c| c.iter().sum::<f32>() / channels as f32)
                .collect();
        }

        // Resample to 16kHz if needed
        if sample_rate != 16000 {
            let ratio = sample_rate as f32 / 16000.0;
            let target_len = (chunk.len() as f32 / ratio) as usize;
            let mut resampled = Vec::with_capacity(target_len);

            for i in 0..target_len {
                let src_pos = i as f32 * ratio;
                let src_idx = src_pos as usize;

                if src_idx + 1 < chunk.len() {
                    let frac = src_pos - src_idx as f32;
                    let interpolated = chunk[src_idx] * (1.0 - frac) + chunk[src_idx + 1] * frac;
                    resampled.push(interpolated);
                } else if src_idx < chunk.len() {
                    resampled.push(chunk[src_idx]);
                }
            }
            chunk = resampled;
        }

        // Ensure exactly 2560 samples
        if chunk.len() < target_samples_16k {
            chunk.resize(target_samples_16k, 0.0);
        } else if chunk.len() > target_samples_16k {
            chunk.truncate(target_samples_16k);
        }

        // Show processing indicator every 5 chunks (800ms)
        if !verbose && chunks_processed % 5 == 0 {
            eprint!(".");
            std::io::Write::flush(&mut std::io::stderr()).unwrap();
        }
        chunks_processed += 1;

        // Verbose mode: show detailed chunk info including audio level
        if verbose {
            // Calculate RMS (root mean square) for audio level indication
            let rms: f32 = (chunk.iter().map(|&s| s * s).sum::<f32>() 
                / chunk.len() as f32).sqrt();
            let db = if rms > 0.0 {
                20.0 * rms.log10()
            } else {
                -100.0
            };
            
            eprintln!(
                "[Chunk {}] Processing {} samples, Audio Level: {:.1} dB (RMS: {:.4})",
                chunks_processed,
                chunk.len(),
                db,
                rms
            );
        }

        // Create AudioData for provider
        let audio = AudioData {
            samples: chunk,
            sample_rate: 16000,
            channels: 1,
        };

        // Process chunk
        match provider.process_chunk(audio.samples.clone()).await {
            Ok(chunk_result) => {
                if verbose {
                    eprintln!(
                        "[Chunk {}] Result: text='{}', is_final={}",
                        chunks_processed,
                        chunk_result.text,
                        chunk_result.is_final
                    );
                }

                if !chunk_result.text.is_empty() {
                    // Clear the dots line when we get text
                    if current_utterance.is_empty() && !verbose {
                        eprintln!(); // New line after dots
                    }
                    
                    current_utterance.push_str(&chunk_result.text);
                    print!("{}", chunk_result.text);
                    std::io::Write::flush(&mut std::io::stdout()).unwrap();
                }

                if chunk_result.is_final {
                    if !current_utterance.is_empty() {
                        println!(" [EOU]");
                        println!();
                        println!("Transcript: {}", current_utterance);
                        println!();
                        current_utterance.clear();
                    } else if verbose {
                        eprintln!("[EOU detected but no text accumulated]");
                    }
                    provider.reset().await;
                    chunks_processed = 0; // Reset counter for new utterance
                }
            }
            Err(e) => {
                eprintln!();
                eprintln!("Error processing chunk: {}", e);
            }
        }

        // Stop after reasonable time for demo
        if i >= 99 {
            println!();
            println!("Demo complete!");
            break;
        }
    }

    drop(stream);

    Ok(())
}