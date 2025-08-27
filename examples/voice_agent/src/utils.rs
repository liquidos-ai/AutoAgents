use crate::cli::SimpleMessage;
use crate::{
    audio::AudioPlayback, cli::AudioBufferMessage, kokoros::tts::koko::TTSKoko, stt::STTProcessor,
    vad::VADSegmenter,
};
use anyhow::Result;
use autoagents::core::runtime::TypedRuntime;
use autoagents::core::{actor::Topic, runtime::SingleThreadedRuntime};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::sync::{Arc, Mutex};
use tokio;

pub async fn run_file_mode(
    stt_processor: &mut STTProcessor,
    tts: &TTSKoko,
    input_path: &str,
    output_path: Option<&str>,
    language: &str,
    style: &str,
    speed: f32,
    mono: bool,
    initial_silence: Option<usize>,
) -> Result<()> {
    println!("Processing audio file: {}", input_path);

    // Transcribe the audio file
    let transcribed_text = stt_processor.process_file(input_path)?;
    println!("Transcribed text: {}", transcribed_text);

    if transcribed_text.trim().is_empty() {
        println!("No speech detected in the audio file.");
        return Ok(());
    }

    // Generate speech from the transcribed text
    match output_path {
        Some(path) => {
            println!("Generating speech to file: {}", path);
            if let Err(e) = tts.tts(crate::kokoros::tts::koko::TTSOpts {
                txt: &transcribed_text,
                lan: language,
                style_name: style,
                save_path: path,
                mono,
                speed,
                initial_silence,
            }) {
                return Err(anyhow::anyhow!("TTS error: {}", e));
            }
            println!("Speech saved to: {}", path);
        }
        None => {
            println!("Generating and playing speech...");
            let raw_audio = match tts.tts_raw_audio(
                &transcribed_text,
                language,
                style,
                speed,
                initial_silence,
                None,
                None,
                None,
            ) {
                Ok(audio) => audio,
                Err(e) => return Err(anyhow::anyhow!("TTS error: {}", e)),
            };

            let playback = AudioPlayback::new()?;
            playback.play_audio(raw_audio)?;
            println!("Speech playback completed.");
        }
    }

    Ok(())
}

pub async fn run_test_mode(
    tts: &TTSKoko,
    text: &str,
    language: &str,
    style: &str,
    speed: f32,
    mono: bool,
    initial_silence: Option<usize>,
) -> Result<()> {
    println!("🧪 Testing TTS audio generation and playback");
    println!("📝 Text: '{}'", text);
    println!("🌍 Language: {}", language);
    println!("🎭 Style: {}", style);
    println!("⚡ Speed: {}", speed);
    println!("📻 Mono: {}", mono);

    // Test TTS generation
    println!("🎵 Generating TTS audio...");
    let raw_audio = match tts.tts_raw_audio(
        text,
        language,
        style,
        speed,
        initial_silence,
        None,
        None,
        None,
    ) {
        Ok(audio) => audio,
        Err(e) => {
            eprintln!("❌ TTS generation failed: {}", e);
            return Err(anyhow::anyhow!("TTS error: {}", e));
        }
    };

    println!("✨ Generated {} audio samples", raw_audio.len());

    if raw_audio.is_empty() {
        println!("⚠️ No audio data generated!");
        return Ok(());
    }

    // Test audio playback
    println!("🔊 Testing audio playback...");
    let playback = AudioPlayback::new()?;

    if let Err(e) = playback.play_audio(raw_audio) {
        eprintln!("❌ Audio playback failed: {}", e);
        return Err(e);
    }

    println!("✅ Test completed successfully!");
    Ok(())
}

pub async fn run_realtime_mode_actor_based(
    runtime: Arc<SingleThreadedRuntime>,
    stt_topic: Topic<AudioBufferMessage>,
    tts_topic: Topic<SimpleMessage>,
    _chunk_duration_seconds: u32, // Not used in VAD mode
    recording_control: Option<Arc<tokio::sync::RwLock<bool>>>,
) -> Result<()> {
    println!("🎬 Starting VAD-based continuous recording mode...");
    println!("🎯 Using Voice Activity Detection for smart speech segmentation");

    // Publish first welcome message
    runtime
        .publish(
            &tts_topic,
            SimpleMessage {
                content: "I am Bella, Your AI Assistant. How can I help you?".into(),
            },
        )
        .await?;

    // Use provided recording control or create a new one
    let recording_enabled =
        recording_control.unwrap_or_else(|| Arc::new(tokio::sync::RwLock::new(true)));

    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device found"))?;

    let config = input_device.default_input_config()?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    println!("Using input device: {}", input_device.name()?);
    println!("Sample rate: {}Hz, Channels: {}", sample_rate, channels);

    // Create VAD segmenter - we'll use 16kHz for compatibility with Whisper
    let target_sample_rate = 16000_usize;
    let mut vad_segmenter_instance = VADSegmenter::new(
        target_sample_rate,
        500,   // min_speech_duration_ms: 500ms minimum speech (increased)
        20000, // max_duration_ms: 20 seconds max recording (increased)
        0.6,   // speech_threshold: 0.6 probability (higher threshold to start)
        0.25,  // silence_threshold: 0.25 to end speech (lower threshold - more tolerant)
        None,  // Use default chunk size
    )?;

    // Configure silence timeout for automatic completion (increased for natural pauses)
    vad_segmenter_instance.set_silence_timeout(2500); // 2.5 seconds of silence to auto-complete
    let vad_segmenter = Arc::new(Mutex::new(vad_segmenter_instance));

    let vad_chunk_size = vad_segmenter.lock().unwrap().chunk_size_samples();
    println!(
        "VAD chunk size: {} samples @ {}Hz",
        vad_chunk_size, target_sample_rate
    );

    // Calculate input chunk size needed to get exactly vad_chunk_size after resampling
    let input_chunk_size = if sample_rate as usize != target_sample_rate {
        // Calculate input samples needed to produce vad_chunk_size output samples
        (vad_chunk_size * sample_rate as usize) / target_sample_rate
    } else {
        vad_chunk_size
    };

    println!(
        "Input chunk size: {} samples @ {}Hz",
        input_chunk_size, sample_rate
    );

    // Buffer for accumulating audio before VAD processing
    let audio_buffer = Arc::new(Mutex::new(Vec::<f32>::with_capacity(input_chunk_size * 2)));
    let buffer_clone = audio_buffer.clone();
    let recording_enabled_clone = recording_enabled.clone();
    let vad_clone = vad_segmenter.clone();

    let (audio_tx, mut audio_rx) = tokio::sync::mpsc::channel::<AudioBufferMessage>(10);

    // Continuous recording stream with VAD processing
    let stream = input_device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            // Check if recording is enabled (non-blocking check)
            let is_recording = recording_enabled_clone
                .try_read()
                .map_or(true, |guard| *guard);
            if !is_recording {
                return; // Skip processing if recording is disabled
            }

            let mut buffer = buffer_clone.lock().unwrap();

            // Convert multi-channel to mono and accumulate samples
            for chunk in data.chunks(channels) {
                if let Some(&sample) = chunk.first() {
                    buffer.push(sample);
                }
            }

            // Process in input-sized chunks when we have enough data
            while buffer.len() >= input_chunk_size {
                let audio_chunk: Vec<f32> = buffer.drain(..input_chunk_size).collect();

                // Resample to 16kHz if needed
                let final_chunk = if sample_rate as usize != target_sample_rate {
                    match resample_audio_buffer(audio_chunk, sample_rate, target_sample_rate as u32)
                    {
                        Ok(mut resampled) => {
                            // Ensure we have exactly vad_chunk_size samples
                            if resampled.len() != vad_chunk_size {
                                resampled.resize(vad_chunk_size, 0.0);
                            }
                            resampled
                        }
                        Err(e) => {
                            eprintln!("❌ Resampling failed: {}", e);
                            continue;
                        }
                    }
                } else {
                    audio_chunk
                };

                // Process through VAD
                let mut vad = vad_clone.lock().unwrap();
                match vad.process_chunk(&final_chunk) {
                    Ok(Some(speech_segment)) => {
                        // Complete speech segment detected
                        println!(
                            "📦 Speech segment complete: {} samples ({:.2}s)",
                            speech_segment.len(),
                            speech_segment.len() as f32 / target_sample_rate as f32
                        );

                        let audio_message = AudioBufferMessage {
                            audio_data: speech_segment,
                            sample_rate: target_sample_rate as u32,
                        };

                        if let Err(_) = audio_tx.try_send(audio_message) {
                            eprintln!("⚠️ Audio buffer channel full, dropping segment");
                        }
                    }
                    Ok(None) => {
                        // Still collecting or no speech
                    }
                    Err(e) => {
                        eprintln!("❌ VAD error: {}", e);
                    }
                }
            }
        },
        move |err| eprintln!("Audio input error: {}", err),
        None,
    )?;

    stream.play()?;
    println!("🎤 VAD-based recording started!");
    println!("🔊 Speak naturally - I'll detect when you start and stop speaking");
    println!("🤫 1.5 seconds of silence will automatically complete your speech");

    // Process incoming speech segments
    let runtime_clone = runtime.clone();
    let topic_clone = stt_topic.clone();

    tokio::spawn(async move {
        while let Some(audio_message) = audio_rx.recv().await {
            // Audio is already at 16kHz from VAD processing, no resampling needed
            println!(
                "📡 Publishing speech segment to STT: {} samples @ {}Hz ({:.2}s)",
                audio_message.audio_data.len(),
                audio_message.sample_rate,
                audio_message.audio_data.len() as f32 / audio_message.sample_rate as f32
            );

            if let Err(e) = runtime_clone.publish(&topic_clone, audio_message).await {
                eprintln!("❌ Failed to publish audio to STT actor: {}", e);
            }
        }
    });

    // Keep the stream alive
    loop {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
}

// Whisper expects 16kHz audio
pub const WHISPER_SAMPLE_RATE: u32 = 16000;

// Create a resampler for converting audio to Whisper's expected format
pub fn create_resampler(
    input_rate: u32,
    output_rate: u32,
    chunk_size: usize,
) -> Result<SincFixedIn<f32>> {
    let params = SincInterpolationParameters {
        sinc_len: 256,  // Higher quality filtering
        f_cutoff: 0.95, // Anti-aliasing cutoff
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2, // High-quality window
    };

    let resampler = SincFixedIn::<f32>::new(
        output_rate as f64 / input_rate as f64,
        2.0, // max relative ratio change (not used for FixedIn)
        params,
        chunk_size,
        1, // number of channels
    )?;

    Ok(resampler)
}

// Resample audio buffer to target sample rate
pub fn resample_audio_buffer(
    audio: Vec<f32>,
    input_rate: u32,
    output_rate: u32,
) -> Result<Vec<f32>> {
    if input_rate == output_rate {
        return Ok(audio);
    }

    // Process the entire buffer at once for better quality
    let chunk_size = audio.len();
    let mut resampler = create_resampler(input_rate, output_rate, chunk_size)?;

    // Create input buffer with the audio data
    let input_buffer = vec![audio];

    // Process the entire chunk at once
    let output_data = resampler.process(&input_buffer, None)?;

    // Return the resampled audio
    Ok(output_data[0].clone())
}
