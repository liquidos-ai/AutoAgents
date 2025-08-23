use crate::{
    audio::AudioPlayback, cli::AudioBufferMessage, kokoros::tts::koko::TTSKoko, stt::STTProcessor,
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
    chunk_duration_seconds: u32,
) -> Result<()> {
    println!("🎬 Starting continuous recording mode...");
    println!("💬 Recording in {}-second chunks", chunk_duration_seconds);

    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device found"))?;

    let config = input_device.default_input_config()?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;
    let chunk_size = (sample_rate * chunk_duration_seconds) as usize;

    println!("Using input device: {}", input_device.name()?);
    println!("Sample rate: {}Hz, Channels: {}", sample_rate, channels);
    println!("Chunk size: {} samples", chunk_size);

    let audio_buffer = Arc::new(Mutex::new(Vec::<f32>::with_capacity(chunk_size)));
    let buffer_clone = audio_buffer.clone();

    let (audio_tx, mut audio_rx) = tokio::sync::mpsc::channel::<AudioBufferMessage>(10);

    // Continuous recording stream
    let stream = input_device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut buffer = buffer_clone.lock().unwrap();

            // Convert multi-channel to mono and add to buffer
            for chunk in data.chunks(channels) {
                if let Some(&sample) = chunk.first() {
                    buffer.push(sample);

                    // When buffer reaches chunk size, send it
                    if buffer.len() >= chunk_size {
                        let audio_data: Vec<f32> = buffer.drain(..).collect();
                        let audio_message = AudioBufferMessage {
                            audio_data,
                            sample_rate,
                        };

                        if let Err(_) = audio_tx.try_send(audio_message) {
                            eprintln!("⚠️ Audio buffer channel full, dropping chunk");
                        }
                    }
                }
            }
        },
        move |err| eprintln!("Audio input error: {}", err),
        None,
    )?;

    stream.play()?;
    println!("🎤 Continuous recording started!");

    // Process incoming audio chunks
    let runtime_clone = runtime.clone();
    let topic_clone = stt_topic.clone();

    tokio::spawn(async move {
        while let Some(mut audio_message) = audio_rx.recv().await {
            // Resample if necessary
            if audio_message.sample_rate != WHISPER_SAMPLE_RATE {
                println!(
                    "🔄 Resampling audio from {}Hz to {}Hz...",
                    audio_message.sample_rate, WHISPER_SAMPLE_RATE
                );

                match resample_audio_buffer(
                    audio_message.audio_data,
                    audio_message.sample_rate,
                    WHISPER_SAMPLE_RATE,
                ) {
                    Ok(resampled) => {
                        println!(
                            "✅ Resampled to {} samples at {}Hz",
                            resampled.len(),
                            WHISPER_SAMPLE_RATE
                        );
                        audio_message.audio_data = resampled;
                        audio_message.sample_rate = WHISPER_SAMPLE_RATE;
                    }
                    Err(e) => {
                        eprintln!("❌ Resampling failed: {}", e);
                        continue;
                    }
                }
            }

            // Normalize audio to [-1, 1] range for Whisper
            let max_val = audio_message
                .audio_data
                .iter()
                .map(|&x| x.abs())
                .fold(0.0f32, f32::max);

            if max_val > 1.0 {
                println!("📊 Normalizing audio (peak: {:.2})", max_val);
                for sample in &mut audio_message.audio_data {
                    *sample /= max_val;
                }
            }

            println!(
                "📡 Publishing to STT: {} samples @ {}Hz ({:.2}s)",
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
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
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

    // Create resampler with appropriate chunk size
    let chunk_size = 1024; // Process in chunks
    let mut resampler = create_resampler(input_rate, output_rate, chunk_size)?;

    let mut output = Vec::new();
    let mut input_buffer = vec![vec![0.0f32; chunk_size]; 1];

    // Process audio in chunks
    for chunk in audio.chunks(chunk_size) {
        // Fill input buffer (pad with zeros if necessary)
        input_buffer[0].clear();
        input_buffer[0].extend_from_slice(chunk);
        while input_buffer[0].len() < chunk_size {
            input_buffer[0].push(0.0);
        }

        // Resample
        let resampled = resampler.process(&input_buffer, None)?;
        output.extend_from_slice(&resampled[0]);
    }

    // Process any remaining samples
    let remaining = audio.len() % chunk_size;
    if remaining > 0 {
        input_buffer[0].clear();
        for i in (audio.len() - remaining)..audio.len() {
            input_buffer[0].push(audio[i]);
        }
        while input_buffer[0].len() < chunk_size {
            input_buffer[0].push(0.0);
        }

        let resampled = resampler.process(&input_buffer, None)?;
        let useful_samples = (remaining as f64 * output_rate as f64 / input_rate as f64) as usize;
        output.extend_from_slice(&resampled[0][..useful_samples.min(resampled[0].len())]);
    }

    Ok(output)
}
