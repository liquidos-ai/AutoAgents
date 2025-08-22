use crate::agent::{AgentOutput, VoiceAgent};
use crate::kokoros::actor::{TTSActor, TTSActorArgs, TTSConfig};
use crate::kokoros::tts::koko::TTSKoko;
use crate::stt::actor::{STTActor, STTActorArgs};
use crate::stt::model::WhichModel;
use crate::stt::{AudioPlayback, STTProcessor, SilenceDetector, VadResult, VoiceActivityDetector};
use anyhow::Result;
use autoagents::core::actor::{ActorMessage, CloneableMessage, Topic};
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::ReActAgentOutput;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::AgentBuilder;
use autoagents::core::environment::Environment;
use autoagents::core::protocol::{Event, TaskResult};
use autoagents::core::ractor::Actor;
use autoagents::core::runtime::SingleThreadedRuntime;
use autoagents::core::runtime::TypedRuntime;
use autoagents::llm::backends::openai::OpenAI;
use autoagents::llm::builder::LLMBuilder;
use clap::{Parser, Subcommand};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{mpsc, Arc, Mutex};
use tokio_stream::{wrappers::ReceiverStream, StreamExt as TokioStreamExt};

#[derive(Subcommand, Debug, Clone)]
pub enum Mode {
    /// Take an input audio file and convert to text, then generate speech
    File {
        /// Path to input audio file (WAV format)
        #[arg(short, long)]
        input: String,

        /// Path to output audio file (optional, plays directly if not specified)
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Use microphone for real-time audio input and playback with streaming
    Realtime {
        /// Silence threshold (0.0 - 1.0)
        #[arg(long, default_value = "0.02")]
        silence_threshold: f32,

        /// Minimum silence duration in milliseconds to trigger processing
        #[arg(long, default_value = "2000")]
        silence_duration: u32,

        /// Recording duration in seconds (0 for unlimited)
        #[arg(long, default_value = "10")]
        max_recording_duration: u32,
    },
    /// Test TTS audio generation and playback
    Test {
        /// Text to synthesize and play
        #[arg(short, long, default_value = "Hello, Welcome to AutoAgents.")]
        text: String,
    },
}

#[derive(Parser, Debug)]
#[command(name = "voice-agent")]
#[command(version = "0.1")]
#[command(author = "AutoAgents Team")]
pub struct Cli {
    /// A language identifier from https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md
    #[arg(
        short = 'l',
        long = "lan",
        value_name = "LANGUAGE",
        default_value = "en-us"
    )]
    pub lan: String,

    /// Path to the Kokoro v1.0 ONNX model on the filesystem
    #[arg(
        short = 'm',
        long = "model",
        value_name = "MODEL_PATH",
        default_value = "checkpoints/kokoro-v1.0.onnx"
    )]
    pub model_path: String,

    /// Path to the voices data file on the filesystem
    #[arg(
        short = 'd',
        long = "data",
        value_name = "DATA_PATH",
        default_value = "examples/voice_agent/audio/voices-v1.0.bin"
    )]
    pub data_path: String,

    /// Which single voice to use or voices to combine to serve as the style of speech
    #[arg(
        short = 's',
        long = "style",
        value_name = "STYLE",
        default_value = "af_sarah.4+af_nicole.6"
    )]
    pub style: String,

    /// Rate of speech, as a coefficient of the default
    #[arg(
        short = 'p',
        long = "speed",
        value_name = "SPEED",
        default_value_t = 1.0
    )]
    pub speed: f32,

    /// Output audio in mono (as opposed to stereo)
    #[arg(long = "mono", default_value_t = false)]
    pub mono: bool,

    /// Initial silence duration in tokens
    #[arg(long = "initial-silence", value_name = "INITIAL_SILENCE")]
    pub initial_silence: Option<usize>,

    /// STT model to use
    #[arg(long = "stt-model", default_value = "tiny.en")]
    pub stt_model: WhichModel,

    /// Language for STT (optional, auto-detect if not specified)
    #[arg(long = "stt-language")]
    pub stt_language: Option<String>,

    #[command(subcommand)]
    pub mode: Mode,
}

impl Cli {
    pub fn new() -> Self {
        Cli::parse()
    }
}

/// Simple message type for basic actor communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleMessage {
    pub content: String,
}

impl CloneableMessage for SimpleMessage {}

impl ActorMessage for SimpleMessage {}

/// Message type for audio buffer communication between actors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioBufferMessage {
    pub audio_data: Vec<f32>,
    pub sample_rate: u32,
}

impl CloneableMessage for AudioBufferMessage {}

impl ActorMessage for AudioBufferMessage {}

/// Message type for TTS audio output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTSAudioMessage {
    pub audio_data: Vec<f32>,
    pub text: String,
}

impl CloneableMessage for TTSAudioMessage {}

impl ActorMessage for TTSAudioMessage {}

pub async fn run(cli: Cli) -> Result<()> {
    println!("🚀 Initializing Voice Agent...");
    println!("📁 TTS Model path: {}", cli.model_path);
    println!("📁 TTS Data path: {}", cli.data_path);
    println!("🤖 STT Model: {:?}", cli.stt_model);

    // Check if TTS files exist
    if !std::path::Path::new(&cli.model_path).exists() {
        println!(
            "📥 TTS model file not found, will download: {}",
            cli.model_path
        );
    }
    if !std::path::Path::new(&cli.data_path).exists() {
        println!(
            "📥 TTS data file not found, will download: {}",
            cli.data_path
        );
    }

    println!("🎤 Initializing TTS...");
    let tts = TTSKoko::new(&cli.model_path, &cli.data_path).await;
    println!("✅ TTS initialized successfully");

    println!("🎧 Initializing STT...");
    let mut stt_processor = STTProcessor::new(cli.stt_model, cli.stt_language.clone()).await?;
    println!("✅ STT initialized successfully");

    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or("".into());

    // Initialize and configure the LLM client
    let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(api_key) // Set the API key
        .model("gpt-4o") // Use GPT-4o-mini model
        .max_tokens(512) // Limit response length
        .temperature(0.2) // Control response randomness (0.0-1.0)
        .stream(false) // Disable streaming responses
        .build()
        .expect("Failed to build LLM");

    // Create runtime
    let runtime = SingleThreadedRuntime::new(Some(10));

    // Create environment
    let mut environment = Environment::new(None);
    environment.register_runtime(runtime.clone()).await?;

    // Create topics for pub/sub messaging
    let stt_topic = Topic::<AudioBufferMessage>::new("stt_topic");
    let tts_topic = Topic::<SimpleMessage>::new("tts_topic");
    let agent_topic = Topic::<Task>::new("agent_topic");

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(100));
    let agent = VoiceAgent {};
    let _ = AgentBuilder::new(agent)
        .with_llm(llm)
        .runtime(runtime.clone())
        .subscribe_topic(agent_topic.clone())
        .with_memory(sliding_window_memory)
        .build()
        .await?;

    // Set up event handling
    let receiver = environment.take_event_receiver(None).await?;

    // Create actors with proper configuration
    let tts_config = TTSConfig {
        language: cli.lan.clone(),
        style: cli.style.clone(),
        speed: cli.speed,
        mono: cli.mono,
        initial_silence: cli.initial_silence,
    };

    let tts_actor = TTSActor::new("TTS_Actor", runtime.clone(), tts_config);
    let stt_actor = STTActor::new("STT_Actor", runtime.clone());

    // Initialize the actors with their models
    tts_actor
        .initialize_tts(&cli.model_path, &cli.data_path)
        .await?;
    stt_actor
        .initialize_stt(cli.stt_model, cli.stt_language.clone())
        .await?;

    let (tts_actor_ref, _) = Actor::spawn(None, tts_actor, TTSActorArgs {}).await?;
    let (stt_actor_ref, _) = Actor::spawn(None, stt_actor, STTActorArgs {}).await?;

    // Subscribe actors to topics
    runtime.subscribe(&tts_topic, tts_actor_ref.clone()).await?;
    runtime.subscribe(&stt_topic, stt_actor_ref.clone()).await?;

    //Run indefinitely listening to the events
    handle_streaming_events(receiver, runtime.clone(), tts_topic.clone());
    let _ = environment.run();

    match cli.mode {
        Mode::File { input, output } => {
            run_file_mode(
                &mut stt_processor,
                &tts,
                &input,
                output.as_deref(),
                &cli.lan,
                &cli.style,
                cli.speed,
                cli.mono,
                cli.initial_silence,
            )
            .await
        }
        Mode::Realtime {
            silence_threshold,
            silence_duration,
            max_recording_duration,
        } => {
            run_realtime_mode_actor_based(
                runtime.clone(),
                stt_topic.clone(),
                silence_threshold,
                silence_duration,
                max_recording_duration,
            )
            .await
        }
        Mode::Test { text } => {
            run_test_mode(
                &tts,
                &text,
                &cli.lan,
                &cli.style,
                cli.speed,
                cli.mono,
                cli.initial_silence,
            )
            .await
        }
    }
}

fn handle_streaming_events(
    mut event_stream: ReceiverStream<Event>,
    runtime: Arc<SingleThreadedRuntime>,
    topic: Topic<SimpleMessage>,
) {
    tokio::spawn(async move {
        while let Some(event) = TokioStreamExt::next(&mut event_stream).await {
            match event {
                Event::TaskComplete { result, .. } => {
                    match result {
                        TaskResult::Value(val) => {
                            match serde_json::from_value::<ReActAgentOutput>(val) {
                                Ok(agent_out) => {
                                    // Try to parse as streaming output
                                    if let Ok(streaming_output) =
                                        serde_json::from_str::<AgentOutput>(&agent_out.response)
                                    {
                                        println!(
                                            "{}",
                                            format!(
                                                "🌊 Streaming Response ({:?})",
                                                streaming_output.response
                                            )
                                        );
                                        let _ = runtime
                                            .publish(
                                                &topic,
                                                SimpleMessage {
                                                    content: streaming_output.response,
                                                },
                                            )
                                            .await;
                                    }
                                }
                                Err(e) => {
                                    println!("{}", format!("❌ Failed to parse response: {}", e));
                                }
                            }
                        }
                        TaskResult::Failure(error) => {
                            println!("{}", format!("❌ Task failed: {}", error));
                        }
                        TaskResult::Aborted => todo!(),
                    }
                }
                Event::StreamChunk { sub_id, chunk } => {
                    println!("{}", format!("📦 Stream chunk ({}): {}", sub_id, chunk));
                }
                _ => {
                    // Handle other streaming-specific events if they exist
                }
            }
        }
    });
}

async fn run_file_mode(
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

async fn run_realtime_mode(
    mut stt_processor: STTProcessor,
    tts: TTSKoko,
    language: &str,
    style: &str,
    speed: f32,
    _mono: bool,
    initial_silence: Option<usize>,
    silence_threshold: f32,
    silence_duration_ms: u32,
    max_recording_duration: u32,
) -> Result<()> {
    println!("Starting real-time mode...");
    println!(
        "Speak into your microphone. The system will automatically detect when you stop speaking."
    );
    println!("Press Ctrl+C to exit.");

    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device found"))?;

    let config = input_device.default_input_config()?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    println!("Using input device: {}", input_device.name()?);
    println!("Sample rate: {}Hz, Channels: {}", sample_rate, channels);

    let mut vad = VoiceActivityDetector::new(sample_rate, 250, silence_duration_ms); // 250ms min voice (more responsive)
    println!(
        "🎯 VAD initialized: min_voice=250ms, min_silence={}ms",
        silence_duration_ms
    );

    let (audio_tx, audio_rx) = mpsc::channel::<Vec<f32>>();
    let (_control_tx, control_rx) = mpsc::channel::<bool>();

    let audio_buffer = Arc::new(Mutex::new(VecDeque::<f32>::new()));
    let buffer_clone = audio_buffer.clone();
    let is_recording = Arc::new(Mutex::new(false));
    let recording_clone = is_recording.clone();
    let audio_tx_clone = audio_tx.clone();

    // Create a separate VAD for the audio callback
    let vad_for_callback = Arc::new(Mutex::new(VoiceActivityDetector::new(
        sample_rate,
        250, // Match the reduced minimum voice duration
        silence_duration_ms,
    )));
    let vad_clone = vad_for_callback.clone();

    // Audio input stream with advanced VAD
    let stream = input_device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut buffer = buffer_clone.lock().unwrap();
            let mut recording = recording_clone.lock().unwrap();
            let mut vad_callback = vad_clone.lock().unwrap();

            // Convert multi-channel to mono
            let mono_audio: Vec<f32> = data
                .chunks(channels)
                .map(|chunk| chunk.first().copied().unwrap_or(0.0))
                .collect();

            // Run VAD analysis
            let vad_result = vad_callback.process_audio(&mono_audio);
            let stats = vad_callback.get_stats();

            match vad_result {
                VadResult::Voice => {
                    if !*recording {
                        println!(
                            "🎤 Voice detected! (noise_level: {:.6}, threshold: {:.6})",
                            stats.background_noise_level, stats.current_threshold
                        );
                        *recording = true;
                        buffer.clear(); // Start fresh
                    }
                    // Add audio to buffer
                    for &sample in &mono_audio {
                        buffer.push_back(sample);
                    }
                }
                VadResult::Silence => {
                    if *recording {
                        println!("🔇 Silence detected after voice, processing speech...");
                        *recording = false;
                        let audio_data: Vec<f32> = buffer.drain(..).collect();
                        if !audio_data.is_empty() {
                            let _ = audio_tx_clone.send(audio_data);
                        }
                    }
                }
                VadResult::Transition => {
                    if *recording {
                        // Continue recording during transition
                        for &sample in &mono_audio {
                            buffer.push_back(sample);
                        }
                    }
                }
            }

            // Safety: Limit recording duration
            if *recording && buffer.len() > (max_recording_duration * sample_rate) as usize {
                println!("⏰ Maximum recording duration reached, processing...");
                *recording = false;
                let audio_data: Vec<f32> = buffer.drain(..).collect();
                let _ = audio_tx_clone.send(audio_data);
            }
        },
        move |err| eprintln!("Audio input error: {}", err),
        None,
    )?;

    stream.play()?;

    // Main processing loop
    loop {
        // Check for stop signal
        if control_rx.try_recv().is_ok() {
            break;
        }

        // VAD processing now happens in the audio callback
        // This loop just handles the processed audio

        // Process audio if available
        if let Ok(audio_data) = audio_rx.try_recv() {
            println!("📝 Transcribing {} samples...", audio_data.len());

            // Pause the input stream during processing
            stream
                .pause()
                .unwrap_or_else(|e| eprintln!("Warning: couldn't pause stream: {}", e));

            match stt_processor.transcribe_audio(&audio_data) {
                Ok(text) => {
                    println!("🎯 STT Raw result: '{}'", text);
                    let cleaned_text = text.trim();
                    println!("🧹 Cleaned text: '{}'", cleaned_text);
                    if !cleaned_text.is_empty() {
                        println!("🗣️  You said: '{}'", cleaned_text);

                        // Generate and play response
                        println!("🎵 Generating TTS audio for: '{}'", cleaned_text);
                        match tts.tts_raw_audio(
                            cleaned_text,
                            language,
                            style,
                            speed,
                            initial_silence,
                            None,
                            None,
                            None,
                        ) {
                            Ok(raw_audio) => {
                                println!("✨ TTS generated {} audio samples", raw_audio.len());
                                if !raw_audio.is_empty() {
                                    // Create a new playback instance for each audio
                                    match AudioPlayback::new() {
                                        Ok(new_playback) => {
                                            if let Err(e) = new_playback.play_audio(raw_audio) {
                                                eprintln!("❌ Error playing audio: {}", e);
                                            } else {
                                                println!("✅ Audio playback completed");
                                            }
                                        }
                                        Err(e) => {
                                            eprintln!("❌ Error creating playback device: {}", e)
                                        }
                                    }
                                } else {
                                    println!("⚠️ TTS generated empty audio");
                                }
                            }
                            Err(e) => eprintln!("❌ Error generating speech: {}", e),
                        }
                    } else {
                        println!("🔇 No speech detected in the recording");
                    }
                }
                Err(e) => eprintln!("❌ STT Error: {}", e),
            }

            // Resume the input stream
            std::thread::sleep(std::time::Duration::from_millis(500)); // Brief pause
            stream
                .play()
                .unwrap_or_else(|e| eprintln!("Warning: couldn't resume stream: {}", e));
            println!("🎤 Ready for next input...");

            vad.reset();
        }

        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    Ok(())
}

async fn run_test_mode(
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

async fn run_realtime_mode_actor_based(
    runtime: Arc<SingleThreadedRuntime>,
    stt_topic: Topic<AudioBufferMessage>,
    _silence_threshold: f32,
    silence_duration_ms: u32,
    max_recording_duration: u32,
) -> Result<()> {
    println!("🎬 Starting actor-based real-time mode...");
    println!("💬 This is an interactive voice conversation!");
    println!("📋 Instructions:");
    println!("   • Speak clearly into your microphone");
    println!("   • Wait for the agent to respond completely");
    println!("   • Then ask your next question");
    println!("   • Press Ctrl+C to exit");
    println!("────────────────────────────────────────");

    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device found"))?;

    let config = input_device.default_input_config()?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    println!("Using input device: {}", input_device.name()?);
    println!("Sample rate: {}Hz, Channels: {}", sample_rate, channels);

    let _vad = VoiceActivityDetector::new(sample_rate, 250, silence_duration_ms);
    println!(
        "🎯 VAD initialized: min_voice=250ms, min_silence={}ms",
        silence_duration_ms
    );

    let audio_buffer = Arc::new(Mutex::new(VecDeque::<f32>::new()));
    let buffer_clone = audio_buffer.clone();
    let is_recording = Arc::new(Mutex::new(false));
    let recording_clone = is_recording.clone();

    // Create a separate VAD for the audio callback
    let vad_for_callback = Arc::new(Mutex::new(VoiceActivityDetector::new(
        sample_rate,
        250,
        silence_duration_ms,
    )));
    let vad_clone = vad_for_callback.clone();

    // Create a channel to send audio data from the callback to the async context
    let (audio_tx, mut audio_rx) = tokio::sync::mpsc::channel::<AudioBufferMessage>(10);

    // Audio input stream with advanced VAD and channel-based messaging
    let stream = input_device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut buffer = buffer_clone.lock().unwrap();
            let mut recording = recording_clone.lock().unwrap();
            let mut vad_callback = vad_clone.lock().unwrap();

            // Convert multi-channel to mono
            let mono_audio: Vec<f32> = data
                .chunks(channels)
                .map(|chunk| chunk.first().copied().unwrap_or(0.0))
                .collect();

            // Run VAD analysis
            let vad_result = vad_callback.process_audio(&mono_audio);
            let stats = vad_callback.get_stats();

            match vad_result {
                VadResult::Voice => {
                    if !*recording {
                        println!(
                            "🎤 Voice detected! (noise_level: {:.6}, threshold: {:.6})",
                            stats.background_noise_level, stats.current_threshold
                        );
                        *recording = true;
                        buffer.clear(); // Start fresh
                    }
                    // Add audio to buffer
                    for &sample in &mono_audio {
                        buffer.push_back(sample);
                    }
                }
                VadResult::Silence => {
                    if *recording {
                        println!("🔇 Processing your speech...");
                        *recording = false;
                        let audio_data: Vec<f32> = buffer.drain(..).collect();
                        if !audio_data.is_empty() {
                            // Send audio buffer via channel (non-blocking)
                            let audio_message = AudioBufferMessage {
                                audio_data,
                                sample_rate,
                            };

                            if let Err(_) = audio_tx.try_send(audio_message) {
                                eprintln!("⚠️ Audio buffer channel full, dropping message");
                            }
                        }
                    }
                }
                VadResult::Transition => {
                    if *recording {
                        // Continue recording during transition
                        for &sample in &mono_audio {
                            buffer.push_back(sample);
                        }
                    }
                }
            }

            // Safety: Limit recording duration
            if *recording && buffer.len() > (max_recording_duration * sample_rate) as usize {
                println!("⏰ Maximum recording duration reached, processing via actor...");
                *recording = false;
                let audio_data: Vec<f32> = buffer.drain(..).collect();
                if !audio_data.is_empty() {
                    let audio_message = AudioBufferMessage {
                        audio_data,
                        sample_rate,
                    };

                    if let Err(_) = audio_tx.try_send(audio_message) {
                        eprintln!("⚠️ Audio buffer channel full, dropping message");
                    }
                }
            }
        },
        move |err| eprintln!("Audio input error: {}", err),
        None,
    )?;

    stream.play()?;

    println!("🎤 Voice agent is ready! Say something to start the conversation...");

    // Spawn task to receive audio from channel and publish to STT actor
    let runtime_clone = runtime.clone();
    let topic_clone = stt_topic.clone();

    let turn_counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let turn_counter_clone = turn_counter.clone();

    tokio::spawn(async move {
        while let Some(audio_message) = audio_rx.recv().await {
            let turn_num =
                turn_counter_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            println!("🔄 Turn {} - Processing your speech...", turn_num);
            if let Err(e) = runtime_clone.publish(&topic_clone, audio_message).await {
                eprintln!("❌ Failed to publish audio buffer to STT actor: {}", e);
            }
        }
    });

    // Keep the stream alive - the actors handle the processing pipeline
    loop {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
}
