#![allow(
    dead_code,
    unused_variables,
    unused_mut,
    unreachable_patterns,
    unused_variables,
    unused_imports,
    unreachable_code
)]

use crate::agent::VoiceAgent;
use crate::cli;
use crate::kokoros::actor::{TTSActor, TTSActorArgs, TTSConfig};
use crate::kokoros::tts::koko::TTSKoko;
use crate::stt::actor::{STTActor, STTActorArgs};
use crate::stt::STTProcessor;
use crate::ui::VoiceAgentApp;
use crate::utils::{
    resample_audio_buffer, run_file_mode, run_realtime_mode_actor_based, run_test_mode,
    WHISPER_SAMPLE_RATE,
};
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
use std::cmp::PartialEq;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio_stream::{wrappers::ReceiverStream, StreamExt as TokioStreamExt};

#[derive(Subcommand, Debug, Clone, PartialEq)]
pub enum Mode {
    /// Take an input audio file and convert to text, then generate speech
    File {
        /// Path to input audio file (WAV format)
        #[arg(short, long, default_value = "examples/voice_agent/data/input.wav")]
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
        #[arg(long, default_value = "6")]
        max_recording_duration: u32,
    },
    UI,
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
        default_value_t = 1.3
    )]
    pub speed: f32,

    /// Output audio in mono (as opposed to stereo)
    #[arg(long = "mono", default_value_t = false)]
    pub mono: bool,

    /// Initial silence duration in tokens
    #[arg(long = "initial-silence", value_name = "INITIAL_SILENCE")]
    pub initial_silence: Option<usize>,

    /// STT model to use
    #[arg(
        long = "stt-model",
        default_value = "./examples/voice_agent/models/ggml-base.en.bin"
    )]
    pub stt_model: String,

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

/// UI update message type for sending updates to the GUI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UIUpdate {
    TranscriptionUpdate(String),
    AgentResponse(String),
    ProcessingStarted,
    ProcessingComplete,
    SpeakingStarted,
    SpeakingComplete,
}

/// Buffer for streaming text with natural pause detection
#[derive(Debug)]
pub struct StreamingBuffer {
    buffer: String,
    last_token_time: Instant,
    flush_timeout_ms: u64,
}

impl StreamingBuffer {
    pub(crate) fn new(flush_timeout_ms: u64) -> Self {
        Self {
            buffer: String::new(),
            last_token_time: Instant::now(),
            flush_timeout_ms,
        }
    }

    pub(crate) fn add_token(&mut self, token: &str) {
        self.buffer.push_str(token);
        self.last_token_time = Instant::now();
    }

    pub(crate) fn should_flush(&self) -> bool {
        if self.buffer.is_empty() {
            return false;
        }

        // Check for natural pause tokens (sentence endings, phrase pauses)
        if self.buffer.ends_with('.')
            || self.buffer.ends_with('!')
            || self.buffer.ends_with('?')
            || self.buffer.ends_with(", ")
            || self.buffer.ends_with("; ")
        {
            return true;
        }

        // Check timeout rule: if buffer has content and timeout exceeded
        let elapsed = self.last_token_time.elapsed();
        elapsed >= Duration::from_millis(self.flush_timeout_ms) && !self.buffer.trim().is_empty()
    }

    pub(crate) fn flush(&mut self) -> String {
        let content = self.buffer.clone();
        self.buffer.clear();
        content.trim().to_string()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.buffer.trim().is_empty()
    }
}

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
    let mut stt_processor = STTProcessor::new(PathBuf::from(cli.stt_model.clone())).await?;
    println!("✅ STT initialized successfully");

    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or("".into());

    // Initialize and configure the LLM client
    let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(api_key) // Set the API key
        .model("gpt-4o") // Use GPT-4o-mini model
        .max_tokens(512) // Limit response length
        .temperature(0.2) // Control response randomness (0.0-1.0)
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
        .stream(true)
        .build()
        .await?;

    // Set up event handling
    let receiver = environment.take_event_receiver(None).await?;

    // Create recording control for synchronization
    let recording_control = Arc::new(tokio::sync::RwLock::new(true));

    // Create actors with proper configuration
    let tts_config = TTSConfig {
        language: cli.lan.clone(),
        style: cli.style.clone(),
        speed: cli.speed,
        mono: cli.mono,
        initial_silence: cli.initial_silence,
    };

    let tts_actor = TTSActor::new("TTS_Actor", runtime.clone(), tts_config)
        .with_recording_control(recording_control.clone());
    let stt_actor = STTActor::new("STT_Actor", runtime.clone());

    // Initialize the actors with their models
    tts_actor
        .initialize_tts(&cli.model_path, &cli.data_path)
        .await?;
    stt_actor
        .initialize_stt(PathBuf::from(cli.stt_model))
        .await?;

    let (tts_actor_ref, _) = Actor::spawn(None, tts_actor, TTSActorArgs {}).await?;
    let (stt_actor_ref, _) = Actor::spawn(None, stt_actor, STTActorArgs {}).await?;

    // Subscribe actors to topics
    runtime.subscribe(&tts_topic, tts_actor_ref.clone()).await?;
    runtime.subscribe(&stt_topic, stt_actor_ref.clone()).await?;

    //Run indefinitely listening to the events
    handle_streaming_events(receiver, runtime.clone(), tts_topic.clone(), None);
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
                tts_topic.clone(),
                max_recording_duration,
                Some(recording_control.clone()),
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
        _ => Ok(()),
    }
}

fn handle_streaming_events(
    mut event_stream: ReceiverStream<Event>,
    runtime: Arc<SingleThreadedRuntime>,
    topic: Topic<SimpleMessage>,
    ui_sender: Option<tokio::sync::mpsc::UnboundedSender<UIUpdate>>,
) {
    tokio::spawn(async move {
        // Create streaming buffer with 500ms timeout for flushing
        let buffer = Arc::new(Mutex::new(StreamingBuffer::new(500)));
        let runtime_clone = runtime.clone();
        let topic_clone = topic.clone();
        let buffer_clone = buffer.clone();
        let ui_sender_clone = ui_sender.clone();

        // Start periodic flush task for timeout-based flushing
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(100));

            loop {
                interval.tick().await;

                let should_flush = {
                    let buffer_guard = buffer_clone.lock().unwrap();
                    buffer_guard.should_flush()
                };

                if should_flush {
                    let content = {
                        let mut buffer_guard = buffer_clone.lock().unwrap();
                        buffer_guard.flush()
                    };

                    if !content.is_empty() {
                        println!("⏰ Timeout flush: {}", content);
                        let _ = runtime_clone
                            .publish(
                                &topic_clone,
                                SimpleMessage {
                                    content: content.clone(),
                                },
                            )
                            .await;

                        // Send UI update if UI sender is available
                        if let Some(ref sender) = ui_sender_clone {
                            let _ = sender.send(UIUpdate::AgentResponse(content));
                        }
                    }
                }
            }
        });

        // Main event processing loop
        while let Some(event) = TokioStreamExt::next(&mut event_stream).await {
            match event {
                Event::TaskComplete { result, .. } => {
                    // Flush any remaining buffer content when task completes
                    let remaining_content = {
                        let mut buffer_guard = buffer.lock().unwrap();
                        if !buffer_guard.is_empty() {
                            buffer_guard.flush()
                        } else {
                            String::new()
                        }
                    };

                    if !remaining_content.is_empty() {
                        println!("✅ Final flush on completion: {}", remaining_content);
                        let _ = runtime
                            .publish(
                                &topic,
                                SimpleMessage {
                                    content: remaining_content.clone(),
                                },
                            )
                            .await;

                        // Send UI update if UI sender is available
                        if let Some(ref sender) = ui_sender {
                            let _ = sender.send(UIUpdate::AgentResponse(remaining_content));
                        }
                    }

                    match result {
                        TaskResult::Value(val) => {
                            match serde_json::from_value::<ReActAgentOutput>(val) {
                                Ok(agent_out) => {
                                    println!(
                                        "🌊 Task completed with response: {:?}",
                                        agent_out.response
                                    );
                                }
                                Err(_) => continue,
                            }
                        }
                        TaskResult::Failure(error) => {
                            println!("❌ Task failed: {}", error);
                        }
                        TaskResult::Aborted => {
                            println!("⚠️ Task was aborted");
                        }
                    }
                }
                Event::StreamChunk { sub_id, chunk } => {
                    let content = chunk.delta.content.unwrap_or_default();

                    if !content.is_empty() {
                        println!("📦 Stream token ({}): '{}'", sub_id, content);

                        // Add token to buffer
                        {
                            let mut buffer_guard = buffer.lock().unwrap();
                            buffer_guard.add_token(&content);
                        }

                        // Check if we should flush immediately based on natural pauses
                        let should_flush = {
                            let buffer_guard = buffer.lock().unwrap();
                            buffer_guard.should_flush()
                        };

                        if should_flush {
                            let chunk_content = {
                                let mut buffer_guard = buffer.lock().unwrap();
                                buffer_guard.flush()
                            };

                            if !chunk_content.is_empty() {
                                println!("🚀 Natural pause flush: {}", chunk_content);
                                let _ = runtime
                                    .publish(
                                        &topic,
                                        SimpleMessage {
                                            content: chunk_content.clone(),
                                        },
                                    )
                                    .await;

                                // Send UI update if UI sender is available
                                if let Some(ref sender) = ui_sender {
                                    let _ = sender.send(UIUpdate::AgentResponse(chunk_content));
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    });
}
