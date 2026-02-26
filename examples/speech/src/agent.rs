use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::ReActAgent;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, DirectAgent, DirectAgentHandle};
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents::llm::backends::openai::OpenAI;
use autoagents::llm::builder::LLMBuilder;
use autoagents_derive::{AgentHooks, ToolInput, agent, tool};
use autoagents_speech::audio_capture::{
    AudioCapture, AudioCaptureConfig, AudioCaptureError, AudioCaptureStream,
};
use autoagents_speech::playback::AudioPlayer;
use autoagents_speech::providers::parakeet::Parakeet;
use autoagents_speech::providers::pocket_tts::PocketTTS;
use autoagents_speech::vad::{SegmentTranscription, SileroVad, VadSttConfig, VadSttPipeline};
use autoagents_speech::{
    AudioFormat, SpeechRequest, TTSSpeechProvider, TextChunk, VoiceIdentifier,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::vad_stt::{InputMode, build_parakeet_streaming_provider, build_vad_segmenter};

const DEFAULT_AGENT_MODEL: &str = "gpt-4o-mini";
const DEFAULT_VOICE: &str = "eponine";

type VadPipeline = VadSttPipeline<SileroVad, Parakeet>;
type SpeechAgentHandle = DirectAgentHandle<ReActAgent<SpeechAgent>>;

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct ExitConversationArgs {
    #[input(description = "Optional reason for ending the conversation")]
    reason: Option<String>,
}

#[tool(
    name = "exit_conversation",
    description = "End the conversation when the user says goodbye or asks to stop.",
    input = ExitConversationArgs,
)]
struct ExitConversation {
    exit_flag: Arc<AtomicBool>,
}

#[autoagents::async_trait]
impl ToolRuntime for ExitConversation {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let typed: ExitConversationArgs = serde_json::from_value(args)?;
        self.exit_flag.store(true, Ordering::SeqCst);
        Ok(json!({
            "status": "ok",
            "reason": typed.reason,
        }))
    }
}

#[agent(
    name = "speech_assistant",
    description = "You are a concise voice assistant, Your name is Tess, You are more sentient and human like. Respond in short, speakable sentences.

If the user asks to stop, end the conversation, or says goodbye, call the exit_conversation tool instead of continuing.",
    tools = [ExitConversation { exit_flag: self.exit_flag.clone() }],
)]
#[derive(Clone, AgentHooks)]
pub struct SpeechAgent {
    exit_flag: Arc<AtomicBool>,
}

impl SpeechAgent {
    fn new(exit_flag: Arc<AtomicBool>) -> Self {
        Self { exit_flag }
    }
}

#[derive(Debug)]
pub struct AgentArgs {
    pub input: InputMode,
    pub audio_file: Option<PathBuf>,
    pub language: Option<String>,
    pub agent_model: Option<String>,
    pub voice: Option<String>,
}

pub async fn run(args: AgentArgs) -> Result<(), Box<dyn std::error::Error>> {
    let exit_flag = Arc::new(AtomicBool::new(false));
    let agent_handle = build_agent(args.agent_model.as_deref(), exit_flag.clone()).await?;
    let tts = PocketTTS::new(None)?;
    let player = AudioPlayer::try_new().ok();

    match args.input {
        InputMode::File => {
            let mut pipeline = build_pipeline(&args)?;
            run_file(
                &args,
                &exit_flag,
                &mut pipeline,
                &agent_handle,
                &tts,
                player.as_ref(),
            )
            .await?
        }
        InputMode::Mic => {
            run_mic_streaming(&args, &exit_flag, &agent_handle, &tts, player.as_ref()).await?
        }
    }

    Ok(())
}

fn build_pipeline(args: &AgentArgs) -> Result<VadPipeline, Box<dyn std::error::Error>> {
    let segmenter = build_vad_segmenter()?;
    let stt_provider = build_parakeet_streaming_provider()?;
    let pipeline = VadSttPipeline::new(
        segmenter,
        stt_provider,
        VadSttConfig {
            language: args.language.clone(),
            include_timestamps: false,
        },
    );
    Ok(pipeline)
}

async fn build_agent(
    model_override: Option<&str>,
    exit_flag: Arc<AtomicBool>,
) -> Result<SpeechAgentHandle, Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| "OPENAI_API_KEY must be set for the agent example")?;
    let model = model_override.unwrap_or(DEFAULT_AGENT_MODEL);

    let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(api_key)
        .model(model)
        .max_tokens(256)
        .temperature(0.4)
        .build()
        .map_err(|e| format!("Failed to build OpenAI client: {e}"))?;

    let memory = Box::new(SlidingWindowMemory::new(8));
    let agent = ReActAgent::new(SpeechAgent::new(exit_flag));
    let handle = AgentBuilder::<_, DirectAgent>::new(agent)
        .llm(llm)
        .memory(memory)
        .build()
        .await?;

    Ok(handle)
}

async fn run_file(
    args: &AgentArgs,
    exit_flag: &Arc<AtomicBool>,
    pipeline: &mut VadPipeline,
    agent: &SpeechAgentHandle,
    tts: &PocketTTS,
    player: Option<&AudioPlayer>,
) -> Result<(), Box<dyn std::error::Error>> {
    let audio_path = args
        .audio_file
        .as_ref()
        .ok_or("Provide --audio-file for file input")?;
    let audio = AudioCapture::read_audio_with_config(audio_path, AudioCaptureConfig::default())?;

    let mut segments = pipeline.process_audio(&audio).await?;
    if let Some(final_segment) = pipeline.finalize().await? {
        segments.push(final_segment);
    }

    if segments.is_empty() {
        println!("No speech segments detected.");
        return Ok(());
    }

    for segment in segments {
        let should_exit =
            handle_segment(args, exit_flag, &segment, agent, tts, player, None).await?;
        if should_exit {
            println!("Exit requested. Ending session.");
            break;
        }
    }

    Ok(())
}

async fn run_mic_streaming(
    args: &AgentArgs,
    exit_flag: &Arc<AtomicBool>,
    agent: &SpeechAgentHandle,
    tts: &PocketTTS,
    player: Option<&AudioPlayer>,
) -> Result<(), Box<dyn std::error::Error>> {
    let capture = match AudioCapture::with_config(AudioCaptureConfig::default()) {
        Ok(capture) => capture,
        Err(AudioCaptureError::NoInputDevice) => {
            println!("No input device available.");
            return Ok(());
        }
        Err(err) => return Err(err.into()),
    };

    let stt = build_parakeet_streaming_provider()?;
    let mut segmenter = build_vad_segmenter()?;
    let model_variant = stt.config().model_variant;
    let chunk_samples = model_variant.chunk_size();
    let chunk_ms = model_variant.chunk_duration_ms();
    let poll_interval = Duration::from_millis((chunk_ms / 4).max(10) as u64);

    let stream = capture.start_stream()?;
    println!("Listening for speech. Press Ctrl+C to stop.");

    let mut current_text = String::new();
    let mut last_print_len = 0usize;

    loop {
        if exit_flag.load(Ordering::SeqCst) {
            break;
        }

        tokio::time::sleep(poll_interval).await;
        while let Some(chunk) = stream.read_chunk(chunk_samples)? {
            let segments = segmenter.process_audio(&chunk)?;

            // Only run STT inference during active speech or at segment boundaries.
            // This avoids wasting inference cycles on silent audio.
            let active = segmenter.in_speech() || !segments.is_empty();
            let text_chunk = if active {
                stt.process_chunk(chunk.samples).await?
            } else {
                TextChunk {
                    text: String::new(),
                    is_final: false,
                }
            };

            if !text_chunk.text.is_empty() {
                merge_partial_text(&mut current_text, &text_chunk.text);
                let display = format!("User: {}", current_text.trim_end());
                if last_print_len > display.len() {
                    let padding = " ".repeat(last_print_len - display.len());
                    print!("\r{display}{padding}");
                } else {
                    print!("\r{display}");
                }
                last_print_len = display.len();
                std::io::stdout().flush()?;
            }

            if text_chunk.is_final || !segments.is_empty() {
                println!();
                let should_exit = handle_user_text(
                    args,
                    exit_flag,
                    current_text.trim(),
                    agent,
                    tts,
                    player,
                    Some(&stream),
                )
                .await?;
                current_text.clear();
                last_print_len = 0;
                stt.reset().await;
                if should_exit {
                    println!("Exit requested. Ending session.");
                    return Ok(());
                }
            }
        }
    }

    Ok(())
}

async fn handle_segment(
    args: &AgentArgs,
    exit_flag: &Arc<AtomicBool>,
    segment: &SegmentTranscription,
    agent: &SpeechAgentHandle,
    tts: &PocketTTS,
    player: Option<&AudioPlayer>,
    stream: Option<&AudioCaptureStream>,
) -> Result<bool, Box<dyn std::error::Error>> {
    let user_text = segment.transcription.text.trim();
    if user_text.is_empty() {
        return Ok(false);
    }

    let start_sec = segment.segment.start_ms as f32 / 1000.0;
    let end_sec = segment.segment.end_ms as f32 / 1000.0;
    println!("[{start_sec:.2}s - {end_sec:.2}s] User: {user_text}");

    handle_user_text(args, exit_flag, user_text, agent, tts, player, stream).await
}

async fn handle_user_text(
    args: &AgentArgs,
    exit_flag: &Arc<AtomicBool>,
    user_text: &str,
    agent: &SpeechAgentHandle,
    tts: &PocketTTS,
    player: Option<&AudioPlayer>,
    stream: Option<&AudioCaptureStream>,
) -> Result<bool, Box<dyn std::error::Error>> {
    if user_text.is_empty() {
        return Ok(false);
    }

    let output = agent.agent.run(Task::new(user_text)).await?;
    let response_text = output.trim().to_string();
    let should_exit = exit_flag.load(Ordering::SeqCst);
    if response_text.is_empty() && !should_exit {
        return Ok(false);
    }

    if !response_text.is_empty() {
        println!("Assistant: {response_text}");
    }

    if !response_text.is_empty() {
        let request = SpeechRequest {
            text: response_text,
            voice: VoiceIdentifier::new(args.voice.as_deref().unwrap_or(DEFAULT_VOICE)),
            format: AudioFormat::Wav,
            sample_rate: Some(24_000),
        };
        let response = tts.generate_speech(request).await?;

        if let Some(player) = player {
            player.play_samples(&response.audio.samples, response.audio.sample_rate);
            player.wait_until_end();

            if let Some(stream) = stream {
                stream.clear_buffer()?;
            }
        }
    }

    if stream.is_some() && !should_exit {
        println!("Listening...");
    }

    Ok(should_exit)
}

fn merge_partial_text(current: &mut String, incoming: &str) {
    let incoming = incoming.trim();
    if incoming.is_empty() {
        return;
    }

    if current.is_empty() {
        current.push_str(incoming);
        return;
    }

    if incoming.starts_with(current.as_str()) {
        current.clear();
        current.push_str(incoming);
        return;
    }

    if current.starts_with(incoming) {
        return;
    }

    if !current.ends_with(' ') {
        current.push(' ');
    }
    current.push_str(incoming);
}
