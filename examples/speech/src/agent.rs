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
use autoagents_speech::{AudioFormat, SpeechRequest, TTSSpeechProvider, VoiceIdentifier};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::vad_stt::{InputMode, build_parakeet_provider, build_vad_segmenter};

const DEFAULT_AGENT_MODEL: &str = "gpt-4o-mini";
const DEFAULT_VOICE: &str = "eponine";

type VadPipeline = VadSttPipeline<SileroVad, Parakeet>;
type SpeechAgentHandle = DirectAgentHandle<ReActAgent<SpeechAgent>>;

static EXIT_REQUESTED: AtomicBool = AtomicBool::new(false);

fn reset_exit_flag() {
    EXIT_REQUESTED.store(false, Ordering::SeqCst);
}

fn exit_requested() -> bool {
    EXIT_REQUESTED.load(Ordering::SeqCst)
}

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
struct ExitConversation;

#[autoagents::async_trait]
impl ToolRuntime for ExitConversation {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let typed: ExitConversationArgs = serde_json::from_value(args)?;
        EXIT_REQUESTED.store(true, Ordering::SeqCst);
        Ok(json!({
            "status": "ok",
            "reason": typed.reason,
        }))
    }
}

#[agent(
    name = "speech_assistant",
    description = "You are a concise voice assistant, Your name is Tess. Respond in short, speakable sentences.

If the user asks to stop, end the conversation, or says goodbye, call the exit_conversation tool instead of continuing.",
    tools = [ExitConversation],
)]
#[derive(Default, Clone, AgentHooks)]
pub struct SpeechAgent;

#[derive(Debug)]
pub struct AgentArgs {
    pub input: InputMode,
    pub audio_file: Option<PathBuf>,
    pub language: Option<String>,
    pub agent_model: Option<String>,
    pub voice: Option<String>,
}

pub async fn run(args: AgentArgs) -> Result<(), Box<dyn std::error::Error>> {
    reset_exit_flag();
    let mut pipeline = build_pipeline(&args)?;
    let agent_handle = build_agent(args.agent_model.as_deref()).await?;
    let tts = PocketTTS::new(None)?;
    let player = AudioPlayer::try_new().ok();

    match args.input {
        InputMode::File => {
            run_file(&args, &mut pipeline, &agent_handle, &tts, player.as_ref()).await?
        }
        InputMode::Mic => {
            run_mic(&args, &mut pipeline, &agent_handle, &tts, player.as_ref()).await?
        }
    }

    Ok(())
}

fn build_pipeline(args: &AgentArgs) -> Result<VadPipeline, Box<dyn std::error::Error>> {
    let segmenter = build_vad_segmenter()?;
    let stt_provider = build_parakeet_provider()?;
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
    let agent = ReActAgent::new(SpeechAgent);
    let handle = AgentBuilder::<_, DirectAgent>::new(agent)
        .llm(llm)
        .memory(memory)
        .build()
        .await?;

    Ok(handle)
}

async fn run_file(
    args: &AgentArgs,
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
        let should_exit = handle_segment(args, &segment, agent, tts, player, None).await?;
        if should_exit {
            println!("Exit requested. Ending session.");
            break;
        }
    }

    Ok(())
}

async fn run_mic(
    args: &AgentArgs,
    pipeline: &mut VadPipeline,
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

    let window_samples = pipeline.window_samples();
    let window_ms = pipeline.window_ms();
    let poll_interval = Duration::from_millis((window_ms / 3).max(10) as u64);
    let stream = capture.start_stream()?;
    println!("Listening for speech. Press Ctrl+C to stop.");

    loop {
        tokio::time::sleep(poll_interval).await;
        while let Some(chunk) = stream.read_chunk(window_samples)? {
            let segments = pipeline.process_audio(&chunk).await?;
            for segment in segments {
                let should_exit =
                    handle_segment(args, &segment, agent, tts, player, Some(&stream)).await?;
                if should_exit {
                    println!("Exit requested. Ending session.");
                    return Ok(());
                }
            }
        }
    }
}

async fn handle_segment(
    args: &AgentArgs,
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

    let output = agent.agent.run(Task::new(user_text)).await?;
    let response_text = output.trim().to_string();
    let should_exit = exit_requested();
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
