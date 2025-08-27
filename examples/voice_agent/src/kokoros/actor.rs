use crate::kokoros::tts::koko::TTSKoko;
use crate::{audio::AudioPlayback, cli::SimpleMessage};
use autoagents::{
    async_trait,
    core::{
        ractor::{Actor, ActorProcessingErr, ActorRef},
        runtime::SingleThreadedRuntime,
    },
};
use std::sync::Arc;
use std::sync::Mutex;

/// TTS Actor that processes text messages and generates speech audio
pub struct TTSActor {
    pub name: String,
    pub runtime: Arc<SingleThreadedRuntime>,
    pub tts_engine: Arc<Mutex<Option<TTSKoko>>>,
    pub tts_config: TTSConfig,
    pub recording_control: Option<Arc<tokio::sync::RwLock<bool>>>,
}

#[derive(Clone)]
pub struct TTSConfig {
    pub language: String,
    pub style: String,
    pub speed: f32,
    pub mono: bool,
    pub initial_silence: Option<usize>,
}

impl TTSActor {
    pub fn new(
        name: impl Into<String>,
        runtime: Arc<SingleThreadedRuntime>,
        config: TTSConfig,
    ) -> Self {
        Self {
            name: name.into(),
            runtime,
            tts_engine: Arc::new(Mutex::new(None)),
            tts_config: config,
            recording_control: None,
        }
    }

    pub fn with_recording_control(
        mut self,
        recording_control: Arc<tokio::sync::RwLock<bool>>,
    ) -> Self {
        self.recording_control = Some(recording_control);
        self
    }

    pub async fn initialize_tts(&self, model_path: &str, data_path: &str) -> anyhow::Result<()> {
        let tts = TTSKoko::new(model_path, data_path).await;
        let mut tts_lock = self.tts_engine.lock().unwrap();
        *tts_lock = Some(tts);
        Ok(())
    }
}

pub struct TTSActorArgs {}

#[async_trait]
impl Actor for TTSActor {
    type Msg = SimpleMessage;
    type State = ();
    type Arguments = TTSActorArgs;

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        _args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        println!("🎭 TTS Actor '{}' started", self.name);
        Ok(())
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        _state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        // Disable recording when TTS starts processing
        if let Some(ref recording_control) = self.recording_control {
            let mut recording_guard = recording_control.write().await;
            *recording_guard = false;
            println!("🔇 Recording disabled during TTS processing");
        }

        println!("🗣️ Agent response: \"{}\"", message.content);
        println!("🎵 Generating speech...");

        // Process the text through TTS to generate audio
        let audio_data = {
            let tts_lock = self.tts_engine.lock().unwrap();
            if let Some(ref tts) = *tts_lock {
                match tts.tts_raw_audio(
                    &message.content,
                    &self.tts_config.language,
                    &self.tts_config.style,
                    self.tts_config.speed,
                    self.tts_config.initial_silence,
                    None,
                    None,
                    None,
                ) {
                    Ok(audio) => audio,
                    Err(e) => {
                        eprintln!("❌ TTS processing error: {}", e);
                        return Err(format!("TTS processing error: {}", e).into());
                    }
                }
            } else {
                eprintln!("❌ TTS engine not initialized");
                return Err("Processing error".into());
            }
        };

        // TTS generation completed

        // Play the audio directly
        if !audio_data.is_empty() {
            match AudioPlayback::new() {
                Ok(playback) => {
                    if let Err(e) = playback.play_audio(audio_data.clone()) {
                        eprintln!("❌ Audio playback error: {}", e);
                        // Re-enable recording even on error
                        if let Some(ref recording_control) = self.recording_control {
                            let mut recording_guard = recording_control.write().await;
                            *recording_guard = true;
                            println!("🎤 Recording re-enabled after TTS error");
                        }
                        return Err(format!("TTS processing error: {}", e).into());
                    } else {
                        println!("✅ Response completed!");

                        // Re-enable recording after successful audio playback
                        if let Some(ref recording_control) = self.recording_control {
                            let mut recording_guard = recording_control.write().await;
                            *recording_guard = true;
                            println!("🎤 Recording re-enabled - your turn to speak!");
                        } else {
                            println!("🎤 Your turn - speak now...");
                        }
                    }
                }
                Err(e) => {
                    eprintln!("❌ Failed to create audio playback: {}", e);
                    // Re-enable recording on audio playback creation error
                    if let Some(ref recording_control) = self.recording_control {
                        let mut recording_guard = recording_control.write().await;
                        *recording_guard = true;
                        println!("🎤 Recording re-enabled after audio playback creation error");
                    }
                    return Err("Processing error".into());
                }
            }
        } else {
            println!("⚠️ TTS generated empty audio for: '{}'", message.content);
            // Re-enable recording when no audio is generated
            if let Some(ref recording_control) = self.recording_control {
                let mut recording_guard = recording_control.write().await;
                *recording_guard = true;
                println!("🎤 Recording re-enabled after empty TTS output");
            }
        }

        Ok(())
    }
}
