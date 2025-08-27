use crate::cli::AudioBufferMessage;
use crate::stt::STTProcessor;
use autoagents::{
    async_trait,
    core::{
        actor::Topic,
        agent::task::Task,
        ractor::{Actor, ActorProcessingErr, ActorRef},
        runtime::{SingleThreadedRuntime, TypedRuntime},
    },
};
use std::sync::Mutex;
use std::{path::PathBuf, sync::Arc};

/// STT Actor that processes audio buffers and publishes transcribed text
pub struct STTActor {
    pub name: String,
    pub runtime: Arc<SingleThreadedRuntime>,
    pub stt_processor: Arc<Mutex<Option<STTProcessor>>>,
}

impl STTActor {
    pub fn new(name: impl Into<String>, runtime: Arc<SingleThreadedRuntime>) -> Self {
        Self {
            name: name.into(),
            runtime,
            stt_processor: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn initialize_stt(&self, model_path: PathBuf) -> anyhow::Result<()> {
        let processor = STTProcessor::new(model_path).await?;
        let mut stt_lock = self.stt_processor.lock().unwrap();
        *stt_lock = Some(processor);
        Ok(())
    }
}

pub struct STTActorArgs {}

#[async_trait]
impl Actor for STTActor {
    type Msg = AudioBufferMessage;
    type State = ();
    type Arguments = STTActorArgs;

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        _args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        println!("🎭 STT Actor '{}' started", self.name);
        Ok(())
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        _state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        // Process the audio buffer through STT
        let transcribed_text = {
            let mut stt_lock = self.stt_processor.lock().unwrap();
            if let Some(ref mut processor) = *stt_lock {
                // Processing speech silently for cleaner UI
                match processor.transcribe_audio(&message.audio_data) {
                    Ok(text) => {
                        // STT completed
                        text
                    }
                    Err(e) => {
                        eprintln!("❌ STT processing error: {}", e);
                        return Err(format!("STT processing error: {}", e).into());
                    }
                }
            } else {
                eprintln!("❌ STT processor not initialized");
                return Err("Processing error".into());
            }
        };

        // If we have non-empty text, send it to the LLM agent
        if !transcribed_text.trim().is_empty() {
            println!("🤖 You said: \"{}\"", transcribed_text);
            println!("💭 Thinking...");
            if let Err(e) = self
                .runtime
                .publish(
                    &Topic::<Task>::new("agent_topic"),
                    Task::new(transcribed_text),
                )
                .await
            {
                eprintln!("❌ Failed to publish to agent topic: {}", e);
                return Err("Processing error".into());
            }
        } else {
            println!("🔇 No speech detected, skipping agent processing");
        }

        Ok(())
    }
}
