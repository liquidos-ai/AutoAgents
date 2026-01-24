#[cfg(not(target_arch = "wasm32"))]
use crate::actor::{ActorMessage, Topic};
use crate::agent::AgentConfig;
use crate::agent::memory::MemoryProvider;
use crate::agent::state::AgentState;
use crate::protocol::Event;
use crate::tool::ToolT;
use autoagents_llm::LLMProvider;
use autoagents_llm::chat::ChatMessage;
use std::any::Any;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::{Mutex, mpsc};

#[cfg(target_arch = "wasm32")]
use futures::channel::mpsc;
#[cfg(target_arch = "wasm32")]
use futures::lock::Mutex;

#[cfg(feature = "tts")]
use autoagents_tts::{AudioStoragePolicy, SharedAudioData, SpeechRequest, SpeechResponse, TTSProvider, VoiceIdentifier};

#[cfg(feature = "tts")]
/// Audio output with metadata
#[derive(Clone, Debug)]
pub struct AudioOutput {
    pub audio: Option<SharedAudioData>,
    pub text: String,
    pub duration_ms: u64,
}

/// Execution context shared across an agent run.
///
/// Holds the configured LLM provider, accumulated chat messages, optional
/// memory and tools, agent configuration, ephemeral execution state, and
/// an optional event transmitter used by actor-based agents to emit protocol
/// `Event`s. Also carries a `stream` flag to indicate streaming mode.
pub struct Context {
    llm: Arc<dyn LLMProvider>,
    messages: Vec<ChatMessage>,
    memory: Option<Arc<Mutex<Box<dyn MemoryProvider>>>>,
    tools: Vec<Box<dyn ToolT>>,
    config: AgentConfig,
    state: Arc<Mutex<AgentState>>,
    tx: Option<mpsc::Sender<Event>>,
    stream: bool,
    #[cfg(feature = "tts")]
    tts: Option<Arc<dyn TTSProvider>>,
    #[cfg(feature = "tts")]
    audio_outputs: Vec<AudioOutput>,
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum ContextError {
    #[error("Tx value is None, Tx is only set for Actor agents")]
    EmptyTx,
    /// Error when sending events
    #[error("Failed to send event: {0}")]
    EventSendError(String),
    /// TTS provider is not configured
    #[cfg(feature = "tts")]
    #[error("TTS provider is not configured")]
    MissingTTS,
    /// TTS error
    #[cfg(feature = "tts")]
    #[error("TTS error: {0}")]
    TTSError(String),
}

impl Context {
    pub fn new(llm: Arc<dyn LLMProvider>, tx: Option<mpsc::Sender<Event>>) -> Self {
        Self {
            llm,
            messages: vec![],
            memory: None,
            tools: vec![],
            config: AgentConfig::default(),
            state: Arc::new(Mutex::new(AgentState::new())),
            stream: false,
            tx,
            #[cfg(feature = "tts")]
            tts: None,
            #[cfg(feature = "tts")]
            audio_outputs: vec![],
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn publish<M: ActorMessage>(
        &self,
        topic: Topic<M>,
        message: M,
    ) -> Result<(), ContextError> {
        self.tx
            .as_ref()
            .ok_or(ContextError::EmptyTx)?
            .send(Event::PublishMessage {
                topic_name: topic.name().to_string(),
                message: Arc::new(message) as Arc<dyn Any + Send + Sync>,
                topic_type: topic.type_id(),
            })
            .await
            .map_err(|e| ContextError::EventSendError(e.to_string()))
    }

    pub fn with_memory(mut self, memory: Option<Arc<Mutex<Box<dyn MemoryProvider>>>>) -> Self {
        self.memory = memory;
        self
    }

    pub fn with_tools(mut self, tools: Vec<Box<dyn ToolT>>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_config(mut self, config: AgentConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_messages(mut self, messages: Vec<ChatMessage>) -> Self {
        self.messages = messages;
        self
    }

    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    #[cfg(feature = "tts")]
    pub fn with_tts(mut self, tts: Option<Arc<dyn TTSProvider>>) -> Self {
        self.tts = tts;
        self
    }

    // Getters
    pub fn llm(&self) -> &Arc<dyn LLMProvider> {
        &self.llm
    }

    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    pub fn memory(&self) -> Option<Arc<Mutex<Box<dyn MemoryProvider>>>> {
        self.memory.clone()
    }

    pub fn tools(&self) -> &[Box<dyn ToolT>] {
        &self.tools
    }

    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    pub fn state(&self) -> Arc<Mutex<AgentState>> {
        self.state.clone()
    }

    /// Get a clone of the event transmitter (only present for Actor agents).
    pub fn tx(&self) -> Result<mpsc::Sender<Event>, ContextError> {
        Ok(self.tx.as_ref().ok_or(ContextError::EmptyTx)?.clone())
    }

    pub fn stream(&self) -> bool {
        self.stream
    }

    #[cfg(feature = "tts")]
    pub fn tts(&self) -> Option<&Arc<dyn TTSProvider>> {
        self.tts.as_ref()
    }

    #[cfg(feature = "tts")]
    pub fn audio_outputs(&self) -> &[AudioOutput] {
        &self.audio_outputs
    }

    #[cfg(feature = "tts")]
    pub fn latest_audio(&self) -> Option<&AudioOutput> {
        self.audio_outputs.last()
    }

    /// Generate speech from text (requires `tts` feature)
    #[cfg(feature = "tts")]
    pub async fn speak(&mut self, text: &str, voice: Option<VoiceIdentifier>) -> Result<Option<SharedAudioData>, ContextError> {
        let tts = self.tts.as_ref().ok_or(ContextError::MissingTTS)?;
        
        // Get voice identifier
        let voice_id = if let Some(v) = voice {
            v
        } else if let Some(default_voice) = &self.config.default_voice {
            VoiceIdentifier::Predefined(default_voice.clone())
        } else {
            // Use provider's default voice
            VoiceIdentifier::Predefined(tts.default_voice())
        };

        // Create speech request
        let request = SpeechRequest {
            text: text.to_string(),
            voice: voice_id,
            format: autoagents_tts::AudioFormat::Wav,
            sample_rate: Some(tts.default_sample_rate()),
        };

        // Generate speech
        let response: SpeechResponse = tts
            .generate_speech(request)
            .await
            .map_err(|e| ContextError::TTSError(e.to_string()))?;

        let audio = Arc::new(response.audio);
        
        // Apply storage policy
        match self.config.audio_storage_policy {
            AudioStoragePolicy::None => Ok(None),
            AudioStoragePolicy::OutputOnly => Ok(Some(Arc::clone(&audio))),
            AudioStoragePolicy::HistoryOnly => {
                self.audio_outputs.push(AudioOutput {
                    audio: Some(Arc::clone(&audio)),
                    text: response.text,
                    duration_ms: response.duration_ms,
                });
                Ok(None)
            }
            AudioStoragePolicy::Full => {
                self.audio_outputs.push(AudioOutput {
                    audio: Some(Arc::clone(&audio)),
                    text: response.text,
                    duration_ms: response.duration_ms,
                });
                Ok(Some(Arc::clone(&audio)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::memory::SlidingWindowMemory;
    use autoagents_llm::chat::{ChatMessage, ChatMessageBuilder, ChatRole};
    use autoagents_test_utils::llm::MockLLMProvider;
    use std::sync::Arc;

    #[test]
    fn test_context_creation() {
        let llm = Arc::new(MockLLMProvider);
        let context = Context::new(llm, None);

        assert!(context.messages.is_empty());
        assert!(context.memory.is_none());
        assert!(context.tools.is_empty());
        assert!(!context.stream);
    }

    #[test]
    fn test_context_with_llm_provider() {
        let llm = Arc::new(MockLLMProvider);
        let context = Context::new(llm.clone(), None);

        // Verify the LLM provider is set correctly
        let context_llm = context.llm();
        assert!(Arc::strong_count(context_llm) > 0);
    }

    #[test]
    fn test_context_with_memory() {
        let llm = Arc::new(MockLLMProvider);
        let memory = Box::new(SlidingWindowMemory::new(5));
        let context = Context::new(llm, None).with_memory(Some(Arc::new(Mutex::new(memory))));

        assert!(context.memory().is_some());
    }

    #[test]
    fn test_context_with_messages() {
        let llm = Arc::new(MockLLMProvider);
        let message = ChatMessage::user().content("Hello".to_string()).build();
        let context = Context::new(llm, None).with_messages(vec![message]);

        assert_eq!(context.messages().len(), 1);
        assert_eq!(context.messages()[0].role, ChatRole::User);
        assert_eq!(context.messages()[0].content, "Hello");
    }

    #[test]
    fn test_context_streaming_flag() {
        let llm = Arc::new(MockLLMProvider);
        let context = Context::new(llm, None).with_stream(true);
        assert!(context.stream());
    }

    #[test]
    fn test_context_fluent_interface() {
        let llm = Arc::new(MockLLMProvider);
        let memory = Box::new(SlidingWindowMemory::new(3));
        let message = ChatMessageBuilder::new(ChatRole::System)
            .content("System prompt".to_string())
            .build();

        let context = Context::new(llm, None)
            .with_memory(Some(Arc::new(Mutex::new(memory))))
            .with_messages(vec![message])
            .with_stream(true);

        assert!(context.memory().is_some());
        assert_eq!(context.messages().len(), 1);
        assert!(context.stream());
    }
}
