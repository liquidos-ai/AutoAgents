#![allow(
    dead_code,
    unused_variables,
    unused_mut,
    unreachable_patterns,
    unused_variables,
    unused_imports,
    unreachable_code
)]

use iced::widget::{button, column, container, row, scrollable, text};
use iced::{Color, Element, Length, Task, Theme};
use lazy_static::lazy_static;
use rand;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};

// Global UI message sender
lazy_static! {
    static ref UI_SENDER: Mutex<Option<mpsc::UnboundedSender<Message>>> = Mutex::new(None);
}

// Voice Agent imports
use crate::agent::VoiceAgent;
use crate::cli::StreamingBuffer;
use crate::cli::{setup_voice_system_with_ui, AudioBufferMessage, SimpleMessage, UIUpdate};
use crate::kokoros::actor::{TTSActor, TTSActorArgs, TTSConfig};
use crate::kokoros::tts::koko::TTSKoko;
use crate::stt::actor::{STTActor, STTActorArgs};
use crate::stt::STTProcessor;
use crate::utils::run_realtime_mode_actor_based;
use anyhow::Result;
use autoagents::core::actor::{ActorMessage, Topic};
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::ReActAgentOutput;
use autoagents::core::agent::task::Task as AgentTask;
use autoagents::core::agent::AgentBuilder;
use autoagents::core::environment::Environment;
use autoagents::core::protocol::Event;
use autoagents::core::protocol::TaskResult;
use autoagents::core::ractor::Actor;
use autoagents::core::runtime::SingleThreadedRuntime;
use autoagents::core::runtime::TypedRuntime;
use autoagents::llm::backends::openai::OpenAI;
use autoagents::llm::builder::LLMBuilder;
use serde_json;
use std::path::PathBuf;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

#[derive(Clone, Debug)]
pub enum ConversationEntry {
    User(String, Instant),
    Assistant(String, Instant),
    System(String, Instant),
}

pub struct VoiceAgentApp {
    // Audio state
    audio_level: f32,
    is_recording: bool,

    // Conversation state
    conversation_history: VecDeque<ConversationEntry>,
    current_user_speech: String,
    current_ai_response: String,
    is_ai_speaking: bool,

    // System state
    voice_system_initialized: bool,
    system_status: String,
    error_message: Option<String>,

    // UI message receiver
    ui_receiver: Option<mpsc::UnboundedReceiver<Message>>,

    // Performance metrics
    last_update: Instant,
    fps_counter: u32,
}

#[derive(Clone, Debug)]
pub enum Message {
    // Initialization
    InitializeVoiceSystem,
    VoiceSystemInitialized,
    InitializationError(String),

    // Audio updates
    AudioLevelUpdate(f32),
    RecordingStateChanged(bool),

    // Conversation updates
    UserSpeechStart,
    UserSpeechUpdate(String),
    UserSpeechEnd(String),
    AISpeechStart,
    AISpeechUpdate(String),
    AISpeechEnd(String),

    // System updates
    SystemStatus(String),
    ErrorOccurred(String),
    ClearError,

    // UI updates
    Tick,
    CheckForUpdates,
    ClearConversation,
    ScrollToBottom,
}

impl Default for VoiceAgentApp {
    fn default() -> Self {
        Self::new()
    }
}

impl VoiceAgentApp {
    pub fn new() -> Self {
        // Create UI message channel
        let (sender, receiver) = mpsc::unbounded_channel();

        // Store sender globally
        {
            let mut global_sender = UI_SENDER.lock().unwrap();
            *global_sender = Some(sender);
        }

        VoiceAgentApp {
            audio_level: 0.0,
            is_recording: false,
            conversation_history: VecDeque::new(),
            current_user_speech: String::new(),
            current_ai_response: String::new(),
            is_ai_speaking: false,
            voice_system_initialized: false,
            system_status: "Starting up...".to_string(),
            error_message: None,
            ui_receiver: Some(receiver),
            last_update: Instant::now(),
            fps_counter: 0,
        }
    }

    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::InitializeVoiceSystem => {
                self.system_status = "Initializing voice system...".to_string();
                Task::perform(Self::setup_voice_system(), |result| match result {
                    Ok(_) => Message::VoiceSystemInitialized,
                    Err(e) => Message::InitializationError(format!("Failed to initialize: {}", e)),
                })
            }
            Message::VoiceSystemInitialized => {
                self.voice_system_initialized = true;
                self.system_status = "Ready - Speak to Bella!".to_string();
                self.conversation_history
                    .push_back(ConversationEntry::System(
                        "Voice system initialized. Ready to chat!".to_string(),
                        Instant::now(),
                    ));

                // Start tick loop for updates
                Task::batch([
                    Task::perform(
                        async {
                            tokio::time::sleep(Duration::from_millis(16)).await;
                        },
                        |_| Message::Tick,
                    ),
                    Task::perform(async {}, |_| Message::CheckForUpdates),
                ])
            }
            Message::InitializationError(error) => {
                self.error_message = Some(error.clone());
                self.system_status = "Initialization failed".to_string();
                Task::none()
            }
            Message::AudioLevelUpdate(level) => {
                self.audio_level = level;
                Task::none()
            }
            Message::RecordingStateChanged(is_recording) => {
                self.is_recording = is_recording;
                if is_recording {
                    self.system_status = "Listening...".to_string();
                } else if self.voice_system_initialized {
                    self.system_status = "Processing...".to_string();
                }
                Task::none()
            }
            Message::UserSpeechStart => {
                self.current_user_speech.clear();
                self.is_recording = true;
                self.system_status = "Listening to you...".to_string();
                Task::none()
            }
            Message::UserSpeechUpdate(text) => {
                self.current_user_speech = text;
                Task::none()
            }
            Message::UserSpeechEnd(text) => {
                if !text.trim().is_empty() {
                    self.conversation_history
                        .push_back(ConversationEntry::User(text, Instant::now()));
                }
                self.current_user_speech.clear();
                self.is_recording = false;
                self.system_status = "Processing your request...".to_string();
                Task::perform(async {}, |_| Message::ScrollToBottom)
            }
            Message::AISpeechStart => {
                self.current_ai_response.clear();
                self.is_ai_speaking = true;
                self.system_status = "Bella is responding...".to_string();
                Task::none()
            }
            Message::AISpeechUpdate(text) => {
                self.current_ai_response = text;
                Task::none()
            }
            Message::AISpeechEnd(text) => {
                if !text.trim().is_empty() {
                    self.conversation_history
                        .push_back(ConversationEntry::Assistant(text, Instant::now()));
                }
                self.current_ai_response.clear();
                self.is_ai_speaking = false;
                self.system_status = "Ready - Speak to Bella!".to_string();
                Task::perform(async {}, |_| Message::ScrollToBottom)
            }
            Message::SystemStatus(status) => {
                self.system_status = status;
                Task::none()
            }
            Message::ErrorOccurred(error) => {
                self.error_message = Some(error.clone());
                self.conversation_history
                    .push_back(ConversationEntry::System(
                        format!("Error: {}", error),
                        Instant::now(),
                    ));
                Task::none()
            }
            Message::ClearError => {
                self.error_message = None;
                Task::none()
            }
            Message::ClearConversation => {
                self.conversation_history.clear();
                self.current_user_speech.clear();
                self.current_ai_response.clear();
                Task::none()
            }
            Message::ScrollToBottom => {
                // This would trigger a scroll in the view
                Task::none()
            }
            Message::Tick => {
                self.fps_counter += 1;
                if self.last_update.elapsed() >= Duration::from_secs(1) {
                    self.last_update = Instant::now();
                    self.fps_counter = 0;
                }

                // Continue tick loop and check for updates
                Task::batch([
                    Task::perform(
                        async {
                            tokio::time::sleep(Duration::from_millis(16)).await;
                        },
                        |_| Message::Tick,
                    ),
                    Task::perform(async {}, |_| Message::CheckForUpdates),
                ])
            }
            Message::CheckForUpdates => {
                // Process pending messages from voice system
                let mut pending_messages = Vec::new();

                if let Some(receiver) = &mut self.ui_receiver {
                    // Collect messages without borrowing self
                    for _ in 0..10 {
                        match receiver.try_recv() {
                            Ok(message) => {
                                pending_messages.push(message);
                            }
                            Err(_) => break,
                        }
                    }
                }

                // Process collected messages
                if !pending_messages.is_empty() {
                    let tasks: Vec<_> = pending_messages
                        .into_iter()
                        .map(|msg| self.update(msg))
                        .collect();
                    return Task::batch(tasks);
                }

                Task::none()
            }
        }
    }

    pub fn view(&self) -> Element<Message> {
        let mut main_column = column![];

        // Header
        main_column = main_column.push(
            container(
                row![
                    text("🤖 LiquidOS - AI Voice Assistant")
                        .size(28)
                        .color(Color::from_rgb(0.9, 0.95, 1.0)),
                    text(format!(" - {}", self.system_status))
                        .size(16)
                        .color(Color::from_rgb(0.6, 0.7, 0.8))
                ]
                .spacing(10)
                .align_y(iced::Alignment::Center),
            )
            .width(Length::Fill)
            .padding(20)
            .style(|_theme| container::Style {
                background: Some(iced::Background::Color(Color::from_rgb(0.1, 0.15, 0.25))),
                ..Default::default()
            }),
        );

        // Error display
        if let Some(error) = &self.error_message {
            main_column = main_column.push(
                container(
                    text(format!("⚠️ {}", error))
                        .size(14)
                        .color(Color::from_rgb(1.0, 0.5, 0.5)),
                )
                .width(Length::Fill)
                .padding(10)
                .style(|_theme| container::Style {
                    background: Some(iced::Background::Color(Color::from_rgb(0.3, 0.1, 0.1))),
                    ..Default::default()
                }),
            );
        }

        // Audio level indicator
        let audio_level_bar = Self::create_visual_level_bar(self.audio_level);
        main_column = main_column.push(
            container(
                column![
                    text("Audio Level")
                        .size(12)
                        .color(Color::from_rgb(0.6, 0.6, 0.6)),
                    container(audio_level_bar)
                        .width(Length::Fill)
                        .height(Length::Fixed(8.0))
                        .style(|_theme| container::Style {
                            background: Some(iced::Background::Color(Color::from_rgb(
                                0.1, 0.1, 0.1
                            ))),
                            border: iced::Border {
                                color: Color::from_rgb(0.3, 0.3, 0.3),
                                width: 1.0,
                                radius: 4.0.into(),
                            },
                            ..Default::default()
                        }),
                ]
                .spacing(5),
            )
            .width(Length::Fill)
            .padding([10, 20]),
        );

        // Conversation history
        let conversation_scroll = scrollable(Self::build_conversation_view(
            &self.conversation_history,
            &self.current_user_speech,
            &self.current_ai_response,
            self.is_recording,
            self.is_ai_speaking,
        ))
        .height(Length::Fill)
        .id(scrollable::Id::new("conversation"));

        main_column = main_column.push(
            container(conversation_scroll)
                .width(Length::Fill)
                .height(Length::Fill)
                .padding(20)
                .style(|_theme| container::Style {
                    background: Some(iced::Background::Color(Color::from_rgb(0.05, 0.05, 0.1))),
                    ..Default::default()
                }),
        );

        // Controls
        main_column = main_column.push(
            container(
                row![
                    button(text("Clear").size(14))
                        .on_press(Message::ClearConversation)
                        .style(|theme, status| {
                            let appearance = button::primary(theme, status);
                            button::Style {
                                background: Some(iced::Background::Color(Color::from_rgb(
                                    0.2, 0.3, 0.4,
                                ))),
                                ..appearance
                            }
                        }),
                    text(if self.is_recording {
                        "🔴 Recording"
                    } else {
                        "⭕ Ready"
                    })
                    .size(14)
                    .color(if self.is_recording {
                        Color::from_rgb(1.0, 0.3, 0.3)
                    } else {
                        Color::from_rgb(0.5, 0.5, 0.5)
                    }),
                ]
                .spacing(20)
                .align_y(iced::Alignment::Center),
            )
            .width(Length::Fill)
            .padding(15)
            .style(|_theme| container::Style {
                background: Some(iced::Background::Color(Color::from_rgb(0.08, 0.08, 0.15))),
                ..Default::default()
            }),
        );

        container(main_column)
            .width(Length::Fill)
            .height(Length::Fill)
            .style(|_theme| container::Style {
                background: Some(iced::Background::Color(Color::from_rgb(0.02, 0.02, 0.08))),
                ..Default::default()
            })
            .into()
    }

    fn create_visual_level_bar(level: f32) -> Element<'static, Message> {
        let width_percentage = (level * 100.0).min(100.0);

        container(
            container("")
                .width(Length::FillPortion(
                    (width_percentage * 10.0).max(1.0) as u16
                ))
                .height(Length::Fill)
                .style(move |_theme| {
                    let green_color = Color::from_rgb(0.2 + level * 0.3, 0.8 + level * 0.2, 0.3);
                    container::Style {
                        background: Some(iced::Background::Color(green_color)),
                        border: iced::Border {
                            radius: 2.0.into(),
                            ..Default::default()
                        },
                        ..Default::default()
                    }
                }),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
    }

    fn build_conversation_view(
        history: &VecDeque<ConversationEntry>,
        current_user: &str,
        current_ai: &str,
        is_recording: bool,
        is_ai_speaking: bool,
    ) -> Element<'static, Message> {
        let mut col = column![].spacing(15);

        // Add historical entries
        for entry in history {
            col = col.push(Self::create_conversation_bubble(entry));
        }

        // Add current user speech if recording
        if is_recording && !current_user.is_empty() {
            col = col.push(Self::create_message_bubble(
                current_user.to_string(),
                Color::from_rgb(0.3, 0.5, 0.8),
                true,
                true,
            ));
        }

        // Add current AI response if speaking
        if is_ai_speaking && !current_ai.is_empty() {
            col = col.push(Self::create_message_bubble(
                current_ai.to_string(),
                Color::from_rgb(0.5, 0.3, 0.8),
                false,
                true,
            ));
        }

        col.into()
    }

    fn create_conversation_bubble(entry: &ConversationEntry) -> Element<'static, Message> {
        match entry {
            ConversationEntry::User(msg, _) => Self::create_message_bubble(
                msg.clone(),
                Color::from_rgb(0.2, 0.4, 0.7),
                true,
                false,
            ),
            ConversationEntry::Assistant(msg, _) => Self::create_message_bubble(
                msg.clone(),
                Color::from_rgb(0.4, 0.2, 0.7),
                false,
                false,
            ),
            ConversationEntry::System(msg, _) => container(
                text(msg.clone())
                    .size(12)
                    .color(Color::from_rgb(0.6, 0.6, 0.6)),
            )
            .width(Length::Fill)
            .padding(10)
            .style(|_theme| container::Style {
                background: Some(iced::Background::Color(Color::from_rgb(0.15, 0.15, 0.2))),
                border: iced::Border {
                    color: Color::from_rgb(0.3, 0.3, 0.4),
                    width: 1.0,
                    radius: 8.0.into(),
                },
                ..Default::default()
            })
            .into(),
        }
    }

    fn create_message_bubble(
        message: String,
        color: Color,
        is_user: bool,
        is_active: bool,
    ) -> Element<'static, Message> {
        let bubble = container(
            column![
                text(if is_user { "You" } else { "Bella" })
                    .size(11)
                    .color(Color::from_rgb(0.7, 0.7, 0.7)),
                text(message)
                    .size(14)
                    .color(Color::from_rgb(0.95, 0.95, 0.95))
            ]
            .spacing(5),
        )
        .padding(12)
        .max_width(600)
        .style(move |_theme| container::Style {
            background: Some(iced::Background::Color(color)),
            border: iced::Border {
                color: if is_active {
                    Color::from_rgb(0.8, 0.8, 0.9)
                } else {
                    Color::TRANSPARENT
                },
                width: if is_active { 2.0 } else { 0.0 },
                radius: 12.0.into(),
            },
            ..Default::default()
        });

        if is_user {
            row![container("").width(Length::Fill), bubble]
        } else {
            row![bubble, container("").width(Length::Fill)]
        }
        .into()
    }

    // Helper function to send messages to UI
    fn send_ui_message(message: Message) {
        if let Ok(sender_guard) = UI_SENDER.lock() {
            if let Some(sender) = sender_guard.as_ref() {
                let _ = sender.send(message);
            }
        }
    }

    async fn setup_voice_system() -> Result<()> {
        println!("🚀 Setting up voice system with UI integration...");

        // Initialize paths
        let model_path = "checkpoints/kokoro-v1.0.onnx";
        let data_path = "examples/voice_agent/audio/voices-v1.0.bin";
        let stt_model = "./examples/voice_agent/models/ggml-base.en.bin";

        // Send status update
        Self::send_ui_message(Message::SystemStatus("Initializing TTS...".to_string()));

        println!("🎤 Initializing TTS...");
        let _tts = TTSKoko::new(model_path, data_path).await;
        println!("✅ TTS initialized successfully");

        // Send status update
        Self::send_ui_message(Message::SystemStatus("Initializing STT...".to_string()));

        println!("🎧 Initializing STT...");
        let mut _stt_processor = STTProcessor::new(PathBuf::from(stt_model)).await?;
        println!("✅ STT initialized successfully");

        // Send status update
        Self::send_ui_message(Message::SystemStatus("Configuring LLM...".to_string()));

        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY not set"))?;

        // Initialize and configure the LLM client
        let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
            .api_key(api_key)
            .model("gpt-4o")
            .max_tokens(512)
            .temperature(0.2)
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
        let agent_topic = Topic::<AgentTask>::new("agent_topic");

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

        // Set up event handling with UI updates
        let receiver = environment.take_event_receiver(None).await?;

        // Create recording control for synchronization
        let recording_control = Arc::new(tokio::sync::RwLock::new(false));

        // Create actors with proper configuration
        let tts_config = TTSConfig {
            language: "en-us".to_string(),
            style: "af_sarah.4+af_nicole.6".to_string(),
            speed: 1.3,
            mono: false,
            initial_silence: None,
        };

        let tts_actor = TTSActor::new("TTS_Actor", runtime.clone(), tts_config)
            .with_recording_control(recording_control.clone());
        let stt_actor = STTActor::new("STT_Actor", runtime.clone());

        // Initialize the actors with their models
        tts_actor.initialize_tts(model_path, data_path).await?;
        stt_actor.initialize_stt(PathBuf::from(stt_model)).await?;

        let (tts_actor_ref, _) = Actor::spawn(None, tts_actor, TTSActorArgs {}).await?;
        let (stt_actor_ref, _) = Actor::spawn(None, stt_actor, STTActorArgs {}).await?;

        // Subscribe actors to topics
        runtime.subscribe(&tts_topic, tts_actor_ref.clone()).await?;
        runtime.subscribe(&stt_topic, stt_actor_ref.clone()).await?;

        // Start event handling that sends UI updates
        tokio::spawn(Self::handle_streaming_events_with_ui(
            receiver,
            runtime.clone(),
            tts_topic.clone(),
        ));

        // Start STT monitoring for user speech
        tokio::spawn(Self::monitor_stt_updates(
            stt_topic.clone(),
            runtime.clone(),
        ));

        tokio::spawn(async move {
            let _ = environment.run();
        });

        // Start the original realtime voice processing
        tokio::spawn({
            let runtime_clone = runtime.clone();
            let stt_topic_clone = stt_topic.clone();
            let tts_topic_clone = tts_topic.clone();
            let recording_control_clone = recording_control.clone();
            async move {
                if let Err(e) = run_realtime_mode_actor_based(
                    runtime_clone,
                    stt_topic_clone,
                    tts_topic_clone,
                    6,
                    Some(recording_control_clone),
                )
                .await
                {
                    Self::send_ui_message(Message::ErrorOccurred(format!(
                        "Voice processing error: {}",
                        e
                    )));
                }
            }
        });

        // Start audio level monitoring
        tokio::spawn(async move {
            Self::monitor_audio_levels().await;
        });

        // Enable recording
        *recording_control.write().await = true;

        println!("✅ Voice system setup complete");
        Ok(())
    }

    async fn monitor_stt_updates(
        topic: Topic<AudioBufferMessage>,
        runtime: Arc<SingleThreadedRuntime>,
    ) {
        // This would monitor STT updates and send appropriate UI messages
        // Implementation depends on your STT actor's output format
    }

    async fn handle_streaming_events_with_ui(
        mut event_stream: ReceiverStream<Event>,
        runtime: Arc<SingleThreadedRuntime>,
        topic: Topic<SimpleMessage>,
    ) {
        let buffer = Arc::new(std::sync::Mutex::new(StreamingBuffer::new(500)));
        let runtime_clone = runtime.clone();
        let topic_clone = topic.clone();
        let buffer_clone = buffer.clone();

        let mut full_response = String::new();
        let mut is_streaming = false;

        // Start periodic flush task
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
                        let _ = runtime_clone
                            .publish(
                                &topic_clone,
                                SimpleMessage {
                                    content: content.clone(),
                                },
                            )
                            .await;
                    }
                }
            }
        });

        // Main event processing loop
        while let Some(event) = StreamExt::next(&mut event_stream).await {
            match event {
                Event::TaskComplete { result, .. } => {
                    // Flush any remaining buffer content
                    let remaining_content = {
                        let mut buffer_guard = buffer.lock().unwrap();
                        if !buffer_guard.is_empty() {
                            buffer_guard.flush()
                        } else {
                            String::new()
                        }
                    };

                    if !remaining_content.is_empty() {
                        full_response.push_str(&remaining_content);
                        let _ = runtime
                            .publish(
                                &topic,
                                SimpleMessage {
                                    content: remaining_content,
                                },
                            )
                            .await;
                    }

                    // Send final AI response
                    if !full_response.is_empty() {
                        Self::send_ui_message(Message::AISpeechEnd(full_response.clone()));
                        full_response.clear();
                    }

                    is_streaming = false;

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
                            Self::send_ui_message(Message::ErrorOccurred(format!(
                                "Task failed: {}",
                                error
                            )));
                        }
                        TaskResult::Aborted => {
                            Self::send_ui_message(Message::SystemStatus(
                                "Task aborted".to_string(),
                            ));
                        }
                    }
                }
                Event::StreamChunk { sub_id: _, chunk } => {
                    let content = chunk.delta.content.unwrap_or_default();

                    if !content.is_empty() {
                        if !is_streaming {
                            is_streaming = true;
                            Self::send_ui_message(Message::AISpeechStart);
                        }

                        full_response.push_str(&content);

                        // Send incremental updates to UI
                        Self::send_ui_message(Message::AISpeechUpdate(full_response.clone()));

                        // Add to buffer for TTS
                        {
                            let mut buffer_guard = buffer.lock().unwrap();
                            buffer_guard.add_token(&content);
                        }

                        // Check if we should flush
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
                                let _ = runtime
                                    .publish(
                                        &topic,
                                        SimpleMessage {
                                            content: chunk_content,
                                        },
                                    )
                                    .await;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    async fn monitor_audio_levels() {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        if let Ok(host) = std::panic::catch_unwind(cpal::default_host) {
            if let Some(input_device) = host.default_input_device() {
                if let Ok(config) = input_device.default_input_config() {
                    let channels = config.channels() as usize;

                    // Track recording state
                    let mut was_speaking = false;
                    let mut silence_counter = 0;
                    let speech_threshold = 0.02;
                    let silence_threshold = 30; // frames of silence before stopping

                    if let Ok(stream) = input_device.build_input_stream(
                        &config.into(),
                        move |data: &[f32], _: &cpal::InputCallbackInfo| {
                            // Calculate RMS level
                            let mut sum_squares = 0.0f32;
                            let mut sample_count = 0;

                            for chunk in data.chunks(channels) {
                                if let Some(&sample) = chunk.first() {
                                    sum_squares += sample * sample;
                                    sample_count += 1;
                                }
                            }

                            if sample_count > 0 {
                                let rms = (sum_squares / sample_count as f32).sqrt();
                                let normalized_level = (rms * 3.0).min(1.0);

                                // Send audio level update
                                Self::send_ui_message(Message::AudioLevelUpdate(normalized_level));

                                // Detect speech start/stop
                                if rms > speech_threshold {
                                    if !was_speaking {
                                        was_speaking = true;
                                        silence_counter = 0;
                                        Self::send_ui_message(Message::UserSpeechStart);
                                        Self::send_ui_message(Message::RecordingStateChanged(true));
                                    }
                                } else if was_speaking {
                                    silence_counter += 1;
                                    if silence_counter > silence_threshold {
                                        was_speaking = false;
                                        silence_counter = 0;
                                        Self::send_ui_message(Message::RecordingStateChanged(
                                            false,
                                        ));
                                    }
                                }
                            }
                        },
                        move |err| {
                            Self::send_ui_message(Message::ErrorOccurred(format!(
                                "Audio error: {}",
                                err
                            )));
                        },
                        None,
                    ) {
                        let _ = stream.play();

                        // Keep stream alive
                        loop {
                            tokio::time::sleep(Duration::from_secs(1)).await;
                        }
                    }
                }
            }
        }
    }
}

pub fn run_voice_agent_app() -> iced::Result {
    iced::application(
        "🤖 Bella - AI Voice Assistant",
        VoiceAgentApp::update,
        VoiceAgentApp::view,
    )
    .theme(|_| Theme::Dark)
    .run_with(|| {
        (
            VoiceAgentApp::new(),
            Task::perform(async {}, |_| Message::InitializeVoiceSystem),
        )
    })
}
