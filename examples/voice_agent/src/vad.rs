use anyhow::Result;
use std::sync::Mutex;
use voice_activity_detector::VoiceActivityDetector;

/// Voice Activity Detection processor using Silero VAD
pub struct VADProcessor {
    detector: Mutex<VoiceActivityDetector>,
    sample_rate: usize,
    chunk_size_samples: usize,
}

impl VADProcessor {
    /// Create a new VAD processor
    ///
    /// # Arguments
    /// * `sample_rate` - Audio sample rate (8000 or 16000 Hz supported)
    /// * `chunk_size_samples` - Chunk size in samples (default based on sample rate)
    pub fn new(sample_rate: usize, chunk_size_samples: Option<usize>) -> Result<Self> {
        // Validate sample rate
        if sample_rate != 8000 && sample_rate != 16000 {
            return Err(anyhow::anyhow!(
                "Silero VAD only supports 8000 or 16000 Hz sample rates, got: {}",
                sample_rate
            ));
        }

        // Determine chunk size based on sample rate if not provided
        let chunk_size_samples = chunk_size_samples.unwrap_or_else(|| {
            match sample_rate {
                8000 => 512,  // 64ms chunks for 8kHz
                16000 => 512, // 32ms chunks for 16kHz
                _ => 512,
            }
        });

        let detector = VoiceActivityDetector::builder()
            .sample_rate(sample_rate as i64)
            .chunk_size(chunk_size_samples)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to create VAD: {}", e))?;

        Ok(Self {
            detector: Mutex::new(detector),
            sample_rate,
            chunk_size_samples,
        })
    }

    /// Process audio chunk and return speech probability
    /// Returns probability between 0.0 and 1.0
    pub fn process_chunk(&self, audio_chunk: &[f32]) -> Result<f32> {
        if audio_chunk.len() != self.chunk_size_samples {
            return Err(anyhow::anyhow!(
                "Audio chunk size mismatch: expected {}, got {}",
                self.chunk_size_samples,
                audio_chunk.len()
            ));
        }

        let mut detector = self.detector.lock().unwrap();
        let probability = detector.predict(audio_chunk.to_vec());

        Ok(probability)
    }

    /// Get the required chunk size in samples
    pub fn chunk_size_samples(&self) -> usize {
        self.chunk_size_samples
    }

    /// Get the sample rate
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }
}

/// State machine for VAD-based audio segmentation
#[derive(Debug, Clone, PartialEq)]
pub enum VADState {
    /// Waiting for speech to begin
    WaitingForSpeech,
    /// Currently detecting speech
    InSpeech {
        /// Accumulated audio samples during speech
        buffer: Vec<f32>,
        /// Start time of speech segment (in chunk counts)
        start_time: usize,
    },
}

/// VAD-based audio segmenter that collects speech segments
pub struct VADSegmenter {
    vad: VADProcessor,
    state: VADState,
    audio_buffer: Vec<f32>,
    min_speech_duration_ms: usize,
    max_duration_ms: usize,
    chunk_count: usize,
    speech_threshold: f32,
    silence_threshold: f32,
    was_speaking: bool,         // Track previous speech state
    silence_timeout_ms: usize,  // Maximum silence duration before stopping
    silence_chunk_count: usize, // Count chunks of continuous silence
}

impl VADSegmenter {
    pub fn new(
        sample_rate: usize,
        min_speech_duration_ms: usize,
        max_duration_ms: usize,
        speech_threshold: f32,
        silence_threshold: f32,
        chunk_size_samples: Option<usize>,
    ) -> Result<Self> {
        let vad = VADProcessor::new(sample_rate, chunk_size_samples)?;

        Ok(Self {
            vad,
            state: VADState::WaitingForSpeech,
            audio_buffer: Vec::new(),
            min_speech_duration_ms,
            max_duration_ms,
            chunk_count: 0,
            speech_threshold,
            silence_threshold,
            was_speaking: false,
            silence_timeout_ms: 2000, // 2 seconds of silence timeout
            silence_chunk_count: 0,
        })
    }

    /// Set the silence timeout duration in milliseconds
    pub fn set_silence_timeout(&mut self, timeout_ms: usize) {
        self.silence_timeout_ms = timeout_ms;
    }

    /// Process audio chunk and return completed speech segment if available
    ///
    /// Returns Some(audio_buffer) when a complete speech segment is detected,
    /// None otherwise
    pub fn process_chunk(&mut self, audio_chunk: &[f32]) -> Result<Option<Vec<f32>>> {
        // Ensure chunk is the right size
        if audio_chunk.len() != self.vad.chunk_size_samples() {
            return Err(anyhow::anyhow!(
                "Chunk size mismatch: expected {}, got {}",
                self.vad.chunk_size_samples(),
                audio_chunk.len()
            ));
        }

        // Process through VAD first
        let speech_probability = self.vad.process_chunk(audio_chunk)?;
        let is_speaking = speech_probability > self.speech_threshold;
        let chunk_duration_ms = (self.vad.chunk_size_samples() * 1000) / self.vad.sample_rate();
        let current_time_ms = self.chunk_count * chunk_duration_ms;
        self.chunk_count += 1;

        // Detect transitions
        let speech_started = !self.was_speaking && is_speaking;
        let speech_ended =
            self.was_speaking && !is_speaking && speech_probability < self.silence_threshold;

        // Track silence duration
        if !is_speaking {
            self.silence_chunk_count += 1;
        } else {
            self.silence_chunk_count = 0; // Reset silence counter when speech is detected
        }

        let silence_duration_ms = self.silence_chunk_count * chunk_duration_ms;
        self.was_speaking = is_speaking;

        match &mut self.state {
            VADState::WaitingForSpeech => {
                if speech_started {
                    println!(
                        "🎤 Speech started at {}ms (prob: {:.3})",
                        current_time_ms, speech_probability
                    );
                    // Initialize fresh buffer with the current chunk (first speech chunk)
                    self.audio_buffer.clear();
                    self.audio_buffer.extend_from_slice(audio_chunk);
                    println!(
                        "🆕 Started new speech buffer: {} samples",
                        self.audio_buffer.len()
                    );
                    self.state = VADState::InSpeech {
                        buffer: Vec::new(), // Don't store unused buffer here
                        start_time: self.chunk_count - 1,
                    };
                } else {
                    // Clear buffer when not speaking to prevent accumulation
                    self.audio_buffer.clear();
                }
                Ok(None)
            }
            VADState::InSpeech { buffer, start_time } => {
                // Add current chunk to buffer (we're already in speech mode)
                let old_len = self.audio_buffer.len();
                self.audio_buffer.extend_from_slice(audio_chunk);
                if self.chunk_count % 10 == 0 {
                    // Debug every 10 chunks to avoid spam
                    println!(
                        "🔄 Buffer growing: {} -> {} samples",
                        old_len,
                        self.audio_buffer.len()
                    );
                }

                let speech_duration_ms = (self.chunk_count - *start_time) * chunk_duration_ms;

                if speech_ended {
                    println!(
                        "🔇 Speech ended at {}ms (duration: {}ms, prob: {:.3})",
                        current_time_ms, speech_duration_ms, speech_probability
                    );

                    // Check if speech was long enough
                    if speech_duration_ms >= self.min_speech_duration_ms {
                        let completed_buffer = self.audio_buffer.clone();
                        println!(
                            "📤 Completing segment: {} samples, {:.2}s audio",
                            completed_buffer.len(),
                            completed_buffer.len() as f32 / self.vad.sample_rate() as f32
                        );
                        // Completely reset state for clean separation between segments
                        self.state = VADState::WaitingForSpeech;
                        self.audio_buffer.clear();
                        self.silence_chunk_count = 0;
                        self.was_speaking = false; // Reset speaking state
                        return Ok(Some(completed_buffer));
                    } else {
                        println!(
                            "⚠️ Speech too short ({}ms < {}ms), discarding",
                            speech_duration_ms, self.min_speech_duration_ms
                        );
                        // Completely reset state
                        self.state = VADState::WaitingForSpeech;
                        self.audio_buffer.clear();
                        self.silence_chunk_count = 0;
                        self.was_speaking = false; // Reset speaking state
                    }
                } else if speech_duration_ms >= self.max_duration_ms {
                    println!(
                        "⚠️ Max speech duration reached ({}ms), processing segment",
                        speech_duration_ms
                    );
                    let completed_buffer = self.audio_buffer.clone();
                    // Completely reset state for clean separation between segments
                    self.state = VADState::WaitingForSpeech;
                    self.audio_buffer.clear();
                    self.silence_chunk_count = 0;
                    self.was_speaking = false; // Reset speaking state
                    return Ok(Some(completed_buffer));
                } else if silence_duration_ms >= self.silence_timeout_ms
                    && speech_duration_ms >= self.min_speech_duration_ms
                {
                    // Extended silence detected - auto-complete the segment
                    println!("🤫 Extended silence detected ({}ms), completing speech segment (duration: {}ms)", 
                        silence_duration_ms, speech_duration_ms);
                    let completed_buffer = self.audio_buffer.clone();
                    // Completely reset state for clean separation between segments
                    self.state = VADState::WaitingForSpeech;
                    self.audio_buffer.clear();
                    self.silence_chunk_count = 0;
                    self.was_speaking = false; // Reset speaking state
                    return Ok(Some(completed_buffer));
                }
                Ok(None)
            }
        }
    }

    /// Force completion of current segment (if any)
    pub fn force_complete(&mut self) -> Result<Option<Vec<f32>>> {
        match &self.state {
            VADState::InSpeech { start_time, .. } => {
                let chunk_duration_ms =
                    (self.vad.chunk_size_samples() * 1000) / self.vad.sample_rate();
                let speech_duration_ms = (self.chunk_count - *start_time) * chunk_duration_ms;

                if speech_duration_ms >= self.min_speech_duration_ms
                    && !self.audio_buffer.is_empty()
                {
                    println!(
                        "🔚 Forcing completion of speech segment (duration: {}ms)",
                        speech_duration_ms
                    );
                    let completed_buffer = self.audio_buffer.clone();
                    self.state = VADState::WaitingForSpeech;
                    self.audio_buffer.clear();
                    Ok(Some(completed_buffer))
                } else {
                    self.state = VADState::WaitingForSpeech;
                    self.audio_buffer.clear();
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    /// Reset the segmenter state
    pub fn reset(&mut self) -> Result<()> {
        self.state = VADState::WaitingForSpeech;
        self.audio_buffer.clear();
        self.chunk_count = 0;
        self.was_speaking = false;
        self.silence_chunk_count = 0;
        Ok(())
    }

    /// Get the VAD chunk size in samples
    pub fn chunk_size_samples(&self) -> usize {
        self.vad.chunk_size_samples()
    }

    /// Get the sample rate
    pub fn sample_rate(&self) -> usize {
        self.vad.sample_rate()
    }
}
