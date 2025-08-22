pub mod model;

use crate::stt::model::{Decoder, Model, WhichModel};
use anyhow::Result;
use candle::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, audio, Config};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hf_hub::{api::sync::Api, Repo, RepoType};
use rubato::Resampler;
pub mod actor;
use std::collections::VecDeque;
use std::sync::mpsc;
use tokenizers::Tokenizer;

pub struct STTProcessor {
    decoder: Decoder,
    config: Config,
    mel_filters: Vec<f32>,
    device: Device,
}

impl STTProcessor {
    pub async fn new(model: WhichModel, _language: Option<String>) -> Result<Self> {
        let device = Device::Cpu;
        let (default_model, default_revision) = model.model_and_revision();
        let model_id = default_model.to_string();
        let revision = default_revision.to_string();

        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let model_path = repo.get("model.safetensors")?;

        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], m::DTYPE, &device)? };
        let model = Model::Normal(m::model::Whisper::load(&vb, config.clone())?);

        let decoder = Decoder::new(
            model,
            tokenizer,
            299792458, // seed
            &device,
            None, // language_token will be set later
            Some(model::Task::Transcribe),
            false, // timestamps
            false, // verbose
        )?;

        let mel_bytes = match config.num_mel_bins {
            80 => include_bytes!("./melfilters.bytes").as_slice(),
            128 => include_bytes!("./melfilters128.bytes").as_slice(),
            nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
        };
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
            mel_bytes,
            &mut mel_filters,
        );

        Ok(Self {
            decoder,
            config,
            mel_filters,
            device,
        })
    }

    pub fn transcribe_audio(&mut self, pcm_data: &[f32]) -> Result<String> {
        println!("🎵 Processing {} audio samples for STT", pcm_data.len());

        // Limit audio length for faster processing (max 30 seconds at 16kHz = 480,000 samples)
        let max_samples_at_16khz = 480_000;
        let target_sample_rate = 16000;

        // Quick RMS check
        let rms = {
            let sum_squares: f32 = pcm_data.iter().map(|&x| x * x).sum();
            (sum_squares / pcm_data.len() as f32).sqrt()
        };

        if rms < 0.001 {
            println!("⚠️ Audio level too low, returning empty");
            return Ok(String::new());
        }

        // Estimate input sample rate and resample
        let estimated_input_rate = if pcm_data.len() > 100000 {
            44100
        } else {
            16000
        };

        let mut resampled_data = if estimated_input_rate != target_sample_rate {
            println!(
                "🔄 Resampling from {}Hz to {}Hz",
                estimated_input_rate, target_sample_rate
            );
            let resample_ratio = target_sample_rate as f64 / estimated_input_rate as f64;

            match rubato::FastFixedIn::new(
                resample_ratio,
                10.,
                rubato::PolynomialDegree::Linear,
                512,
                1,
            ) {
                Ok(mut resampler) => {
                    let mut resampled = Vec::new();
                    for chunk in pcm_data.chunks(512) {
                        if chunk.len() == 512 {
                            if let Ok(output) = resampler.process(&[chunk], None) {
                                resampled.extend_from_slice(&output[0]);
                            }
                        }
                    }
                    resampled
                }
                Err(_) => pcm_data.to_vec(),
            }
        } else {
            pcm_data.to_vec()
        };

        // Limit length for performance
        if resampled_data.len() > max_samples_at_16khz {
            println!(
                "⚡ Truncating audio from {} to {} samples for faster processing",
                resampled_data.len(),
                max_samples_at_16khz
            );
            resampled_data.truncate(max_samples_at_16khz);
        }

        let mel = audio::pcm_to_mel(&self.config, &resampled_data, &self.mel_filters);
        let mel_len = mel.len();
        let mel = candle::Tensor::from_vec(
            mel,
            (
                1,
                self.config.num_mel_bins,
                mel_len / self.config.num_mel_bins,
            ),
            &self.device,
        )?;

        println!("🧠 Running Whisper decoder...");
        let start_time = std::time::Instant::now();

        let segments = self.decoder.run(&mel, None)?;
        let decode_time = start_time.elapsed();

        println!(
            "📊 Decoded {} segments in {:.2}s",
            segments.len(),
            decode_time.as_secs_f32()
        );

        let texts: Vec<String> = segments.into_iter().map(|s| s.dr.text).collect();

        let text = texts.join(" ").trim().to_string();
        println!(
            "🎯 Final result: '{}' (total time: {:.2}s)",
            text,
            start_time.elapsed().as_secs_f32()
        );

        Ok(text)
    }

    pub fn process_file(&mut self, file_path: &str) -> Result<String> {
        let mut reader = hound::WavReader::open(file_path)?;
        let samples: Result<Vec<f32>, _> =
            if reader.spec().sample_format == hound::SampleFormat::Float {
                reader.samples::<f32>().collect()
            } else {
                reader
                    .samples::<i16>()
                    .map(|s| s.map(|s| s as f32 / i16::MAX as f32))
                    .collect()
            };
        let samples = samples?;

        // Resample to 16kHz if needed
        let target_sample_rate = 16000;
        let source_sample_rate = reader.spec().sample_rate;

        let resampled_samples = if source_sample_rate != target_sample_rate {
            let resample_ratio = target_sample_rate as f64 / source_sample_rate as f64;
            let mut resampler = rubato::FastFixedIn::new(
                resample_ratio,
                10.,
                rubato::PolynomialDegree::Septic,
                1024,
                1,
            )?;

            let mut resampled = Vec::new();
            let chunk_size = 1024;
            for chunk in samples.chunks(chunk_size) {
                if chunk.len() == chunk_size {
                    let output = resampler.process(&[chunk], None)?;
                    resampled.extend_from_slice(&output[0]);
                }
            }
            resampled
        } else {
            samples
        };

        self.transcribe_audio(&resampled_samples)
    }
}

pub struct VoiceActivityDetector {
    // Energy-based detection
    energy_threshold: f32,
    energy_history: VecDeque<f32>,
    energy_history_size: usize,

    // Spectral features
    sample_rate: u32,
    frame_size: usize,

    // State tracking
    voice_frames: usize,
    silence_frames: usize,
    min_voice_frames: usize,
    min_silence_frames: usize,

    // Adaptive thresholding
    background_noise_level: f32,
    noise_samples: usize,
    adaptation_rate: f32,

    // Zero crossing rate
    prev_sample: f32,
}

impl VoiceActivityDetector {
    pub fn new(sample_rate: u32, min_voice_duration_ms: u32, min_silence_duration_ms: u32) -> Self {
        let frame_size = (sample_rate as f32 * 0.025) as usize; // 25ms frames
        let min_voice_frames = (min_voice_duration_ms * sample_rate / 1000) / frame_size as u32;
        let min_silence_frames = (min_silence_duration_ms * sample_rate / 1000) / frame_size as u32;

        Self {
            energy_threshold: 0.01, // Will be adaptive
            energy_history: VecDeque::with_capacity(50),
            energy_history_size: 50, // ~1.25 seconds of history
            sample_rate,
            frame_size,
            voice_frames: 0,
            silence_frames: 0,
            min_voice_frames: min_voice_frames as usize,
            min_silence_frames: min_silence_frames as usize,
            background_noise_level: 0.005,
            noise_samples: 0,
            adaptation_rate: 0.1,
            prev_sample: 0.0,
        }
    }

    pub fn process_audio(&mut self, audio_chunk: &[f32]) -> VadResult {
        // Process in frames
        let mut voice_detected = false;
        let mut frame_results = Vec::new();

        for frame in audio_chunk.chunks(self.frame_size) {
            if frame.len() < self.frame_size / 2 {
                continue; // Skip incomplete frames
            }

            let is_voice = self.analyze_frame(frame);
            frame_results.push(is_voice);

            if is_voice {
                voice_detected = true;
            }
        }

        // Update state counters
        if voice_detected {
            self.voice_frames += frame_results.len();
            self.silence_frames = 0;
        } else {
            self.silence_frames += frame_results.len();
            if self.voice_frames == 0 {
                // Still in initial silence, update background noise
                self.update_background_noise(audio_chunk);
            } else {
                self.voice_frames = 0; // Reset voice counter
            }
        }

        // Determine overall state
        if self.voice_frames >= self.min_voice_frames {
            VadResult::Voice
        } else if self.silence_frames >= self.min_silence_frames && self.voice_frames == 0 {
            VadResult::Silence
        } else {
            VadResult::Transition
        }
    }

    fn analyze_frame(&mut self, frame: &[f32]) -> bool {
        // 1. Energy-based detection
        let energy = self.compute_energy(frame);

        // 2. Zero crossing rate
        let zcr = self.compute_zero_crossing_rate(frame);

        // 3. Spectral centroid (simple approximation)
        let spectral_activity = self.compute_spectral_activity(frame);

        // Update energy history
        self.energy_history.push_back(energy);
        if self.energy_history.len() > self.energy_history_size {
            self.energy_history.pop_front();
        }

        // Adaptive threshold based on recent energy history
        let adaptive_threshold = self.compute_adaptive_threshold();

        // Voice detection logic - made more sensitive
        let energy_check = energy > adaptive_threshold;
        let zcr_check = zcr > 0.05 && zcr < 0.9; // More relaxed ZCR range
        let spectral_check = spectral_activity > self.background_noise_level * 1.5; // Lower threshold

        // More lenient: at least 1 of 3 features, or energy is significantly high
        let voice_indicators = [energy_check, zcr_check, spectral_check];
        let voice_count = voice_indicators.iter().filter(|&&x| x).count();

        // Accept if at least 1 feature is positive, or if energy is very high
        voice_count >= 1 || energy > (adaptive_threshold * 2.0)
    }

    fn compute_energy(&self, frame: &[f32]) -> f32 {
        let sum_squares: f32 = frame.iter().map(|&x| x * x).sum();
        (sum_squares / frame.len() as f32).sqrt()
    }

    fn compute_zero_crossing_rate(&mut self, frame: &[f32]) -> f32 {
        let mut crossings = 0;
        let mut prev = self.prev_sample;

        for &sample in frame {
            if (prev >= 0.0 && sample < 0.0) || (prev < 0.0 && sample >= 0.0) {
                crossings += 1;
            }
            prev = sample;
        }

        if !frame.is_empty() {
            self.prev_sample = frame[frame.len() - 1];
        }

        crossings as f32 / frame.len() as f32
    }

    fn compute_spectral_activity(&self, frame: &[f32]) -> f32 {
        // Simple high-frequency energy approximation
        // More sophisticated would use FFT, but this is computationally efficient
        let mut high_freq_energy = 0.0;
        for i in 1..frame.len() {
            let diff = frame[i] - frame[i - 1];
            high_freq_energy += diff * diff;
        }
        (high_freq_energy / frame.len() as f32).sqrt()
    }

    fn compute_adaptive_threshold(&self) -> f32 {
        if self.energy_history.is_empty() {
            return self.energy_threshold;
        }

        // Use median of recent energy values plus a margin
        let mut sorted_energy: Vec<f32> = self.energy_history.iter().cloned().collect();
        sorted_energy.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = sorted_energy[sorted_energy.len() / 2];
        let noise_floor = self.background_noise_level;

        // More sensitive adaptive threshold
        (median + 0.005).max(noise_floor * 2.0).max(0.003) // Lower minimum threshold
    }

    fn update_background_noise(&mut self, audio_chunk: &[f32]) {
        let current_energy = self.compute_energy(audio_chunk);

        if self.noise_samples == 0 {
            self.background_noise_level = current_energy;
        } else {
            // Exponential moving average
            self.background_noise_level = self.background_noise_level
                * (1.0 - self.adaptation_rate)
                + current_energy * self.adaptation_rate;
        }
        self.noise_samples += 1;
    }

    pub fn reset(&mut self) {
        self.voice_frames = 0;
        self.silence_frames = 0;
        self.prev_sample = 0.0;
    }

    pub fn get_stats(&self) -> VadStats {
        VadStats {
            background_noise_level: self.background_noise_level,
            current_threshold: self.compute_adaptive_threshold(),
            voice_frames: self.voice_frames,
            silence_frames: self.silence_frames,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum VadResult {
    Voice,      // Active speech detected
    Silence,    // Definite silence (ready to process)
    Transition, // Transitioning between voice/silence
}

#[derive(Debug)]
pub struct VadStats {
    pub background_noise_level: f32,
    pub current_threshold: f32,
    pub voice_frames: usize,
    pub silence_frames: usize,
}

// Keep the old SilenceDetector for compatibility
pub struct SilenceDetector {
    vad: VoiceActivityDetector,
}

impl SilenceDetector {
    pub fn new(threshold: f32, min_silence_duration_ms: u32, sample_rate: u32) -> Self {
        Self {
            vad: VoiceActivityDetector::new(sample_rate, 500, min_silence_duration_ms), // 500ms min voice
        }
    }

    pub fn is_silent(&mut self, audio_chunk: &[f32]) -> bool {
        match self.vad.process_audio(audio_chunk) {
            VadResult::Silence => true,
            _ => false,
        }
    }

    pub fn reset(&mut self) {
        self.vad.reset();
    }
}

pub struct AudioPlayback {
    device: cpal::Device,
    config: cpal::StreamConfig,
}

impl AudioPlayback {
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow::anyhow!("No output device available"))?;

        let supported_config = device.default_output_config()?;
        println!(
            "🔈 Output device: {}",
            device.name().unwrap_or("Unknown".to_string())
        );
        println!("🔈 Supported config: {:?}", supported_config);

        // Try to use 24kHz to match TTS, but fall back to supported rate if needed
        let target_sample_rate = 24000;
        let config = cpal::StreamConfig {
            channels: 1, // Use mono for simplicity
            sample_rate: cpal::SampleRate(target_sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        println!(
            "🔈 Using config: channels={}, sample_rate={}Hz",
            config.channels, config.sample_rate.0
        );

        Ok(Self { device, config })
    }

    pub fn play_audio(&self, audio_data: Vec<f32>) -> Result<()> {
        use std::collections::VecDeque;
        use std::sync::{Arc, Mutex};

        if audio_data.is_empty() {
            return Ok(());
        }

        println!(
            "🔊 Playing {} audio samples at {}Hz...",
            audio_data.len(),
            self.config.sample_rate.0
        );

        // Create a shared buffer for the audio data
        let buffer = Arc::new(Mutex::new(VecDeque::from(audio_data.clone())));
        let buffer_clone = buffer.clone();
        let finished = Arc::new(Mutex::new(false));
        let finished_clone = finished.clone();

        // Calculate expected playback duration
        let duration_secs = audio_data.len() as f32 / self.config.sample_rate.0 as f32;
        println!("🕐 Expected playback duration: {:.2}s", duration_secs);

        let stream = self.device.build_output_stream(
            &self.config,
            move |output: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut buf = buffer_clone.lock().unwrap();
                let mut finished_flag = finished_clone.lock().unwrap();

                let mut samples_written = 0;
                for sample in output.iter_mut() {
                    if let Some(audio_sample) = buf.pop_front() {
                        *sample = audio_sample;
                        samples_written += 1;
                    } else {
                        *sample = 0.0;
                    }
                }

                if samples_written == 0 && !*finished_flag {
                    *finished_flag = true;
                }
            },
            move |err| eprintln!("Audio output error: {}", err),
            None,
        )?;

        stream.play()?;

        // Use a time-based approach combined with buffer checking
        let start_time = std::time::Instant::now();
        let timeout_duration = std::time::Duration::from_secs_f32(duration_secs + 2.0); // Extra buffer time

        loop {
            let buf_empty = {
                let buf = buffer.lock().unwrap();
                buf.is_empty()
            };

            let is_finished = {
                let finished_flag = finished.lock().unwrap();
                *finished_flag
            };

            let elapsed = start_time.elapsed();

            if (buf_empty && is_finished) || elapsed > timeout_duration {
                if elapsed > timeout_duration {
                    println!(
                        "⚠️  Audio playback timeout after {:.2}s",
                        elapsed.as_secs_f32()
                    );
                }
                break;
            }

            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        // Give a little extra time for the audio system to finish
        std::thread::sleep(std::time::Duration::from_millis(100));
        println!(
            "✅ Audio playback completed in {:.2}s",
            start_time.elapsed().as_secs_f32()
        );

        Ok(())
    }
}
