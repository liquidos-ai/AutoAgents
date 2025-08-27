use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

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
