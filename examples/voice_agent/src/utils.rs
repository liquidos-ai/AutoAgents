// utils.rs - Utility functions for the voice agent
use anyhow::Result;
use candle::Device;

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else {
        #[cfg(feature = "cuda")]
        {
            Ok(Device::new_cuda(0)?)
        }
        #[cfg(not(feature = "cuda"))]
        {
            #[cfg(feature = "metal")]
            {
                Ok(Device::new_metal(0)?)
            }
            #[cfg(not(feature = "metal"))]
            {
                println!("Running on CPU, to run on GPU, build with --features cuda or --features metal");
                Ok(Device::Cpu)
            }
        }
    }
}

// Audio utilities for real-time processing
pub mod audio {
    use anyhow::Result;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    /// Simulated audio capture from microphone
    /// In production, use cpal or another audio library
    pub fn capture_audio_from_microphone(duration_secs: u32) -> Result<Vec<f32>> {
        println!("Capturing audio for {} seconds...", duration_secs);

        // This is a placeholder - in production, use cpal or similar
        // Example with cpal:

        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        let host = cpal::default_host();
        let device = host.default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No input device available"))?;

        let config = device.default_input_config()?;
        let sample_rate = config.sample_rate().0;

        let (tx, rx) = mpsc::channel();

        let stream = device.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &_| {
                tx.send(data.to_vec()).unwrap();
            },
            |err| eprintln!("Error in audio stream: {}", err),
            None,
        )?;

        stream.play()?;
        thread::sleep(Duration::from_secs(duration_secs as u64));
        drop(stream);

        let mut all_samples = Vec::new();
        while let Ok(samples) = rx.try_recv() {
            all_samples.extend_from_slice(&samples);
        }

        // Resample to 24kHz if needed
        if sample_rate != 24000 {
            all_samples = kaudio::resample(&all_samples, sample_rate as usize, 24000)?;
        }

        Ok(all_samples)
    }

    /// Play audio through speakers
    /// In production, use cpal or another audio library
    pub fn play_audio(audio: &[f32], sample_rate: u32) -> Result<()> {
        println!("Playing audio: {} samples at {} Hz", audio.len(), sample_rate);

        // This is a placeholder - in production, use cpal or similar
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        let host = cpal::default_host();
        let device = host.default_output_device()
            .ok_or_else(|| anyhow::anyhow!("No output device available"))?;

        let config = cpal::StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let audio = audio.to_vec();
        let audio_len = audio.len();
        let mut sample_index = 0;

        let stream = device.build_output_stream(
            &config,
            move |data: &mut [f32], _: &_| {
                for sample in data.iter_mut() {
                    if sample_index < audio.len() {
                        *sample = audio[sample_index];
                        sample_index += 1;
                    } else {
                        *sample = 0.0;
                    }
                }
            },
            |err| eprintln!("Error in audio playback: {}", err),
            None,
        )?;

        stream.play()?;

        // Wait for playback to complete
        let duration = Duration::from_secs_f32(audio_len as f32 / sample_rate as f32);
        thread::sleep(duration);

        Ok(())
    }

    /// Play audio chunks in real-time (for streaming)
    pub fn create_audio_player(sample_rate: u32) -> Result<AudioPlayer> {
        Ok(AudioPlayer::new(sample_rate))
    }

    pub struct AudioPlayer {
        sample_rate: u32,
        tx: Option<mpsc::Sender<Vec<f32>>>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl AudioPlayer {
        pub fn new(sample_rate: u32) -> Self {
            Self {
                sample_rate,
                tx: None,
                handle: None,
            }
        }

        pub fn start(&mut self) -> Result<()> {
            let (tx, rx) = mpsc::channel::<Vec<f32>>();
            let sample_rate = self.sample_rate;

            let handle = thread::spawn(move || {
                while let Ok(chunk) = rx.recv() {
                    // In production, play the chunk through speakers
                    println!("Playing chunk: {} samples", chunk.len());

                    // Simulate playback delay
                    let duration = Duration::from_secs_f32(
                        chunk.len() as f32 / sample_rate as f32
                    );
                    thread::sleep(duration);
                }
            });

            self.tx = Some(tx);
            self.handle = Some(handle);

            Ok(())
        }

        pub fn play_chunk(&self, chunk: Vec<f32>) -> Result<()> {
            if let Some(tx) = &self.tx {
                tx.send(chunk)?;
            }
            Ok(())
        }

        pub fn stop(mut self) -> Result<()> {
            drop(self.tx.take());
            if let Some(handle) = self.handle.take() {
                handle.join().unwrap();
            }
            Ok(())
        }
    }
}