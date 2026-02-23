use autoagents_speech::AudioData;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Capture audio from microphone for a specified duration
pub(crate) fn capture_audio_from_mic(duration_secs: u32) -> Result<AudioData, Box<dyn std::error::Error>> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("No input device available")?;

    // Use default input config and convert to our needs
    let default_config = device.default_input_config()?;
    let sample_rate = default_config.sample_rate().0;
    let channels = default_config.channels();

    let samples = Arc::new(Mutex::new(Vec::new()));
    let samples_clone = samples.clone();

    let stream = match default_config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &default_config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                samples_clone.lock().unwrap().extend_from_slice(data);
            },
            |err| eprintln!("Audio stream error: {}", err),
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &default_config.into(),
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                let float_samples: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                samples_clone.lock().unwrap().extend(float_samples);
            },
            |err| eprintln!("Audio stream error: {}", err),
            None,
        )?,
        _ => return Err("Unsupported sample format".into()),
    };

    stream.play()?;
    std::thread::sleep(Duration::from_secs(duration_secs as u64));
    drop(stream);

    let mut samples = Arc::try_unwrap(samples)
        .unwrap()
        .into_inner()
        .unwrap();

    // Convert to mono if needed
    if channels > 1 {
        let mono_samples: Vec<f32> = samples
            .chunks(channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect();
        samples = mono_samples;
    }

    // Resample to 16kHz if needed (simple linear interpolation)
    if sample_rate != 16000 {
        let ratio = sample_rate as f32 / 16000.0;
        let target_len = (samples.len() as f32 / ratio) as usize;
        let mut resampled = Vec::with_capacity(target_len);
        
        for i in 0..target_len {
            let src_pos = i as f32 * ratio;
            let src_idx = src_pos as usize;
            
            if src_idx + 1 < samples.len() {
                let frac = src_pos - src_idx as f32;
                let interpolated = samples[src_idx] * (1.0 - frac) + samples[src_idx + 1] * frac;
                resampled.push(interpolated);
            } else if src_idx < samples.len() {
                resampled.push(samples[src_idx]);
            }
        }
        samples = resampled;
    }

    // CRITICAL: Normalize audio (required for models to work properly)
    // This matches what parakeet-rs examples do
    let max_val = samples.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    if max_val > 1e-6 {
        for s in &mut samples {
            *s /= max_val + 1e-5;
        }
    }

    Ok(AudioData {
        samples,
        sample_rate: 16000,
        channels: 1,
    })
}

/// Load audio samples from a WAV file
pub(crate) fn load_audio_from_wav(path: &str) -> Result<AudioData, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    // Convert samples to f32 normalized to [-1.0, 1.0]
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => match spec.bits_per_sample {
            16 => reader
                .samples::<i16>()
                .map(|s| s.unwrap() as f32 / 32768.0)
                .collect(),
            32 => reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / 2147483648.0)
                .collect(),
            _ => {
                return Err(format!("Unsupported bit depth: {}", spec.bits_per_sample).into());
            }
        },
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
    };

    Ok(AudioData {
        samples,
        sample_rate: spec.sample_rate,
        channels: spec.channels as usize,
    })
}

/// Save audio samples to a WAV file
pub(crate) fn save_audio_to_file(
    samples: &[f32],
    sample_rate: u32,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;

    for &sample in samples {
        writer.write_sample(sample)?;
    }

    writer.finalize()?;
    Ok(())
}
