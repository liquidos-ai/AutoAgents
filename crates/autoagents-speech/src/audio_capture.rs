use crate::AudioData;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Configuration for capturing and preparing audio samples.
#[derive(Debug, Clone)]
pub struct AudioCaptureConfig {
    target_sample_rate: u32,
    target_channels: usize,
    normalize: bool,
}

impl AudioCaptureConfig {
    pub fn new(target_sample_rate: u32, target_channels: usize) -> Self {
        Self {
            target_sample_rate,
            target_channels,
            normalize: true,
        }
    }

    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    pub fn target_sample_rate(&self) -> u32 {
        self.target_sample_rate
    }

    pub fn target_channels(&self) -> usize {
        self.target_channels
    }

    pub fn normalize(&self) -> bool {
        self.normalize
    }
}

impl Default for AudioCaptureConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: 16_000,
            target_channels: 1,
            normalize: true,
        }
    }
}

/// Errors that can occur while capturing or processing audio input.
#[derive(Debug, thiserror::Error)]
pub enum AudioCaptureError {
    #[error("No input audio device is available")]
    NoInputDevice,
    #[error("Invalid target configuration: sample_rate={sample_rate}, channels={channels}")]
    InvalidTargetConfig { sample_rate: u32, channels: usize },
    #[error("Failed to query default input configuration: {0}")]
    DefaultInputConfig(#[from] cpal::DefaultStreamConfigError),
    #[error("Failed to build audio input stream: {0}")]
    BuildStream(#[from] cpal::BuildStreamError),
    #[error("Failed to start audio stream: {0}")]
    PlayStream(#[from] cpal::PlayStreamError),
    #[error("Unsupported sample format: {0:?}")]
    UnsupportedSampleFormat(cpal::SampleFormat),
    #[error(
        "Unsupported channel layout: input has {input} channel(s), target requires {target} channel(s)"
    )]
    UnsupportedChannelLayout { input: usize, target: usize },
    #[error("Audio stream reported an error: {0}")]
    StreamError(String),
    #[error("Failed to read WAV file: {0}")]
    WavError(#[from] hound::Error),
    #[error("Unsupported WAV format: {0}")]
    UnsupportedWavFormat(String),
    #[error("Failed to decode audio file: {0}")]
    AudioDecodeError(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type AudioCaptureResult<T> = Result<T, AudioCaptureError>;

/// High-level audio capture helper built on top of `cpal`.
pub struct AudioCapture {
    device: cpal::Device,
    input_config: cpal::SupportedStreamConfig,
    target_config: AudioCaptureConfig,
}

impl AudioCapture {
    pub fn new() -> AudioCaptureResult<Self> {
        Self::with_config(AudioCaptureConfig::default())
    }

    pub fn with_config(target_config: AudioCaptureConfig) -> AudioCaptureResult<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or(AudioCaptureError::NoInputDevice)?;
        Self::with_device(device, target_config)
    }

    pub fn with_device(
        device: cpal::Device,
        target_config: AudioCaptureConfig,
    ) -> AudioCaptureResult<Self> {
        if target_config.target_sample_rate == 0 || target_config.target_channels == 0 {
            return Err(AudioCaptureError::InvalidTargetConfig {
                sample_rate: target_config.target_sample_rate,
                channels: target_config.target_channels,
            });
        }

        let input_config = device.default_input_config()?;
        Ok(Self {
            device,
            input_config,
            target_config,
        })
    }

    pub fn target_config(&self) -> &AudioCaptureConfig {
        &self.target_config
    }

    /// Capture microphone audio for a fixed duration.
    pub fn capture_for(&self, duration: Duration) -> AudioCaptureResult<AudioData> {
        let input_sample_rate = self.input_config.sample_rate();
        let input_channels = self.input_config.channels() as usize;

        let expected_samples =
            (duration.as_secs_f32() * input_sample_rate as f32) as usize * input_channels;
        let buffer = Arc::new(Mutex::new(Vec::with_capacity(expected_samples)));
        let stream_error = Arc::new(Mutex::new(None::<String>));
        {
            let stream = self.build_stream(buffer.clone(), stream_error.clone())?;
            stream.play()?;
            std::thread::sleep(duration);
        }

        if let Some(message) = stream_error.lock().ok().and_then(|e| e.clone()) {
            return Err(AudioCaptureError::StreamError(message));
        }

        let mut samples = Arc::try_unwrap(buffer)
            .map_err(|_| AudioCaptureError::StreamError("capture buffer still in use".to_string()))?
            .into_inner()
            .map_err(|_| {
                AudioCaptureError::StreamError("capture buffer mutex poisoned".to_string())
            })?;

        if samples.is_empty() {
            return Ok(AudioData {
                samples,
                sample_rate: self.target_config.target_sample_rate,
                channels: self.target_config.target_channels,
            });
        }

        samples = apply_target_config(
            samples,
            input_sample_rate,
            input_channels,
            &self.target_config,
        )?;

        Ok(AudioData {
            samples,
            sample_rate: self.target_config.target_sample_rate,
            channels: self.target_config.target_channels,
        })
    }

    /// Start a continuous capture stream.
    pub fn start_stream(&self) -> AudioCaptureResult<AudioCaptureStream> {
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let stream_error = Arc::new(Mutex::new(None::<String>));
        let stream = self.build_stream(buffer.clone(), stream_error.clone())?;

        stream.play()?;

        Ok(AudioCaptureStream {
            _stream: stream,
            buffer,
            input_sample_rate: self.input_config.sample_rate(),
            input_channels: self.input_config.channels() as usize,
            target_config: self.target_config.clone(),
            last_error: stream_error,
        })
    }

    /// Read a WAV file without resampling or normalization.
    pub fn read_wav(path: impl AsRef<Path>) -> AudioCaptureResult<AudioData> {
        let (samples, sample_rate, channels) = read_wav_samples(path.as_ref())?;
        Ok(AudioData {
            samples,
            sample_rate,
            channels,
        })
    }

    /// Read a WAV file and convert it to the target configuration.
    pub fn read_wav_with_config(
        path: impl AsRef<Path>,
        target_config: AudioCaptureConfig,
    ) -> AudioCaptureResult<AudioData> {
        let (samples, sample_rate, channels) = read_wav_samples(path.as_ref())?;
        let processed = apply_target_config(samples, sample_rate, channels, &target_config)?;

        Ok(AudioData {
            samples: processed,
            sample_rate: target_config.target_sample_rate,
            channels: target_config.target_channels,
        })
    }

    /// Read any supported audio file (WAV, FLAC, OGG, MP3) without resampling or normalization.
    pub fn read_audio(path: impl AsRef<Path>) -> AudioCaptureResult<AudioData> {
        let (samples, sample_rate, channels) = read_audio_samples(path.as_ref())?;
        Ok(AudioData {
            samples,
            sample_rate,
            channels,
        })
    }

    /// Read any supported audio file and convert it to the target configuration.
    pub fn read_audio_with_config(
        path: impl AsRef<Path>,
        target_config: AudioCaptureConfig,
    ) -> AudioCaptureResult<AudioData> {
        let (samples, sample_rate, channels) = read_audio_samples(path.as_ref())?;
        let processed = apply_target_config(samples, sample_rate, channels, &target_config)?;
        Ok(AudioData {
            samples: processed,
            sample_rate: target_config.target_sample_rate,
            channels: target_config.target_channels,
        })
    }

    fn build_stream(
        &self,
        buffer: Arc<Mutex<Vec<f32>>>,
        stream_error: Arc<Mutex<Option<String>>>,
    ) -> AudioCaptureResult<cpal::Stream> {
        let config: cpal::StreamConfig = self.input_config.clone().into();
        let sample_format = self.input_config.sample_format();
        let error_callback = move |err: cpal::StreamError| {
            if let Ok(mut slot) = stream_error.lock() {
                *slot = Some(err.to_string());
            }
        };

        match sample_format {
            cpal::SampleFormat::F32 => Ok(self.device.build_input_stream(
                &config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if let Ok(mut samples) = buffer.lock() {
                        samples.extend_from_slice(data);
                    }
                },
                error_callback,
                None,
            )?),
            cpal::SampleFormat::I16 => Ok(self.device.build_input_stream(
                &config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    if let Ok(mut samples) = buffer.lock() {
                        samples.extend(data.iter().map(|&s| s as f32 / i16::MAX as f32));
                    }
                },
                error_callback,
                None,
            )?),
            cpal::SampleFormat::U16 => Ok(self.device.build_input_stream(
                &config,
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    if let Ok(mut samples) = buffer.lock() {
                        samples.extend(
                            data.iter()
                                .map(|&s| (s as f32 / u16::MAX as f32) * 2.0 - 1.0),
                        );
                    }
                },
                error_callback,
                None,
            )?),
            _ => Err(AudioCaptureError::UnsupportedSampleFormat(sample_format)),
        }
    }
}

/// Live audio capture stream with buffered, on-demand chunk reads.
pub struct AudioCaptureStream {
    _stream: cpal::Stream,
    buffer: Arc<Mutex<Vec<f32>>>,
    input_sample_rate: u32,
    input_channels: usize,
    target_config: AudioCaptureConfig,
    last_error: Arc<Mutex<Option<String>>>,
}

impl AudioCaptureStream {
    /// Attempt to read a processed audio chunk.
    ///
    /// Returns `Ok(None)` when there is not enough buffered audio yet.
    pub fn read_chunk(&self, target_samples: usize) -> AudioCaptureResult<Option<AudioData>> {
        if let Some(message) = self.last_error.lock().ok().and_then(|e| e.clone()) {
            return Err(AudioCaptureError::StreamError(message));
        }

        let input_frames_needed = ((target_samples as f32)
            * (self.input_sample_rate as f32 / self.target_config.target_sample_rate as f32))
            .ceil() as usize;
        let raw_samples_needed = input_frames_needed * self.input_channels;

        let mut buffer = self.buffer.lock().map_err(|_| {
            AudioCaptureError::StreamError("capture buffer mutex poisoned".to_string())
        })?;

        if buffer.len() < raw_samples_needed {
            return Ok(None);
        }

        let raw: Vec<f32> = buffer.drain(..raw_samples_needed).collect();
        drop(buffer);

        let mut processed = apply_target_config(
            raw,
            self.input_sample_rate,
            self.input_channels,
            &self.target_config,
        )?;

        let target_len = target_samples * self.target_config.target_channels;
        if processed.len() < target_len {
            processed.resize(target_len, 0.0);
        } else if processed.len() > target_len {
            processed.truncate(target_len);
        }

        Ok(Some(AudioData {
            samples: processed,
            sample_rate: self.target_config.target_sample_rate,
            channels: self.target_config.target_channels,
        }))
    }

    pub fn input_sample_rate(&self) -> u32 {
        self.input_sample_rate
    }

    pub fn input_channels(&self) -> usize {
        self.input_channels
    }

    /// Drop any buffered samples that have not yet been consumed.
    pub fn clear_buffer(&self) -> AudioCaptureResult<()> {
        let mut buffer = self.buffer.lock().map_err(|_| {
            AudioCaptureError::StreamError("capture buffer mutex poisoned".to_string())
        })?;
        buffer.clear();
        Ok(())
    }
}

/// Dispatch to `read_wav_samples` for `.wav` files, or to the symphonia decoder for everything else.
fn read_audio_samples(path: &Path) -> AudioCaptureResult<(Vec<f32>, u32, usize)> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    if ext == "wav" {
        return read_wav_samples(path);
    }

    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error as SymphoniaError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let src = std::fs::File::open(path).map_err(AudioCaptureError::Io)?;
    let mss = MediaSourceStream::new(Box::new(src), MediaSourceStreamOptions::default());

    let mut hint = Hint::new();
    hint.with_extension(&ext);

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| AudioCaptureError::AudioDecodeError(e.to_string()))?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| AudioCaptureError::AudioDecodeError("no audio track found".into()))?;

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| AudioCaptureError::AudioDecodeError("unknown sample rate".into()))?;
    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);
    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| AudioCaptureError::AudioDecodeError(e.to_string()))?;

    let mut sample_buf: Option<SampleBuffer<f32>> = None;
    let mut samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(_)) | Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(AudioCaptureError::AudioDecodeError(e.to_string())),
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let sb = sample_buf.get_or_insert_with(|| {
                    SampleBuffer::new(decoded.capacity() as u64, *decoded.spec())
                });
                sb.copy_interleaved_ref(decoded);
                samples.extend_from_slice(sb.samples());
            }
            Err(SymphoniaError::IoError(_)) | Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(AudioCaptureError::AudioDecodeError(e.to_string())),
        }
    }

    Ok((samples, sample_rate, channels))
}

fn read_wav_samples(path: &Path) -> AudioCaptureResult<(Vec<f32>, u32, usize)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => match spec.bits_per_sample {
            16 => reader
                .samples::<i16>()
                .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
                .collect::<Result<_, _>>()?,
            32 => reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / i32::MAX as f32))
                .collect::<Result<_, _>>()?,
            bits => {
                return Err(AudioCaptureError::UnsupportedWavFormat(format!(
                    "unsupported bit depth: {bits}"
                )));
            }
        },
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
    };

    Ok((samples, sample_rate, channels))
}

fn apply_target_config(
    mut samples: Vec<f32>,
    input_sample_rate: u32,
    input_channels: usize,
    target: &AudioCaptureConfig,
) -> AudioCaptureResult<Vec<f32>> {
    if target.target_channels == 1 && input_channels > 1 {
        samples = downmix_to_mono(&samples, input_channels);
    } else if target.target_channels != input_channels {
        return Err(AudioCaptureError::UnsupportedChannelLayout {
            input: input_channels,
            target: target.target_channels,
        });
    }

    let channel_count = target.target_channels.max(1);

    if input_sample_rate != target.target_sample_rate {
        samples = resample_interleaved(
            &samples,
            channel_count,
            input_sample_rate,
            target.target_sample_rate,
        );
    }

    if target.normalize {
        normalize_audio(&mut samples);
    }

    Ok(samples)
}

fn downmix_to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    samples
        .chunks(channels)
        .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
        .collect()
}

fn resample_interleaved(
    samples: &[f32],
    channels: usize,
    from_rate: u32,
    to_rate: u32,
) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }

    let frame_count = samples.len() / channels;
    if frame_count == 0 {
        return Vec::new();
    }

    let ratio = from_rate as f32 / to_rate as f32;
    let target_frames = (frame_count as f32 / ratio).ceil() as usize;
    let mut out = Vec::with_capacity(target_frames * channels);

    for i in 0..target_frames {
        let src_pos = i as f32 * ratio;
        let src_idx = src_pos.floor() as usize;
        let frac = src_pos - src_idx as f32;

        for ch in 0..channels {
            let base = src_idx * channels + ch;
            let next = base + channels;

            let sample = if next < samples.len() {
                samples[base] * (1.0 - frac) + samples[next] * frac
            } else if base < samples.len() {
                samples[base]
            } else {
                0.0
            };
            out.push(sample);
        }
    }

    out
}

fn normalize_audio(samples: &mut [f32]) {
    let max_val = samples.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));

    if max_val > 1e-5 {
        let scale = 1.0 / (max_val + 1e-5);
        for sample in samples.iter_mut() {
            *sample *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::{SampleFormat, WavSpec, WavWriter};
    use tempfile::tempdir;

    fn write_i16_wav(
        path: &Path,
        sample_rate: u32,
        channels: u16,
        samples: &[i16],
    ) -> Result<(), hound::Error> {
        let spec = WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(path, spec)?;
        for sample in samples {
            writer.write_sample(*sample)?;
        }
        writer.finalize()
    }

    fn write_i32_wav(
        path: &Path,
        sample_rate: u32,
        channels: u16,
        samples: &[i32],
    ) -> Result<(), hound::Error> {
        let spec = WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(path, spec)?;
        for sample in samples {
            writer.write_sample(*sample)?;
        }
        writer.finalize()
    }

    fn write_f32_wav(
        path: &Path,
        sample_rate: u32,
        channels: u16,
        samples: &[f32],
    ) -> Result<(), hound::Error> {
        let spec = WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut writer = WavWriter::create(path, spec)?;
        for sample in samples {
            writer.write_sample(*sample)?;
        }
        writer.finalize()
    }

    #[test]
    fn audio_capture_config_helpers_round_trip() {
        let config = AudioCaptureConfig::new(22_050, 2).with_normalize(false);
        assert_eq!(config.target_sample_rate(), 22_050);
        assert_eq!(config.target_channels(), 2);
        assert!(!config.normalize());

        let default = AudioCaptureConfig::default();
        assert_eq!(default.target_sample_rate(), 16_000);
        assert_eq!(default.target_channels(), 1);
        assert!(default.normalize());
    }

    #[test]
    fn read_wav_and_read_audio_dispatch_for_i16() {
        let dir = tempdir().expect("tempdir should build");
        let path = dir.path().join("stereo.wav");
        write_i16_wav(
            &path,
            8_000,
            2,
            &[i16::MIN / 2, i16::MAX / 2, 0, i16::MAX / 4],
        )
        .expect("wav should write");

        let wav = AudioCapture::read_wav(&path).expect("wav should decode");
        assert_eq!(wav.sample_rate, 8_000);
        assert_eq!(wav.channels, 2);
        assert_eq!(wav.samples.len(), 4);
        assert!(wav.samples[0] < 0.0);
        assert!(wav.samples[1] > 0.0);

        let audio = AudioCapture::read_audio(&path).expect("wav dispatch should work");
        assert_eq!(audio.sample_rate, wav.sample_rate);
        assert_eq!(audio.channels, wav.channels);
        assert_eq!(audio.samples.len(), wav.samples.len());
    }

    #[test]
    fn read_wav_supports_i32_and_f32_formats() {
        let dir = tempdir().expect("tempdir should build");
        let int_path = dir.path().join("int32.wav");
        let float_path = dir.path().join("float.wav");

        write_i32_wav(&int_path, 16_000, 1, &[i32::MIN / 4, i32::MAX / 4, 0])
            .expect("int32 wav should write");
        write_f32_wav(&float_path, 16_000, 1, &[0.25, -0.5, 0.75]).expect("float wav should write");

        let int_audio = AudioCapture::read_wav(&int_path).expect("int32 wav should decode");
        assert_eq!(int_audio.sample_rate, 16_000);
        assert_eq!(int_audio.channels, 1);
        assert_eq!(int_audio.samples.len(), 3);
        assert!(int_audio.samples[0] < 0.0);
        assert!(int_audio.samples[1] > 0.0);

        let float_audio = AudioCapture::read_wav(&float_path).expect("float wav should decode");
        assert_eq!(float_audio.samples, vec![0.25, -0.5, 0.75]);
    }

    #[test]
    fn read_wav_with_config_downmixes_resamples_and_normalizes() {
        let dir = tempdir().expect("tempdir should build");
        let path = dir.path().join("convert.wav");
        write_i16_wav(&path, 8_000, 2, &[0, 0, i16::MAX / 2, i16::MAX]).expect("wav should write");

        let target = AudioCaptureConfig::new(16_000, 1);
        let audio =
            AudioCapture::read_wav_with_config(&path, target).expect("wav should be converted");

        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.samples.len(), 4);
        let max = audio
            .samples
            .iter()
            .fold(0.0_f32, |acc, sample| acc.max(sample.abs()));
        assert!(max > 0.99 && max <= 1.0);
        assert!(audio.samples[0].abs() < audio.samples[1].abs());
    }

    #[test]
    fn read_wav_rejects_unsupported_bit_depth() {
        let dir = tempdir().expect("tempdir should build");
        let path = dir.path().join("unsupported.wav");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 8,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(&path, spec).expect("wav should create");
        writer.write_sample(7_i8).expect("sample should write");
        writer.finalize().expect("wav should finalize");

        let err = AudioCapture::read_wav(&path).expect_err("8-bit wav should be rejected");
        assert!(matches!(
            err,
            AudioCaptureError::UnsupportedWavFormat(message)
            if message.contains("unsupported bit depth")
        ));
    }

    #[test]
    fn read_audio_non_wav_missing_file_bubbles_io_error() {
        let err = AudioCapture::read_audio(Path::new("definitely_missing.mp3"))
            .expect_err("missing file");
        assert!(matches!(err, AudioCaptureError::Io(_)));
    }

    #[test]
    fn apply_target_config_rejects_unsupported_channel_layout() {
        let target = AudioCaptureConfig::new(16_000, 2);
        let err = apply_target_config(vec![0.1, 0.2, 0.3], 16_000, 1, &target).expect_err("layout");

        assert!(matches!(
            err,
            AudioCaptureError::UnsupportedChannelLayout {
                input: 1,
                target: 2
            }
        ));
    }

    #[test]
    fn downmix_and_resample_helpers_cover_edge_cases() {
        let mono = downmix_to_mono(&[1.0, 3.0, 2.0, 4.0], 2);
        assert_eq!(mono, vec![2.0, 3.0]);

        let same_rate = resample_interleaved(&[0.0, 1.0, 2.0, 3.0], 2, 16_000, 16_000);
        assert_eq!(same_rate, vec![0.0, 1.0, 2.0, 3.0]);

        let empty = resample_interleaved(&[], 1, 8_000, 16_000);
        assert!(empty.is_empty());

        let resampled = resample_interleaved(&[0.0, 1.0], 1, 8_000, 16_000);
        assert_eq!(resampled.len(), 4);
        assert!(resampled[1] > 0.0 && resampled[1] < 1.0);
        assert_eq!(resampled[3], 1.0);
    }

    #[test]
    fn normalize_audio_scales_and_ignores_near_zero_inputs() {
        let mut silent = vec![1e-6, -5e-6];
        normalize_audio(&mut silent);
        assert_eq!(silent, vec![1e-6, -5e-6]);

        let mut loud = vec![0.25, -0.5];
        normalize_audio(&mut loud);
        assert!(loud[0] > 0.49);
        assert!(loud[1] < -0.99);
    }
}
